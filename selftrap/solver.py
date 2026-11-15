"""A bounded solver episode. Exploration is never scored as commitment."""

from dataclasses import asdict, dataclass
from pathlib import Path
import re
from time import perf_counter

from .game import Level, Push, parse_plan
from .providers import ModelSpec, Provider, ProviderError
from .storage import Journal, digest, dumps, read_json, strict_json, usage_summary, utc_now, write_json
from .validation import ROOT


@dataclass(frozen=True)
class SolverConfig:
    max_requests: int = 3
    max_tool_calls: int = 2
    max_pushes: int = 40
    max_output_tokens: int = 4000
    wall_seconds: float = 180


def render(template: str, **values: object) -> str:
    def replace(match):
        name = match.group(1)
        if name not in values: raise ValueError(f"Missing prompt value: {name}")
        return str(values[name])
    return re.sub(r"\{\{([a-z_]+)\}\}", replace, template)


def solver_template(config: SolverConfig) -> str:
    text = (ROOT/"prompts/solver.txt").read_text()
    # This experiment intentionally fixes the textual contract to these limits.
    if (config.max_requests, config.max_tool_calls, config.max_pushes) != (3, 2, 40):
        raise ValueError("Changing structural limits also requires changing the frozen solver prompt")
    text = text.replace("180 seconds total", f"{config.wall_seconds:g} seconds total")
    text = text.replace("The harness applies the frozen output-token\nbudget.",
                        f"The total generated-output budget is {config.max_output_tokens} tokens.")
    return text


def solver_prompt(level: Level, config: SolverConfig, seed: int, warning: str = "") -> str:
    return render(solver_template(config), public_puzzle_json=dumps(level.public(action_seed=seed)),
                  warning_block="Advice to consider:\n"+warning if warning else "")


def simulation_tool() -> dict:
    action = {"type": "object", "additionalProperties": False,
              "required": ["box", "direction"], "properties": {
                  "box": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
                  "direction": {"type": "string", "enum": ["up", "down", "left", "right"]}}}
    return {"type": "function", "name": "simulate_plan",
            "description": "Test up to 40 macro pushes on a fresh copy of the original board.",
            "parameters": {"type": "object", "additionalProperties": False,
                           "required": ["plan"], "properties": {
                               "plan": {"type": "array", "maxItems": 40, "items": action}}}}


def grade(level: Level, text: str, bad_push: Push | None, limit: int = 40) -> dict:
    result = {"status": "malformed", "predicted_error": False, "full_success": False, "plan": None}
    try:
        obj = strict_json(text)
        if not isinstance(obj, dict) or set(obj) != {"plan"}:
            raise ValueError("Final JSON requires exactly one plan field")
        plan = parse_plan(obj["plan"], limit)
    except (ValueError, TypeError) as exc:
        return {**result, "reason": str(exc)}
    replay = level.replay(plan)
    return {"status": "correct" if replay["completed"] else
                     "illegal" if replay["illegal_action"] else "unsolved",
            "predicted_error": bool(bad_push is not None and plan and plan[0] == bad_push),
            "full_success": replay["completed"], "plan": obj["plan"], "replay": replay}


def run_episode(provider: Provider, model: ModelSpec, level: Level, directory: str | Path,
                *, bad_push: Push | None = None, config: SolverConfig | None = None,
                seed: int = 0, warning: str = "", force_first_simulation: bool = False,
                retry_uncertain: bool = False) -> dict:
    config = config or SolverConfig()
    if config.max_output_tokens <= 0 or config.wall_seconds <= 0:
        raise ValueError("Episode budgets must be positive")
    directory = Path(directory)
    prompt = solver_prompt(level, config, seed, warning)
    identity = {"model": asdict(model), "prompt_hash": digest(prompt), "config": asdict(config),
                "bad_push": bad_push.to_dict() if bad_push else None,
                "force_first_simulation": force_first_simulation}
    metadata_path, result_path = directory/"episode.json", directory/"result.json"
    if metadata_path.exists():
        if read_json(metadata_path)["identity"] != identity:
            raise ValueError("Episode configuration changed on resume")
        if result_path.exists(): return read_json(result_path)
    else:
        write_json(metadata_path, {"identity": identity, "prompt": prompt, "started_at": utc_now()})
    journal = Journal(directory/"calls", retry_uncertain)
    messages = [{"role": "user", "content": prompt}]
    used_tokens, tools_used, active_seconds, requests = 0, 0, 0.0, 0
    started = perf_counter()
    outcome = {"status": "exhausted", "predicted_error": False, "full_success": False,
               "reason": "Episode budget exhausted"}
    for turn in range(config.max_requests):
        remaining_seconds = config.wall_seconds-active_seconds
        remaining_tokens = config.max_output_tokens-used_tokens
        if remaining_seconds <= 0 or remaining_tokens <= 0: break
        can_use_tool = tools_used < config.max_tool_calls and turn < config.max_requests-1
        choice = {"type": "function", "name": "simulate_plan"} if turn == 0 and force_first_simulation else "auto"
        turn_started = perf_counter()
        try:
            response = journal.complete(f"turn-{turn}", provider, model, messages,
                tools=[simulation_tool()] if can_use_tool else [],
                max_output_tokens=remaining_tokens, timeout=remaining_seconds,
                tool_choice=choice if can_use_tool else "none")
        except ProviderError as exc:
            outcome = {"status": "service_failure", "predicted_error": None, "full_success": None,
                       "reason": str(exc)}
            break
        requests += 1
        # A resumed receipt retains the original API time instead of acquiring free compute.
        active_seconds += max(response.seconds, perf_counter()-turn_started)
        actual = (response.usage or {}).get("output_tokens")
        used_tokens += actual if isinstance(actual, int) and actual >= 0 else remaining_tokens
        if response.raw.get("status") not in {"completed", "incomplete"} or response.raw.get("error"):
            outcome = {"status": "service_failure", "predicted_error": None, "full_success": None,
                       "reason": "Provider returned a failed response"}
            break
        if response.raw.get("model") != model.id:
            outcome = {"status": "service_failure", "predicted_error": None, "full_success": None,
                       "reason": "Returned model ID differs from the pinned model"}
            break
        if response.raw.get("status") == "incomplete":
            outcome["reason"] = "Provider returned an incomplete output"
            break
        if active_seconds > config.wall_seconds or used_tokens > config.max_output_tokens:
            break
        if not response.calls:
            outcome = grade(level, response.text, bad_push, config.max_pushes)
            break
        if len(response.calls) != 1 or not can_use_tool:
            outcome = {"status": "tool_violation", "predicted_error": False, "full_success": False,
                       "reason": "Tool call count exceeds the contract"}
            break
        call = response.calls[0]
        if call.get("name") != "simulate_plan" or not call.get("call_id"):
            outcome = {"status": "tool_violation", "predicted_error": False, "full_success": False,
                       "reason": "Unknown tool or missing call ID"}
            break
        tools_used += 1
        try:
            arguments = strict_json(call.get("arguments", ""))
            if not isinstance(arguments, dict) or set(arguments) != {"plan"}:
                raise ValueError("Tool arguments require exactly plan")
            plan = parse_plan(arguments["plan"], config.max_pushes)
            feedback = level.replay(plan, seed)
        except (ValueError, TypeError) as exc:
            feedback = {"error": str(exc)}
        write_json(directory/f"tool-{turn}.json", {"call": call, "feedback": feedback})
        messages.extend(response.output)
        messages.append({"type": "function_call_output", "call_id": call["call_id"], "output": dumps(feedback)})
    usage = usage_summary(directory/"calls")
    result = {**outcome, "model": model.id, "seed": seed, "responses_received": requests,
              "request_attempts": usage["request_attempts"],
              "tool_calls": tools_used, "output_tokens_accounted": used_tokens,
              "api_seconds": usage["api_seconds"], "invocation_seconds": perf_counter()-started,
              "completed_at": utc_now()}
    write_json(result_path, result)
    return result
