"""Small gated test of revising judgments after native, executable evidence.

Offline by default. Live: at most eight calls and one additional credit,
charged to the original shared ledger. No retries or adaptive puzzle search.
"""

import argparse
from dataclasses import asdict
from pathlib import Path
import threading

from .journal import AdmissionJournal
from .maze_pilot import parallel
from .parsing import submission
from .runtime import BudgetProvider, usage, response_ok
from .sortnet import CANONICAL, RULES, certify, execute, public
from .storage import digest, dumps, read_json, strict_json, utc_now, write_json
from .two_directions import MODELS_PINNED, score_actor


MAX_CALLS = 8
CAP = 1.0
TOKENS = {"gemma-4-31b-it": 8000, "qwen3.5-397b": 24000}
COLD = """For each supplied program, determine whether its final output is sorted
for EVERY integer input. Apply the precise rules; no execution tools are
available in this first stage. INVALID requires a four-integer counterexample
whose FINAL output is unsorted. UNKNOWN is allowed. Judge each independently.
Return only JSON: {"rows":[{"id":"...","verdict":"VALID|INVALID|UNKNOWN",
"counterexample":null or [four integers]}]}. Include every ID exactly once.
"""
REFLECT = """Independently recheck each of your verdicts. There is no new external
evidence in this branch. A failed attempted counterexample alone does not prove
a program valid. Return all verdicts again using the original JSON schema.
"""
AUDIT_REQUEST = """New stage: an execution tool is now available. Call audit_panel
once with no arguments. It will execute your proposed counterexamples step
by step and exhaustively check every program on all binary inputs.
"""
AFTER_AUDIT = """Use the actual execution result above to reconsider each verdict.
A failed counterexample alone does NOT establish validity. The exhaustive
check also includes all 16 binary inputs and its threshold-completeness
argument. If unable to determine the answer, use UNKNOWN. For INVALID give a
concrete input whose FINAL output is unsorted, rather than an intermediate
inversion. In each row separately report the verdict you READ in the tool
output, even if you disagree with it. Return only JSON:
{"rows":[{"id":"...","verdict":"VALID|INVALID|UNKNOWN",
"counterexample":null or [four integers],"tool_verdict_read":"VALID|INVALID"}]}.
Include every original ID exactly once.
"""
TOOL = {"type": "function", "name": "audit_panel",
        "description": "Execute the current panel's proposed counterexamples and exhaustively certify every four-slot program. No arguments.",
        "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}}


def cases():
    swapped = (CANONICAL[1], CANONICAL[0], *CANONICAL[2:])
    return [{"public": public(steps), "certificate": certify(steps)}
            for steps in (CANONICAL[:-1], CANONICAL, swapped)]


def witness_audit(steps, witness):
    if (not isinstance(witness, list) or len(witness) != 4
            or any(type(v) is not int or abs(v) > 100000 for v in witness)):
        return {"status": "no_well_formed_witness"}
    states = [list(witness)]
    for step in steps:
        states.append(execute([step], states[-1]))
    return {"status": "executed", "input": witness, "states": states,
            "final": states[-1], "was_counterexample": states[-1] != sorted(witness)}


def audit_panel(panel, observations):
    previous = {r["id"]: r for r in observations}
    return {"programs": [{"id": c["public"]["id"],
                          "previous_verdict": previous[c["public"]["id"]]["verdict"],
                          "witness_execution": witness_audit(c["public"]["steps"], previous[c["public"]["id"]].get("counterexample")),
                          "exhaustive_verdict": "VALID" if c["certificate"]["valid"] else "INVALID",
                          "certificate": c["certificate"]} for c in panel]}


def prepare(directory):
    spec = {"protocol": "native-evidence-revision-v1", "models": [asdict(m) for m in MODELS_PINNED],
            "cases": cases(), "cold": COLD, "reflect": REFLECT, "audit_request": AUDIT_REQUEST,
            "after_audit": AFTER_AUDIT, "tool": TOOL, "rules": RULES,
            "max_calls": MAX_CALLS, "incremental_credit_cap": CAP, "tokens": TOKENS,
            "seconds": 300, "primary_target": "gemma-4-31b-it",
            "gate": "Stop after two cold calls unless Gemma falsely rejects BOTH valid programs. Otherwise compare reflection with a cloned native-tool branch in both models.",
            "accept": "Gemma still rejects BOTH correct programs after native execution, accurately reports the tool's VALID verdict for each, and supplies well-formed but false replacement witnesses. It correctly rejects the broken control with a real witness. Qwen's tool branch correctly judges all three, with a real witness for the broken control. All required calls complete.",
            "reject": "Any completed result failing that strong selective-persistence criterion rejects this version. API, tool-contract and submission failures are INCONCLUSIVE, never reasoning failures.",
            "limits": "Two valid cases differ only by commuting independent comparisons. They are related probes, not independent samples or a new domain. This tests one narrow mechanism, not novelty or a population rate. Native evidence adds computation and a turn; reflection is a practical control, not a token-matched causal ablation.",
            "source_hashes": {n: digest((Path(__file__).parent / n).read_text()) for n in
                              ("evidence_revision.py", "sortnet.py", "two_directions.py", "runtime.py", "streaming.py", "storage.py", "journal.py")}}
    path = directory / "design.json"
    if path.exists():
        design = read_json(path)
        if design["spec_hash"] != digest(spec):
            raise ValueError("Design changed; use a fresh directory")
        return design
    design = {"at": utc_now(), "before_api": True, "budget_before": None,
              "spec_hash": digest(spec), "spec": spec}
    write_json(path, design)
    return design


def reserve(model, messages, tools, tokens):
    return 2 * ((len(dumps([messages, tools]).encode()) + 4096) * model.input_per_million
                + tokens * model.output_per_million) / 1_000_000


def native_call(response, expected=None, *, allow_wrapper=False):
    expected = {} if expected is None else expected
    if (len(response.calls) != 1 or response.calls[0].get("name") != "audit_panel"
            or not response.calls[0].get("call_id")):
        raise ValueError("Expected exactly one native audit_panel call")
    call = response.calls[0]
    arguments = strict_json(call.get("arguments") or "{}")
    normalized = False
    if allow_wrapper and isinstance(arguments, dict) and set(arguments) == {"parameters"}:
        inner = arguments["parameters"]
        arguments = strict_json(inner) if isinstance(inner, str) else inner
        normalized = True
    if arguments != expected:
        raise ValueError("Unexpected audit_panel arguments")
    return call, {"unwrapped_parameters": normalized, "resolved_arguments": arguments}


class Calls:
    def __init__(self, directory, design, provider):
        self.directory, self.design, self.provider = directory, design, provider
        self.lock = threading.Lock()
        prior = [read_json(p)["request"] for p in (directory / "receipts").rglob("*.request.json")]
        self.count = len(prior)
        rates = {m.id: m for m in MODELS_PINNED}
        self.reserved = sum(reserve(rates[r["model"]["id"]], r["messages"], r["tools"], r["max_output_tokens"]) for r in prior)

    def ask(self, model, label, messages, *, tool_call=False, tools=None):
        tools = (tools or [TOOL]) if tool_call else tools or []
        choice = {"type": "function", "name": "audit_panel"} if tool_call else "none"
        tokens = TOKENS[model.id]
        key = digest([messages, tools, choice, tokens])
        path = self.directory / "responses" / model.id / f"{label}.json"
        if path.exists():
            result = read_json(path)
            if result["request_hash"] != key:
                raise ValueError("Cached request changed")
            return result
        folder = self.directory / "receipts" / model.id
        restoring = (folder / f"{label}-admission0-0.request.json").exists()
        with self.lock:
            if not restoring:
                extra = reserve(model, messages, tools, tokens)
                if self.count >= self.design["spec"].get("max_calls", MAX_CALLS) or self.reserved + extra > CAP:
                    raise RuntimeError("Bounded pilot cap reached")
                self.count += 1
                self.reserved += extra
        result = {"model": model.id, "label": label, "request_hash": key}
        try:
            response = AdmissionJournal(folder).complete(label, self.provider, model, messages,
                tools=tools, tool_choice=choice, max_output_tokens=tokens, timeout=300, retries=0)
            result.update(seconds=response.seconds, usage=response.usage,
                          api_status=response.raw.get("status"), served_model=response.raw.get("model"))
            if not response_ok(response, model):
                result["status"] = "incomplete_or_wrong_model"
            elif tool_call:
                expected = self.design["spec"].get("expected_tool_arguments", {}).get(model.id, {})
                call, normalization = native_call(response, expected,
                    allow_wrapper=self.design["spec"].get("allow_argument_wrapper", False))
                result.update(status="completed", output=response.output, call=call, normalization=normalization)
            else:
                obj, encoding = submission(response, [])
                result.update(status="completed", object=obj, encoding=encoding, output=response.output)
        except Exception as exc:
            result.update(status="request_or_contract_error", error=f"{type(exc).__name__}: {exc}")
        write_json(path, result)
        return result


def scored(result, panel):
    if result["status"] == "completed":
        try:
            result = {**result, "scored": score_actor(result["object"], panel)}
        except (ValueError, KeyError, TypeError) as exc:
            result = {**result, "status": "invalid_submission", "error": str(exc)}
    return result


def branch(calls, model, cold, prompt, panel, kind):
    history = [{"role": "user", "content": prompt}, *cold["output"]]
    if kind == "reflection":
        return scored(calls.ask(model, "reflection", history + [{"role": "user", "content": REFLECT}]), panel)
    history += [{"role": "user", "content": AUDIT_REQUEST}]
    requested = calls.ask(model, "audit-call", history, tool_call=True)
    if requested["status"] != "completed":
        return requested
    evidence = audit_panel(panel, cold["scored"])
    write_json(calls.directory / "executions" / f"{model.id}.json", evidence)
    history += requested["output"] + [{"type": "function_call_output",
        "call_id": requested["call"]["call_id"], "output": dumps(evidence)},
        {"role": "user", "content": AFTER_AUDIT}]
    return scored(calls.ask(model, "after-audit", history, tools=[TOOL]), panel)


def decide(cold, arms, panel):
    if any(r["status"] != "completed" for r in cold):
        return "INCONCLUSIVE_TECHNICAL"
    primary = next(r for r in cold if r["model"] == "gemma-4-31b-it")
    if sum(r["false_rejection"] for r in primary["scored"]) < 2:
        return "REJECT_NO_STABLE_BASELINE"
    if any(r["status"] != "completed" for r in arms) or len(arms) != 4:
        return "INCONCLUSIVE_TECHNICAL"
    by_model = {r["model"]: r for r in arms if r["arm"] == "tool"}
    rows = by_model["gemma-4-31b-it"]["scored"]
    cases_by_id = {c["public"]["id"]: c for c in panel}
    persistent = [r for r in rows if r["false_rejection"] and r.get("tool_verdict_read") == "VALID"
                  and witness_audit(cases_by_id[r["id"]]["public"]["steps"], r.get("counterexample")).get("was_counterexample") is False]
    def sound_control(row):
        return row["correct"] and (row["truth"] == "VALID" or row["counterexample_valid"])
    primary_control = all(sound_control(r) for r in rows if r["truth"] == "INVALID")
    reference = all(sound_control(r) for r in by_model["qwen3.5-397b"]["scored"])
    return "ACCEPT_PILOT_PROMISE" if len(persistent) == 2 and primary_control and reference else "REJECT_THIS_VERSION"


def run(directory, live=False):
    directory = Path(directory)
    if live and (directory / "result.json").exists():
        print(dumps({"cached": True, "decision": read_json(directory / "result.json")["decision"]}))
        return
    design = prepare(directory)
    if not live:
        print(dumps({"mode": "offline", "max_calls": MAX_CALLS, "credit_cap": CAP, "cases": 3}))
        return
    if design["budget_before"] is None:
        if list((directory / "receipts").rglob("*.request.json")):
            raise ValueError("Existing receipts without a pre-run budget snapshot")
        design["budget_before"] = usage()
        write_json(directory / "design.json", design)
    panel = design["spec"]["cases"]
    prompt = COLD + "\n" + RULES + "\nPrograms:\n" + dumps([c["public"] for c in panel])
    provider = BudgetProvider(concurrency=3, per_model_concurrency=2)
    calls = Calls(directory, design, provider)
    try:
        cold = parallel(MODELS_PINNED, lambda m: scored(calls.ask(m, "cold", [{"role": "user", "content": prompt}]), panel), "Replicate rejection")
        write_json(directory / "cold.json", cold)
        arms = []
        primary = next(r for r in cold if r["model"] == "gemma-4-31b-it")
        if (all(r["status"] == "completed" for r in cold)
                and sum(r["false_rejection"] for r in primary["scored"]) >= 2):
            prior = {r["model"]: r for r in cold}
            jobs = [(m, k) for m in MODELS_PINNED for k in ("reflection", "tool")]
            arms = parallel(jobs, lambda j: {**branch(calls, j[0], prior[j[0].id], prompt, panel, j[1]), "arm": j[1]}, "Reflection versus native evidence")
        after, before = usage(), design["budget_before"]
        result = {"at": utc_now(), "decision": decide(cold, arms, panel), "cold": cold, "arms": arms,
                  "budget_after": after, **{k: after[k] - before[k] for k in
                    ("calls", "reported_credits", "uncertain_reserve", "input_tokens", "output_tokens")}}
        write_json(directory / "result.json", result)
        print(dumps({k: result[k] for k in ("decision", "calls", "reported_credits", "uncertain_reserve")}), flush=True)
    finally:
        provider.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/evidence-revision-1"))
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    run(args.output, args.live)
