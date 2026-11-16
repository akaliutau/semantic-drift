"""One declared native-tool compatibility follow-up; reuse all finished trials."""

import argparse
from pathlib import Path

from . import evidence_revision as base
from .providers import ModelResponse
from .runtime import BudgetProvider, response_ok, usage
from .storage import digest, dumps, read_json, utc_now, write_json


TOOL = {**base.TOOL, "parameters": {"type": "object", "properties": {
    "scope": {"type": "string", "enum": ["all"]}}, "required": ["scope"], "additionalProperties": False}}
REQUEST = "New stage: call audit_panel with scope set to all to execute your proposed counterexamples and exhaustively check every program."


def run(directory, previous, live=False):
    directory, previous = Path(directory), Path(previous)
    if live and (directory / "result.json").exists():
        print(dumps({"cached": True, "decision": read_json(directory / "result.json")["decision"]}))
        return
    old = read_json(previous / "result.json")
    old_design = read_json(previous / "design.json")
    if old["decision"] != "INCONCLUSIVE_TECHNICAL":
        raise ValueError("This follow-up only repairs an incomplete technical comparison")
    native_path = previous / "receipts/qwen3.5-397b/audit-call-admission0-0.response.json"
    native = read_json(native_path)
    response = ModelResponse(**native)
    qwen = next(m for m in base.MODELS_PINNED if m.id == "qwen3.5-397b")
    if not response_ok(response, qwen):
        raise ValueError("The previous Qwen native call did not complete")
    call, normalization = base.native_call(response, allow_wrapper=True)
    spec = {"protocol": "native-evidence-revision-compatibility-v1", "max_calls": 3,
            "incremental_credit_cap": 1.0, "expected_tool_arguments": {"gemma-4-31b-it": {"scope": "all"}},
            "allow_argument_wrapper": True, "retry_tool": TOOL, "retry_request": REQUEST,
            "old_design_hash": old_design["spec_hash"], "accept": old_design["spec"]["accept"],
            "references": {str(p.resolve()): digest(read_json(p)) for p in
                           (previous / "result.json", previous / "design.json", native_path)},
            "repair": "Preserve the first technical outcome. Reuse both cold and reflection responses. Continue Qwen's actual native call after losslessly decoding its sole parameters wrapper; preserve the original call in history. One Gemma retry adds a required scope=all argument to avoid the empty-schema generation loop. No retry of a wrong verdict.",
            "source_hashes": {n: digest((Path(__file__).parent / n).read_text()) for n in
                              ("evidence_continue.py", "evidence_revision.py", "sortnet.py", "runtime.py", "streaming.py")}}
    path = directory / "design.json"
    if path.exists():
        design = read_json(path)
        if design["spec_hash"] != digest(spec):
            raise ValueError("Follow-up changed; use a fresh directory")
    else:
        design = {"at": utc_now(), "before_api": True, "budget_before": None,
                  "original_budget_before": old_design["budget_before"], "spec_hash": digest(spec), "spec": spec}
        write_json(path, design)
    if not live:
        print(dumps({"mode": "offline", "max_new_calls": 3, "reuse": "cold, reflection and genuine Qwen native call"}))
        return
    if design["budget_before"] is None:
        if list((directory / "receipts").rglob("*.request.json")):
            raise ValueError("Missing pre-run budget")
        design["budget_before"] = usage()
        write_json(path, design)
    panel = old_design["spec"]["cases"]
    prompt = old_design["spec"]["cold"] + "\n" + old_design["spec"]["rules"] + "\nPrograms:\n" + dumps([c["public"] for c in panel])
    cold = {r["model"]: r for r in old["cold"]}
    provider = BudgetProvider(concurrency=2, per_model_concurrency=1)
    calls = base.Calls(directory, design, provider)

    def finish_model(model):
        history = [{"role": "user", "content": prompt}, *cold[model.id]["output"]]
        if model.id == qwen.id:
            requested = {"status": "completed", "call": call, "output": response.output,
                         "normalization": normalization, "reused_native_receipt": str(native_path.resolve())}
            history += [{"role": "user", "content": old_design["spec"]["audit_request"]}]
            tools = [base.TOOL]
        else:
            history += [{"role": "user", "content": REQUEST}]
            requested = calls.ask(model, "audit-retry", history, tool_call=True, tools=[TOOL])
            tools = [TOOL]
        write_json(directory / "native-calls" / f"{model.id}.json", requested)
        if requested["status"] != "completed":
            return {**requested, "model": model.id, "arm": "tool"}
        evidence = base.audit_panel(panel, cold[model.id]["scored"])
        write_json(directory / "executions" / f"{model.id}.json", evidence)
        history += requested["output"] + [{"type": "function_call_output",
            "call_id": requested["call"]["call_id"], "output": dumps(evidence)},
            {"role": "user", "content": old_design["spec"]["after_audit"]}]
        return {**base.scored(calls.ask(model, "after-audit", history, tools=tools), panel), "arm": "tool"}

    try:
        arms = [r for r in old["arms"] if r["arm"] == "reflection"]
        arms += base.parallel(base.MODELS_PINNED, finish_model, "Complete native evidence comparison")
        after, before = usage(), design["original_budget_before"]
        result = {"at": utc_now(), "decision": base.decide(old["cold"], arms, panel),
                  "cold": old["cold"], "arms": arms, "budget_after": after,
                  "new_phase_calls": after["calls"] - design["budget_before"]["calls"],
                  **{k: after[k] - before[k] for k in
                     ("calls", "reported_credits", "uncertain_reserve", "input_tokens", "output_tokens")}}
        write_json(directory / "result.json", result)
        print(dumps({k: result[k] for k in ("decision", "calls", "reported_credits", "uncertain_reserve")}), flush=True)
    finally:
        provider.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/evidence-revision-2"))
    parser.add_argument("--previous", type=Path, default=Path("runs/evidence-revision-1"))
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    run(args.output, args.previous, args.live)
