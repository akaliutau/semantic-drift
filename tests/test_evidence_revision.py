import copy
import json

from selftrap import evidence_revision as study
from selftrap.providers import ModelResponse
from selftrap.runtime import RESPONSE_IDS
from selftrap.storage import dumps, read_json


def fixtures(panel, *, reject_valid=False, acknowledge=False):
    rows = []
    for c in panel:
        valid = c["certificate"]["valid"]
        verdict = "INVALID" if reject_valid or not valid else "VALID"
        witness = ([3, 2, 1, 0] if valid else c["certificate"]["counterexample"]["input"]) if verdict == "INVALID" else None
        row = {"id": c["public"]["id"], "verdict": verdict, "counterexample": witness}
        if acknowledge:
            row["tool_verdict_read"] = "VALID" if valid else "INVALID"
        rows.append(row)
    return {"rows": rows}


def fake_provider(monkeypatch, *, reject_baseline=True, persist=True):
    calls = []

    class Provider:
        def __init__(self, **kwargs):
            pass

        def complete(self, model, messages, **kwargs):
            calls.append((model.id, messages, kwargs))
            tool_call = isinstance(kwargs["tool_choice"], dict)
            if tool_call:
                assert messages[-1]["content"] == study.AUDIT_REQUEST
                output = [{"type": "function_call", "call_id": "real-fixture-call-" + model.id,
                           "name": "audit_panel", "arguments": "{}"}]
                text = ""
            else:
                evidence = [m for m in messages if m.get("type") == "function_call_output"]
                if evidence:
                    native = [m for m in messages if m.get("type") == "function_call"]
                    assert len(native) == len(evidence) == 1
                    assert native[0]["call_id"] == evidence[0]["call_id"]
                    assert len(json.loads(evidence[0]["output"])["programs"]) == 3
                cold = len(messages) == 1
                reject = model.id == "gemma-4-31b-it" and (reject_baseline if cold else persist)
                obj = fixtures(study.cases(), reject_valid=reject, acknowledge=bool(evidence))
                text = dumps(obj)
                output = [{"type": "message", "role": "assistant",
                           "content": [{"type": "output_text", "text": text}]}]
            return ModelResponse(text, output, {"status": "completed", "model": RESPONSE_IDS[model.id]},
                                 .01, {"input_tokens": 10, "output_tokens": 10, "cost": .0001}, .0001, "credits")

        def close(self):
            pass

    monkeypatch.setattr(study, "BudgetProvider", Provider)
    monkeypatch.setattr(study, "usage", lambda: {"calls": len(calls), "reported_credits": len(calls) * .0001,
        "uncertain_reserve": 0, "input_tokens": len(calls) * 10, "output_tokens": len(calls) * 10,
        "in_flight": 0, "hard_ceiling": 190})
    return calls


def test_failed_witness_does_not_prove_broken_program_valid():
    panel = study.cases()
    observed = fixtures(panel, reject_valid=True)["rows"]
    observed[0]["counterexample"] = [3, 2, 1, 0]
    report = study.audit_panel(panel, observed)["programs"]
    assert report[0]["witness_execution"]["was_counterexample"] is False
    assert report[0]["exhaustive_verdict"] == "INVALID"
    assert report[0]["certificate"]["counterexample"] is not None
    assert report[1]["exhaustive_verdict"] == "VALID"
    assert report[1]["witness_execution"]["final"] == [0, 1, 2, 3]


def test_full_native_round_trip_is_bounded_and_resume_reuses_results(tmp_path, monkeypatch):
    calls = fake_provider(monkeypatch)
    study.run(tmp_path, live=True)
    result = read_json(tmp_path / "result.json")
    assert result["decision"] == "ACCEPT_PILOT_PROMISE"
    assert result["calls"] == len(calls) == 8
    assert len(list((tmp_path / "executions").glob("*.json"))) == 2
    study.run(tmp_path, live=True)
    assert len(calls) == 8


def test_acceptance_requires_acknowledgment_and_sound_negative_control(tmp_path, monkeypatch):
    fake_provider(monkeypatch)
    study.run(tmp_path, live=True)
    result = read_json(tmp_path / "result.json")
    panel = study.cases()
    arms = copy.deepcopy(result["arms"])
    gemma = next(r for r in arms if r["model"] == "gemma-4-31b-it" and r["arm"] == "tool")
    gemma["scored"][1]["tool_verdict_read"] = "INVALID"
    assert study.decide(result["cold"], arms, panel) == "REJECT_THIS_VERSION"
    arms = copy.deepcopy(result["arms"])
    gemma = next(r for r in arms if r["model"] == "gemma-4-31b-it" and r["arm"] == "tool")
    gemma["scored"][0]["counterexample_valid"] = False
    assert study.decide(result["cold"], arms, panel) == "REJECT_THIS_VERSION"


def test_correcting_after_tool_evidence_rejects_hypothesis(tmp_path, monkeypatch):
    fake_provider(monkeypatch, persist=False)
    study.run(tmp_path, live=True)
    assert read_json(tmp_path / "result.json")["decision"] == "REJECT_THIS_VERSION"


def test_no_cold_rejections_stops_after_two_calls(tmp_path, monkeypatch):
    calls = fake_provider(monkeypatch, reject_baseline=False)
    study.run(tmp_path, live=True)
    assert read_json(tmp_path / "result.json")["decision"] == "REJECT_NO_STABLE_BASELINE"
    assert len(calls) == 2
