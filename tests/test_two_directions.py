import json

import pytest

from selftrap import two_directions as study
from selftrap.providers import ModelResponse
from selftrap.runtime import RESPONSE_IDS
from selftrap.sortnet import certify, discovery_panel, pool
from selftrap.storage import read_json


EXACT_CLONE = '''def generate(panel):
    result = {}
    for case in panel:
        valid = True
        for mask in range(16):
            values = [(mask // d) % 2 for d in (8, 4, 2, 1)]
            for i, j in case["steps"]:
                if values[i] > values[j]:
                    values[i], values[j] = values[j], values[i]
            if values != sorted(values):
                valid = False
                break
        result[case["id"]] = "VALID" if valid else "INVALID"
    return result
'''


def test_pure_program_surrogate_can_execute_and_inspector_is_forbidden():
    cases = discovery_panel()["cases"]
    result = study.clone_predictions(EXACT_CLONE, [r["public"] for r in cases])
    assert all(result[r["public"]["id"]] == ("VALID" if r["certificate"]["valid"] else "INVALID") for r in cases)
    with pytest.raises(ValueError):
        study.clone_predictions("def generate(panel):\n return inspect(panel)", [])


def test_false_rejection_is_separate_from_unknown_and_bad_counterexample():
    cases = discovery_panel()["cases"]
    obj = {"rows": [{"id": r["public"]["id"], "verdict": "INVALID", "counterexample": [0, 0, 0, 0]} for r in cases]}
    result = study.score_actor(obj, cases)
    assert sum(r["false_rejection"] for r in result) == 2
    assert not any(r["counterexample_valid"] for r in result)
    for row in obj["rows"]:
        row["verdict"] = "UNKNOWN"
    assert not any(r["false_rejection"] for r in study.score_actor(obj, cases))


def test_surrogate_search_does_not_call_a_model_and_runtime_errors_are_reported():
    cases = pool(79, 2, 2)["cases"]
    assert study.candidate({"eligible": True, "code": EXACT_CLONE}, cases) is None
    constant = 'def generate(panel):\n return {p["id"]: "INVALID" for p in panel}'
    chosen = study.candidate({"eligible": True, "code": constant}, cases)
    assert chosen["certificate"]["valid"]
    broken = {"eligible": True, "code": 'def generate(panel):\n return 1 // 0'}
    assert study.candidate(broken, cases) is None
    assert not broken["eligible"] and "search_error" in broken


def test_no_traps_stop_at_four_requests_and_resume_never_replays(tmp_path, monkeypatch):
    calls = []

    class FakeProvider:
        def __init__(self, **kwargs):
            pass

        def complete(self, model, messages, **kwargs):
            prompt = messages[0]["content"]
            calls.append((model.id, prompt))
            if prompt.startswith(study.ACTOR):
                panel = json.loads(prompt.split("Programs:\n", 1)[1])
                obj = {"rows": []}
                for case in panel:
                    certificate = certify(case["steps"])
                    obj["rows"].append({"id": case["id"], "verdict": "VALID" if certificate["valid"] else "INVALID",
                                        "counterexample": None if certificate["valid"] else certificate["counterexample"]["input"]})
            else:
                assert "observed_programs" in prompt and '"steps"' in prompt
                obj = {"code": EXACT_CLONE, "hypothesis": "Synthetic fixture, never research evidence."}
            return ModelResponse(json.dumps(obj), [], {"id": str(len(calls)), "status": "completed", "model": RESPONSE_IDS[model.id]},
                                 0.01, {"input_tokens": 10, "output_tokens": 10, "cost": .0001}, .0001, "credits")

        def close(self):
            pass

    monkeypatch.setattr(study, "BudgetProvider", FakeProvider)
    monkeypatch.setattr(study, "render", lambda *args: None)
    monkeypatch.setattr(study, "usage", lambda: {"calls": len(calls), "reported_credits": len(calls) * .0001,
        "uncertain_reserve": 0, "input_tokens": len(calls) * 10, "output_tokens": len(calls) * 10,
        "in_flight": 0, "hard_ceiling": 190})
    study.run(tmp_path, live=True)
    result = read_json(tmp_path / "result.json")
    assert len(calls) == result["new_requests"] == 4
    assert result["stop"] == "no_false_rejections_or_self_predicted_traps"
    assert len(result["self_clones"]) == 2
    study.run(tmp_path, live=True)
    assert len(calls) == 4
    followup = tmp_path / "followup"
    study.run(followup, live=True, tokens=24000, seconds=300, reuse_discovery=tmp_path)
    assert len(calls) == 6  # Two new authors, no repeated actors.
    result = read_json(followup / "result.json")
    assert result["new_requests"] == 2
    assert all(r.get("reused_from") for r in result["discovery"])


def test_incomplete_target_is_unscored_and_completed_target_continues(tmp_path, monkeypatch):
    calls = []

    class PartialProvider:
        def __init__(self, **kwargs):
            pass

        def complete(self, model, messages, **kwargs):
            prompt = messages[0]["content"]
            calls.append((model.id, prompt))
            status = "completed"
            if model.id == "qwen3.5-397b":
                status, obj = "incomplete", None
            elif prompt.startswith(study.ACTOR):
                panel = json.loads(prompt.split("Programs:\n", 1)[1])
                obj = {"rows": [{"id": p["id"], "verdict": "VALID" if certify(p["steps"])["valid"] else "INVALID",
                                 "counterexample": None} for p in panel]}
            else:
                obj = {"code": EXACT_CLONE, "hypothesis": "Synthetic exact-checker fixture."}
            return ModelResponse(json.dumps(obj), [], {"status": status, "model": RESPONSE_IDS[model.id]},
                                 .01, {"cost": .0001}, .0001, "credits")

        def close(self):
            pass

    monkeypatch.setattr(study, "BudgetProvider", PartialProvider)
    monkeypatch.setattr(study, "render", lambda *args: None)
    monkeypatch.setattr(study, "usage", lambda: {"calls": len(calls), "reported_credits": len(calls) * .0001,
        "uncertain_reserve": 0, "input_tokens": 0, "output_tokens": 0, "in_flight": 0, "hard_ceiling": 190})
    study.run(tmp_path, live=True)
    result = read_json(tmp_path / "result.json")
    assert len(calls) == 3
    assert result["unscored_targets"] == ["qwen3.5-397b"]
    assert [r["model"] for r in result["self_clones"]] == ["gemma-4-31b-it"]


def test_dotted_model_names_keep_construction_repair_receipts_distinct(tmp_path, monkeypatch):
    calls = []

    class Provider:
        def complete(self, model, messages, **kwargs):
            calls.append(messages)
            return ModelResponse('{"ok":true}', [], {"status": "completed", "model": RESPONSE_IDS[model.id]},
                                 .01, {"cost": .0001}, .0001, "credits")

    snapshot = {"reported_credits": 0, "uncertain_reserve": 0}
    monkeypatch.setattr(study, "usage", lambda: snapshot)
    requests = study.Requests(tmp_path, {"budget_before": snapshot,
        "spec": {"tokens": 12000, "seconds": 180}}, Provider())
    model = study.MODELS_PINNED[0]
    label = "self-clone-for-" + model.id
    first = requests.ask(model, label, "Original author prompt")
    repair = requests.ask(model, label + "-construction-repair", "Different repair prompt")
    assert first["status"] == repair["status"] == "completed"
    assert len(list((tmp_path / "receipts").rglob("*.request.json"))) == 2
    requests.ask(model, label + "-construction-repair", "Different repair prompt")
    assert len(calls) == 2
