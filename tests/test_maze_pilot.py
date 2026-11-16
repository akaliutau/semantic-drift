"""Offline guards: easy tasks must stop early and resume must not spend again."""
import json

import pytest

from selftrap import maze_pilot
from selftrap.providers import ModelResponse
from selftrap.runtime import RESPONSE_IDS
from selftrap.storage import read_json


@pytest.mark.parametrize("cold_correct, expected_calls, expected_stop", [
    (True, 2, "cold_ceiling"), (False, 6, "repair_panel_complete")])
def test_prospective_stop_and_resume_without_provider_replay(tmp_path, monkeypatch, cold_correct, expected_calls, expected_stop):
    calls = []

    class FakeProvider:
        def __init__(self, **kwargs):
            pass

        def complete(self, model, messages, **kwargs):
            prompt = messages[0]["content"]
            certificate = "Mechanically checked counterexample certificate" in prompt
            cold = "Current representation: room" not in prompt
            expression = "(room, has_key)" if certificate or cold and cold_correct else "room"
            calls.append((model.id, expression))
            text = json.dumps({"node_expression": expression, "diagnosis": "Synthetic test fixture, not a model result."})
            return ModelResponse(text, [], {"status": "completed", "model": RESPONSE_IDS[model.id], "id": str(len(calls))},
                                 0.01, {"input_tokens": 100, "output_tokens": 30, "cost": 0.0001}, 0.0001, "credits")

        def close(self):
            pass

    def usage():
        return {"calls": len(calls), "reported_credits": len(calls) * 0.0001, "uncertain_reserve": 0,
                "input_tokens": len(calls) * 100, "output_tokens": len(calls) * 30,
                "in_flight": 0, "hard_ceiling": 190}

    monkeypatch.setattr(maze_pilot, "BudgetProvider", FakeProvider)
    monkeypatch.setattr(maze_pilot, "usage", usage)
    monkeypatch.setattr(maze_pilot, "render", lambda *args: None)
    maze_pilot.run(tmp_path, live=True)
    result = read_json(tmp_path / "result.json")
    assert result["stop"] == expected_stop
    assert result["new_requests"] == expected_calls == len(calls)
    if not cold_correct:
        assert all(t["passed"] == (t["arm"] == "certificate") for t in result["trials"])
    maze_pilot.run(tmp_path, live=True)
    assert len(calls) == expected_calls
