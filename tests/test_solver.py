from copy import deepcopy
from types import SimpleNamespace

import pytest

from selftrap.game import Push
from selftrap.providers import ModelResponse, ModelSpec
from selftrap.solver import grade, run_episode
from selftrap.storage import Journal, dumps, read_json, strict_json, usage_summary, write_json


class FakeProvider:
    def __init__(self, responses):
        self.responses, self.seen = list(responses), []

    def complete(self, model, messages, **kwargs):
        self.seen.append(deepcopy({"messages": messages, **kwargs}))
        value = self.responses.pop(0)
        if isinstance(value, Exception): raise value
        return value


def response(text="", output=None):
    output = output or []
    raw = {"id": "fake", "model": "test", "status": "completed", "output": output}
    return ModelResponse(text, output, raw, 0.01, {"input_tokens": 100, "output_tokens": 20}, 0.001, "credits")


def test_exploration_is_not_commitment_and_resumes(level, tmp_path):
    bad = Push((2, 3), "up")
    call = {"type": "function_call", "call_id": "one", "name": "simulate_plan",
            "arguments": dumps({"plan": [bad.to_dict()]})}
    final = '{"plan":[{"box":[2,3],"direction":"left"},{"box":[4,3],"direction":"right"}]}'
    provider = FakeProvider([response(output=[call]), response(final)])
    result = run_episode(provider, ModelSpec("test"), level, tmp_path, bad_push=bad)
    assert result["full_success"] and not result["predicted_error"]
    assert result["tool_calls"] == 1
    assert run_episode(provider, ModelSpec("test"), level, tmp_path, bad_push=bad) == result
    assert len(provider.seen) == 2
    assert usage_summary(tmp_path)["requests_with_response"] == 2
    public = provider.seen[0]["messages"][0]["content"]
    assert "forecasts" not in public and "predicted_first_push" not in public


def test_exact_event_survives_later_illegal_action(level):
    bad = Push((2, 3), "up")
    plan = [bad.to_dict(), Push((0, 0), "down").to_dict()]
    result = grade(level, dumps({"plan": plan}), bad)
    assert result["predicted_error"] and not result["full_success"]
    assert result["status"] == "illegal"
    assert grade(level, '```json\n{}\n```', bad)["status"] == "malformed"


def test_uncertain_calls_are_not_silently_repeated(tmp_path):
    from selftrap.providers import ProviderError
    journal = Journal(tmp_path)
    provider = FakeProvider([RuntimeError("process died")])
    with pytest.raises(RuntimeError):
        journal.complete("x", provider, ModelSpec("test"), [], tools=[], max_output_tokens=100, timeout=10)
    with pytest.raises(ProviderError, match="unknown outcome"):
        journal.complete("x", FakeProvider([]), ModelSpec("test"), [], tools=[], max_output_tokens=100, timeout=10)


def test_strict_json_and_provider_failure_are_not_wrong_answers(level, tmp_path):
    from selftrap.providers import ProviderError
    with pytest.raises(ValueError): strict_json('{"p":NaN}')
    with pytest.raises(ValueError): strict_json('{"p":1e999}')
    with pytest.raises(ValueError): strict_json('{"p":1,"p":2}')
    result = run_episode(FakeProvider([ProviderError("denied")]), ModelSpec("test"), level, tmp_path)
    assert result["status"] == "service_failure" and result["predicted_error"] is None


def test_resume_after_mid_episode_crash_reuses_completed_call(level, tmp_path):
    call = {"type": "function_call", "call_id": "one", "name": "simulate_plan", "arguments": '{"plan":[]}'}
    first = FakeProvider([response(output=[call]), RuntimeError("process interrupted")])
    with pytest.raises(RuntimeError): run_episode(first, ModelSpec("test"), level, tmp_path)
    final = '{"plan":[{"box":[2,3],"direction":"left"},{"box":[4,3],"direction":"right"}]}'
    second = FakeProvider([response(final)])
    result = run_episode(second, ModelSpec("test"), level, tmp_path, retry_uncertain=True)
    assert result["full_success"] and len(second.seen) == 1
    assert usage_summary(tmp_path)["request_attempts"] == 3
    assert usage_summary(tmp_path)["unknown_billing_requests"] == 1
