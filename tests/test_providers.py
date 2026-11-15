import json
from types import SimpleNamespace

import pytest

from selftrap.providers import AdaptionProvider, ModelSpec, estimated_cost


def test_wrapper_maps_native_tools_without_persistent_memory():
    captured = {}
    raw = {"id": "test", "model": "test-model", "status": "completed",
           "output": [{"type": "function_call", "call_id": "1", "name": "simulate_plan",
                       "arguments": '{"plan":[]}'}],
           "usage": {"input_tokens": 100, "output_tokens": 20}}
    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(output_text="", model_dump=lambda **_: raw)
    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    model = ModelSpec("test-model", input_per_million=1, output_per_million=2)
    response = AdaptionProvider(client=client).complete(
        model, [{"role": "user", "content": "test"}], tools=[{"type": "function", "name": "simulate_plan"}],
        max_output_tokens=100, timeout=5)
    assert "store" not in captured and "agentic_memory" not in captured
    assert "max_tool_calls" not in captured
    assert captured["tools"][0]["name"] == "simulate_plan"
    assert response.calls[0]["name"] == "simulate_plan"
    assert json.loads(response.calls[0]["arguments"]) == {"plan": []}
    assert response.estimated_cost == pytest.approx(0.00014)
    assert estimated_cost(model, None) is None


def test_no_implicit_provider_or_stack_fallback():
    provider = AdaptionProvider(client=object())
    with pytest.raises(ValueError):
        provider.complete(ModelSpec("stack:my-memory"), [], tools=[], max_output_tokens=1, timeout=1)


def test_text_requests_omit_tool_call_controls():
    captured = {}
    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(output_text="A unicorn slept.",
            model_dump=lambda **_: {"model": "qwen3.5-397b", "status": "completed", "output": []})
    provider = AdaptionProvider(client=SimpleNamespace(responses=SimpleNamespace(create=create)))
    provider.complete(ModelSpec("qwen3.5-397b"), [{"role": "user", "content": "story"}],
                      tools=[], max_output_tokens=2000, timeout=60)
    assert "tool_choice" not in captured and "max_tool_calls" not in captured
    assert "tools" not in captured
    assert set(captured) == {"model", "input", "max_output_tokens", "timeout"}


def test_standard_sdk_key_has_precedence(monkeypatch, tmp_path):
    monkeypatch.setenv("ADAPTION_API_KEY", "canonical-test-key")
    monkeypatch.setenv("ADAPTION_API_TOKEN", "legacy-test-token")
    provider = AdaptionProvider(env_file=tmp_path/"absent.env")
    try: assert provider.client.api_key == "canonical-test-key"
    finally: provider.close()


def test_missing_canonical_key_fails_before_network(monkeypatch, tmp_path):
    monkeypatch.delenv("ADAPTION_API_KEY", raising=False)
    monkeypatch.setenv("ADAPTION_API_TOKEN", "legacy-test-token")
    with pytest.raises(ValueError, match="ADAPTION_API_KEY"):
        AdaptionProvider(env_file=tmp_path/"absent.env")
