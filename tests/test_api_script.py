from scripts import test_api as smoke
from selftrap.providers import ModelResponse


def test_default_probe_matches_working_example_and_calls_once(monkeypatch, tmp_path):
    seen = []
    class FakeProvider:
        def complete(self, model, messages, **kwargs):
            seen.append((model, messages, kwargs))
            return ModelResponse("A unicorn dreamed of stars.", [],
                {"model": model.id, "status": "completed"}, .01,
                {"input_tokens": 15, "output_tokens": 8}, .0001, "credits")
        def close(self): pass
    monkeypatch.setattr(smoke, "create_provider", lambda *a: FakeProvider())
    monkeypatch.setattr("sys.argv", ["test_api.py", "--live", "--output", str(tmp_path)])
    smoke.main()
    assert len(seen) == 1
    model, prompt, kwargs = seen[0]
    assert model.id == "qwen3.5-397b"
    assert prompt == "Write a one-sentence bedtime story about a unicorn."
    assert kwargs["tools"] == []
    # A second invocation reuses the receipt, making zero additional calls.
    smoke.main()
    assert len(seen) == 1
