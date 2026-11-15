"""Exercise real SDK serialization against an in-memory HTTP transport."""

import json

from adaption import Adaption
import httpx

from selftrap.game import Push
from selftrap.providers import AdaptionProvider, ModelSpec
from selftrap.solver import run_episode


def test_native_function_roundtrip_without_network(level, tmp_path):
    bodies = []
    final = '{"plan":[{"box":[2,3],"direction":"left"},{"box":[4,3],"direction":"right"}]}'
    def handle(request):
        body = json.loads(request.content)
        bodies.append(body)
        if len(bodies) == 1:
            output = [{"type": "function_call", "call_id": "call-one", "name": "simulate_plan",
                       "arguments": '{"plan":[]}'}]
            text = ""
        else:
            assert body["input"][-1]["type"] == "function_call_output"
            assert body["input"][-1]["call_id"] == "call-one"
            output = [{"type": "message", "role": "assistant",
                       "content": [{"type": "output_text", "text": final}]}]
            text = final
        return httpx.Response(200, json={"id": f"resp-{len(bodies)}", "object": "response",
            "status": "completed", "model": "test", "output": output, "output_text": text,
            "usage": {"input_tokens": 100, "output_tokens": 30, "total_tokens": 130}})
    client = Adaption(api_key="local-test-key", max_retries=0,
                     http_client=httpx.Client(transport=httpx.MockTransport(handle)))
    provider = AdaptionProvider(client=client)
    try:
        result = run_episode(provider, ModelSpec("test"), level, tmp_path,
                             bad_push=Push((2,3),"up"), force_first_simulation=True)
    finally: provider.close()
    assert result["full_success"] and result["tool_calls"] == 1
    assert len(bodies) == 2 and "agentic_memory" not in bodies[0] and "store" not in bodies[0]
    assert "max_tool_calls" not in bodies[0]
    assert bodies[0]["tool_choice"] == {"type": "function", "name": "simulate_plan"}
    assert all("local-test-key" not in p.read_text() for p in tmp_path.rglob("*.json"))
