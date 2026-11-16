"""Provider interface and the official Adaption SDK adapter.

Implement Provider.complete to add another service. The experiment consumes
ordinary JSON messages and normalized responses, never SDK model instances.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
from time import perf_counter
from typing import Protocol

from dotenv import load_dotenv


@dataclass(frozen=True)
class ModelSpec:
    id: str
    provider: str = "adaption"
    input_per_million: float = 0.0
    output_per_million: float = 0.0
    unit: str = "credits"
    settings: dict = field(default_factory=dict)


@dataclass
class ModelResponse:
    text: str
    output: list[dict]
    raw: dict
    seconds: float
    usage: dict | None
    estimated_cost: float | None
    unit: str

    @property
    def calls(self) -> list[dict]:
        return [item for item in self.output if item.get("type") == "function_call"]


class ProviderError(RuntimeError):
    def __init__(self, message: str, retryable: bool = False, status_code: int | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code


class Provider(Protocol):
    def complete(self, model: ModelSpec, messages: list[dict] | str, *, tools: list[dict],
                 max_output_tokens: int, timeout: float,
                 tool_choice: str | dict = "auto") -> ModelResponse: ...


def estimated_cost(model: ModelSpec, usage: dict | None) -> float | None:
    """Catalog estimate at uncached rates, not an invoice; missing usage is unknown."""
    if not usage or any(usage.get(k) is None for k in ("input_tokens", "output_tokens")):
        return None
    return (usage["input_tokens"]*model.input_per_million
            + usage["output_tokens"]*model.output_per_million)/1_000_000


class AdaptionProvider:
    def __init__(self, env_file: str | Path = ".env", client=None):
        if client is not None:
            self.client = client
            return
        from adaption import Adaption
        load_dotenv(env_file, override=False)
        if not os.environ.get("ADAPTION_API_KEY"):
            raise ValueError("Set ADAPTION_API_KEY in the environment/.env")
        # Let the official SDK read its own standard environment configuration.
        # Retries belong to our journal, so every attempt has a receipt.
        self.client = Adaption(max_retries=0)

    def complete(self, model: ModelSpec, messages: list[dict] | str, *, tools: list[dict],
                 max_output_tokens: int, timeout: float,
                 tool_choice: str | dict = "auto") -> ModelResponse:
        from adaption import APIConnectionError, APIStatusError
        if model.provider != "adaption" or model.id.startswith("stack:"):
            raise ValueError("Use a direct Adaption model ID, not a persistent inference stack")
        if set(model.settings)-{"temperature"}:
            raise ValueError("Adaption settings currently support only temperature; no silent overrides")
        started = perf_counter()
        options = {"model": model.id, "input": messages,
                   "max_output_tokens": max_output_tokens, **model.settings}
        if tools:
            options["tools"] = tools
        if tool_choice != "auto":
            options["tool_choice"] = tool_choice
        # Direct model calls have no stack memory and store defaults to false.
        # max_tool_calls is for Adaption's hosted/MCP loop; our local solver
        # enforces its own function-call limit instead.
        try:
            response = self.client.responses.create(**options, timeout=timeout)
        except APIStatusError as exc:
            # Do not include request headers, credentials or an untrusted body in errors.
            raise ProviderError(f"Adaption HTTP {exc.status_code}",
                                exc.status_code in {408, 409, 429} or exc.status_code >= 500,
                                exc.status_code) from None
        except APIConnectionError as exc:
            raise ProviderError(f"Adaption connection error: {type(exc).__name__}", True) from None
        raw = response.model_dump(mode="json", exclude_none=True)
        usage = raw.get("usage")
        return ModelResponse(response.output_text or "", raw.get("output", []), raw,
                             perf_counter()-started, usage, estimated_cost(model, usage), model.unit)

    def close(self):
        self.client.close()


def create_provider(name: str, env_file: str | Path = ".env") -> Provider:
    if name == "adaption": return AdaptionProvider(env_file)
    raise ValueError(f"Provider {name!r} is not implemented")
