"""Stage 3: atomic JSON artifacts and per-request receipts, no database required.

Completed requests are reused on resume. An interrupted in-flight request has
unknown billing and is never silently replayed. There are no SDK-level retries.
"""

from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import time

from .providers import ModelResponse, ModelSpec, Provider, ProviderError


def dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def digest(value: object) -> str:
    return sha256(dumps(value).encode()).hexdigest()


def strict_json(text: str):
    def unique(items):
        result = {}
        for key, value in items:
            if key in result: raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    def invalid(value): raise ValueError(f"Nonfinite JSON number: {value}")
    def finite(value):
        number = float(value)
        if not math.isfinite(number): return invalid(value)
        return number
    return json.loads(text, object_pairs_hook=unique, parse_constant=invalid, parse_float=finite)


def read_json(path: str | Path):
    return strict_json(Path(path).read_text())


def write_json(path: str | Path, value: object):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    with temporary.open("w") as stream:
        stream.write(dumps(value)+"\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Journal:
    def __init__(self, directory: str | Path, retry_uncertain: bool = False):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.retry_uncertain = retry_uncertain

    def complete(self, label: str, provider: Provider, model: ModelSpec,
                 messages: list[dict] | str, *, tools: list[dict], max_output_tokens: int,
                 timeout: float, tool_choice: str | dict = "auto", retries: int = 2) -> ModelResponse:
        request = {"model": asdict(model), "messages": messages, "tools": tools,
                   "max_output_tokens": max_output_tokens, "tool_choice": tool_choice}
        request_hash = digest(request)
        started = time.perf_counter()
        restored_seconds = 0.0
        prior_attempt_seconds = 0.0
        for attempt in range(retries+1):
            stem = self.directory/f"{label}-{attempt}"
            request_path = stem.with_suffix(".request.json")
            response_path = stem.with_suffix(".response.json")
            error_path = stem.with_suffix(".error.json")
            if request_path.exists():
                prior = read_json(request_path)
                if prior["request_hash"] != request_hash:
                    raise ValueError("Resume request differs; use a fresh run directory")
                if response_path.exists():
                    response = ModelResponse(**read_json(response_path))
                    response.seconds += prior_attempt_seconds
                    return response
                if error_path.exists():
                    failure = read_json(error_path)
                    restored_seconds += failure["seconds"]
                    prior_attempt_seconds += failure["seconds"]
                    if not failure["retryable"] or attempt == retries:
                        raise ProviderError(failure["message"], failure["retryable"], failure["status_code"])
                    continue
                if not self.retry_uncertain:
                    raise ProviderError("Interrupted request has unknown outcome; inspect receipt and use --retry-uncertain explicitly")
                # Preserve the uncertain request; a new attempt is a separate receipt.
                if attempt == retries: raise ProviderError("No remaining attempts for uncertain request")
                continue
            remaining = timeout-(time.perf_counter()-started)-restored_seconds
            if remaining <= 0: raise ProviderError("Request/retry deadline exhausted", True)
            write_json(request_path, {"created_at": utc_now(), "request_hash": request_hash,
                                      "request": request})
            attempt_started = time.perf_counter()
            try:
                response = provider.complete(model, messages, tools=tools,
                    max_output_tokens=max_output_tokens, timeout=remaining, tool_choice=tool_choice)
            except ProviderError as exc:
                write_json(error_path, {"at": utc_now(), "message": str(exc),
                           "retryable": exc.retryable, "status_code": exc.status_code,
                           "seconds": time.perf_counter()-attempt_started,
                           "billing": getattr(exc, "billing", "unknown")})
                if not exc.retryable or attempt == retries: raise
                prior_attempt_seconds += time.perf_counter()-attempt_started
                time.sleep(min(2**attempt, max(0, timeout-(time.perf_counter()-started))))
                continue
            write_json(response_path, asdict(response))
            response.seconds += prior_attempt_seconds
            return response
        raise ProviderError("Retry budget exhausted")


def usage_summary(directory: str | Path) -> dict:
    receipts = [read_json(p) for p in Path(directory).rglob("*.response.json")]
    costs: dict[str, float] = {}
    missing_usage = 0
    input_tokens = output_tokens = 0
    for item in receipts:
        usage = item["usage"] or {}
        input_tokens += usage.get("input_tokens", 0) or 0
        output_tokens += usage.get("output_tokens", 0) or 0
        if item["estimated_cost"] is None: missing_usage += 1
        else: costs[item["unit"]] = costs.get(item["unit"], 0)+item["estimated_cost"]
    failed_seconds = 0.0
    unknown_billing = 0
    request_paths = list(Path(directory).rglob("*.request.json"))
    for path in request_paths:
        if path.with_name(path.name.replace(".request.json", ".response.json")).exists(): continue
        error_path = path.with_name(path.name.replace(".request.json", ".error.json"))
        if error_path.exists():
            error = read_json(error_path)
            if error.get("billing") != "not_sent":
                request_at = datetime.fromisoformat(read_json(path)["created_at"])
                failed_seconds += max(0, (datetime.fromisoformat(error["at"])-request_at).total_seconds())
            if error.get("billing") != "not_sent": unknown_billing += 1
        else: unknown_billing += 1
    return {"request_attempts": len(request_paths), "requests_with_response": len(receipts), "input_tokens": input_tokens,
            "output_tokens": output_tokens, "estimated_cost": costs,
            "missing_usage_receipts": missing_usage,
            "api_seconds": sum(r["seconds"] for r in receipts)+failed_seconds,
            "failed_or_uncertain_requests": len(request_paths)-len(receipts),
            "unknown_billing_requests": unknown_billing,
            "cost_note": "Catalog estimate at uncached rates; failed/uncertain billing is unknown"}
