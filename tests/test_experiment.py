from pathlib import Path

import pytest

from selftrap.experiment import build_jobs, init_run, load_run, review_warning, status
from selftrap.storage import read_json, write_json
from selftrap.validation import ROOT


def test_manifest_freezes_config_and_builds_balanced_jobs(tmp_path):
    cfg = read_json(ROOT/"configs/pilot.json")
    cfg["slots_per_condition"] = 1
    cfg["repeats"] = 2
    manifest = init_run(tmp_path, cfg)
    assert len(manifest["slots"]) == 12
    assert load_run(tmp_path)["config_hash"] == manifest["config_hash"]
    for slot in manifest["slots"]:
        write_json(tmp_path/"slots"/slot["id"]/"result.json", {"valid": True, "puzzle_id": slot["id"]})
    jobs = build_jobs(tmp_path, manifest, "baseline")
    assert len(jobs) == 72
    assert len({j["id"] for j in jobs}) == 72
    matching = [j for j in jobs if j["slot"]["id"] == jobs[0]["slot"]["id"] and j["repeat"] == 0]
    assert len({j["seed"] for j in matching}) == 1
    assert status(tmp_path)["valid_puzzles"] == 12


def test_warning_review_must_precede_outcomes(tmp_path):
    cfg = read_json(ROOT/"configs/pilot.json")
    init_run(tmp_path, cfg)
    review = tmp_path/"warnings/slot/review.json"
    write_json(review, {"author": {"text": "Preserve access.", "decision": "pending"}})
    review_warning(tmp_path, "slot", "author", "approve", "tester")
    assert read_json(review)["author"]["decision"] == "approve"
    write_json(tmp_path/"episodes/one/result.json", {})
    with pytest.raises(ValueError, match="before any solver"):
        review_warning(tmp_path, "slot", "author", "reject", "tester")


def test_assignment_tampering_is_detected(tmp_path):
    manifest = init_run(tmp_path, read_json(ROOT/"configs/pilot.json"))
    manifest["slots"][0]["target"] = "some-other-model"
    write_json(tmp_path/"manifest.json", manifest)
    with pytest.raises(ValueError, match="assignments or provenance changed"):
        load_run(tmp_path)


def test_credit_guard_does_not_send_an_over_budget_request(tmp_path, monkeypatch):
    from selftrap.experiment import RunProvider
    from selftrap.providers import ModelSpec, ProviderError
    class NoCalls:
        def complete(self, *a, **kw): raise AssertionError("Should not send this request")
        def close(self): pass
    monkeypatch.setattr("selftrap.experiment.create_provider", lambda *a: NoCalls())
    provider = RunProvider(tmp_path, cap=.000001)
    with pytest.raises(ProviderError, match="no API request sent") as caught:
        provider.complete(ModelSpec("test", input_per_million=1, output_per_million=2),
                          [], tools=[], max_output_tokens=100, timeout=10)
    assert caught.value.billing == "not_sent"
