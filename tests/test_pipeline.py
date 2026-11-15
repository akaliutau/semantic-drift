from selftrap.analysis import analyse
from selftrap.experiment import build_jobs, init_run, solve
from selftrap.game import Level, Push
from selftrap.providers import ModelResponse
from selftrap.reporting import export_dataset, replay_text
from selftrap.storage import dumps, read_json, write_json
from selftrap.validation import ROOT, validate
from test_validation import make_proposal


def test_one_offline_episode_through_runner_analysis_and_export(level, tmp_path, monkeypatch):
    cfg = read_json(ROOT/"configs/pilot.json")
    cfg["slots_per_condition"] = 1
    cfg["validation"]["min_pushes"] = 2
    manifest = init_run(tmp_path/"run", cfg)
    directory = tmp_path/"run"
    first = manifest["slots"][0]
    proposal = make_proposal(level)
    for forecast, model in zip(proposal["forecasts"], cfg["models"]): forecast["solver_id"] = model["id"]
    checked = validate(proposal, [m["id"] for m in cfg["models"]], min_pushes=2)
    for slot in manifest["slots"]:
        result = {"valid": False, "status": "invalid"}
        if slot["id"] == first["id"]:
            result = {"valid": True, "status": "valid", "proposal": proposal,
                      "puzzle_id": checked["certificate"]["puzzle_id"], "validation": checked}
        write_json(directory/"slots"/slot["id"]/"result.json", result)
    class PlannerProvider:
        calls = 0
        def __init__(self, *a, **kw): pass
        def complete(self, model, messages, **kwargs):
            self.calls += 1
            text = dumps({"plan": proposal["solution"]})
            return ModelResponse(text, [], {"status": "completed", "model": model.id}, .001,
                                 {"input_tokens": 10, "output_tokens": 10}, .001, "credits")
        def close(self): pass
    monkeypatch.setattr("selftrap.experiment.RunProvider", PlannerProvider)
    solve(directory, limit=1)
    # A single fake episode is executed despite the larger job manifest.
    results = list((directory/"episodes").glob("*/result.json"))
    assert len(results) == 1 and read_json(results[0])["full_success"]
    report = analyse(directory, tmp_path/"summary", resamples=10, progress=False)
    assert sum(report["terminal_status_counts"].values()) == 1
    export_dataset(directory, tmp_path/"export")
    assert len((tmp_path/"export/dataset.jsonl").read_text().splitlines()) == 12
    assert "Complete legal solution: True" in replay_text(ROOT/"examples/tiny_puzzle.json")
