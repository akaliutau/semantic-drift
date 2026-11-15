import pytest

from selftrap.analysis import analyse
from selftrap.experiment import init_run
from selftrap.storage import read_json, write_json
from selftrap.validation import ROOT


def fixture_run(tmp_path):
    cfg = read_json(ROOT/"configs/pilot.json")
    cfg["slots_per_condition"] = 1
    return init_run(tmp_path, cfg)


def test_unrun_experiment_reports_missing_not_zero(tmp_path):
    fixture_run(tmp_path)
    report = analyse(tmp_path, tmp_path/"analysis", resamples=20, progress=False)
    assert report["primary"]["D_complete_panels"] is None
    assert report["primary"]["D_missing_data_bounds"] == [-2, 2]
    assert not report["primary"]["complete"]


def test_known_self_specific_signal_and_invalid_denominator(tmp_path):
    manifest = fixture_run(tmp_path)
    roster = [m["id"] for m in manifest["config"]["models"]]
    for slot in manifest["slots"]:
        write_json(tmp_path/"slots"/slot["id"]/"result.json", {"valid": True, "status": "valid",
                   "proposal": {"forecasts": []}, "puzzle_id": slot["id"]})
        for solver in roster:
            error = slot["target"] == slot["author"] == solver
            path = tmp_path/"episodes"/f"{slot['id']}-{roster.index(solver)}"
            write_json(path/"job.json", {"slot": slot, "solver": solver, "arm": "baseline", "repeat": 0})
            write_json(path/"result.json", {"model": solver, "status": "unsolved", "predicted_error": error, "full_success": False})
    report = analyse(tmp_path, tmp_path/"analysis", resamples=30, progress=False)
    assert report["primary"]["D_complete_panels"] == 1
    assert report["primary"]["D_missing_data_bounds"] == [1, 1]
    assert report["primary"]["complete"]
    self_slot = next(s for s in manifest["slots"] if s["author"] == s["target"])
    write_json(tmp_path/"slots"/self_slot["id"]/"result.json", {"valid": False, "status": "invalid"})
    report = analyse(tmp_path, tmp_path/"analysis", resamples=20, progress=False)
    assert report["primary"]["D_complete_panels"] == pytest.approx(2/3)
