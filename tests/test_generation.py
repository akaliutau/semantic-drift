from selftrap.generation import ValidationConfig, author_slot, procedural_puzzle
from selftrap.providers import ModelSpec
from selftrap.storage import dumps, read_json
from test_solver import FakeProvider, response
from test_validation import make_proposal


def test_author_acceptance_and_validity_only_repair(level, tmp_path):
    roster = [ModelSpec(m) for m in ["test", "B", "C"]]
    p = make_proposal(level)
    p["forecasts"][0]["solver_id"] = "test"
    provider = FakeProvider([response('{"invalid":true}'), response(dumps(p))])
    result = author_slot(provider, roster[0], roster, "test", tmp_path,
                         validation=ValidationConfig(min_pushes=2))
    assert result["valid"] and result["accepted_attempt"] == 1
    assert len(provider.seen) == 2
    assert "Mechanical validation failed" in provider.seen[1]["messages"][-1]["content"]
    assert author_slot(provider, roster[0], roster, "test", tmp_path,
                       validation=ValidationConfig(min_pushes=2)) == result


def test_procedural_generation_is_seeded_and_checked():
    config = ValidationConfig(min_pushes=2, max_pushes=30)
    a = procedural_puzzle(17, validation=config, max_attempts=30, progress=False)
    b = procedural_puzzle(17, validation=config, max_attempts=30, progress=False)
    assert a["valid"] and b["valid"]
    assert a["puzzle_id"] == b["puzzle_id"] and a["proposal"] == b["proposal"]
    assert a["validation"]["certificate"]["trap_search"]["status"] == "UNSOLVABLE"
