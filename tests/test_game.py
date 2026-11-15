import pytest
from selftrap.game import Push, State, canonical_id, parse_plan, transform_level, transform_push
from selftrap.oracle import Oracle


def test_witness_and_independent_oracle(level):
    result = Oracle(level).search()
    assert result.status == "SOLVED" and len(result.plan) == 2
    assert Oracle(level).replay(result.plan)
    assert level.replay(result.plan)["completed"]
    assert level.board.render(level.state)[2] == "#_.$__#"


def test_legal_deadlock_and_caps(level):
    bad = Push((2, 3), "up")
    after = level.board.push(level.state, bad)
    trapped = type(level)(level.board, after)
    assert Oracle(trapped).search().status == "UNSOLVABLE"
    assert Oracle(level).search(max_states=1).status == "UNKNOWN"


def test_illegal_after_solved_is_not_success(level):
    solution = Oracle(level).search().plan
    replay = level.replay(solution+[Push((0, 0), "up")])
    assert not replay["completed"]
    assert replay["illegal_action"]["index"] == 2
    assert not Oracle(level).replay(solution+[Push((0, 0), "up")])


def test_all_symmetries_and_reachable_player_equivalence(level):
    witness = Oracle(level).search().plan
    key = canonical_id(level)
    for symmetry in range(8):
        other = transform_level(level, symmetry)
        plan = [transform_push(p, 7, symmetry) for p in witness]
        assert canonical_id(other) == key
        assert other.replay(plan)["completed"] and Oracle(other).replay(plan)
        assert len(Oracle(other).search().plan) == len(witness)
    pos = max(level.board.reachable(level.state))
    other = type(level)(level.board, State(pos, level.state.boxes))
    assert canonical_id(other) == key


def test_push_validation_and_no_mutation(level):
    before = level.state
    with pytest.raises(ValueError): level.board.push(before, Push((0, 0), "up"))
    with pytest.raises(ValueError): parse_plan([{"box": [True, 2], "direction": "up"}])
    assert level.state == before
    assert len(level.board.push(before, Push((4, 3), "right")).boxes) == len(before.boxes)
