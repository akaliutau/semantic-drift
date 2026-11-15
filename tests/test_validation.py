from itertools import combinations

from selftrap.game import Board, Level, Push, State
from selftrap.oracle import Oracle
from selftrap.validation import validate


def make_proposal(level):
    return {"schema_version": "0.1", "board": level.board.render(level.state),
            "solution": [p.to_dict() for p in Oracle(level).search().plan],
            "predicted_first_push": Push((2, 3), "up").to_dict(),
            "forecasts": [{"solver_id": m, "p_predicted_first_push": 0.4,
                           "p_full_solution": 0.5} for m in ["A", "B", "C"]],
            "mechanism": "Local goal completion can block later work.",
            "diagnostic_warning": "Preserve access needed by the remaining boxes."}


def test_accept_reject_and_unknown(level):
    p = make_proposal(level)
    good = validate(p, ["A", "B", "C"], min_pushes=2)
    assert good["valid"]
    assert good["certificate"]["trap_search"]["status"] == "UNSOLVABLE"
    assert not validate(p, ["A", "B", "C"], min_pushes=2,
                        seen={good["certificate"]["puzzle_id"]})["valid"]
    p["solution"] = []
    assert not validate(p, ["A", "B", "C"], min_pushes=2)["valid"]
    p = make_proposal(level)
    assert not validate(p, ["A", "B", "C"], min_pushes=2, max_states=1)["valid"]
    p["forecasts"][0]["p_full_solution"] = 0.9
    assert not validate(p, ["A", "B", "C"], min_pushes=2)["valid"]


def test_independent_transitions_on_tiny_state_space():
    # 3x3 floor, two boxes: 252 states with nontrivial legal pushes.
    cells = {(r,c) for r in range(1,4) for c in range(1,4)}
    walls = frozenset((r, c) for r in range(5) for c in range(5) if (r, c) not in cells)
    board = Board(5, 5, walls, frozenset({(1,1),(3,3)}))
    transitions = 0
    for boxes in combinations(sorted(cells), 2):
        for player in cells-set(boxes):
            state = State(player, frozenset(boxes))
            for action in board.legal_pushes(state):
                after = board.push(state, action)
                check = Level(Board(5, 5, walls, after.boxes), state)
                assert Oracle(check).replay([action])
                transitions += 1
    assert transitions > 100
