from dataclasses import replace

import pytest

from selftrap.maze import Maze, State, collision, discover, encoder, evaluate, held_out, plan, verify_certificate


def test_minimal_world_and_exact_missing_state():
    found = discover()
    maze, result = found["world"], found["evaluation"]
    assert len(maze.rooms) == 3
    assert len(maze.passages) == 2
    assert result["optimal_steps"] == 3
    assert result["reachable_states"] == 4
    assert result["plan"] is None
    witness = result["certificate"]
    assert witness["first"]["room"] == witness["second"]["room"]
    assert witness["first"]["has_key"] != witness["second"]["has_key"]
    assert witness["difference"]["kind"] == "legal_move_mismatch"
    assert verify_certificate(maze, "room", witness)
    repaired = evaluate(maze, "(room, has_key)")
    assert repaired["solved"] and repaired["optimal"]
    assert repaired["sufficient_on_reachable_states"]
    assert maze.replay(repaired["plan"]).room == maze.goal


def test_certificate_rejects_unreachable_witness_and_false_claim():
    found = discover()
    maze, witness = found["world"], found["evaluation"]["certificate"]
    wrong = {**witness, "path_to_second": []}
    assert not verify_certificate(maze, "room", wrong)
    wrong = {**witness, "difference": {**witness["difference"], "legal_from_first": True}}
    assert not verify_certificate(maze, "room", wrong)
    assert not verify_certificate(maze, "(room, has_key)", witness)


def test_key_is_persistent_door_is_undirected_and_goal_is_terminal():
    maze = Maze(("A", "B", "K", "G"), (("A", "B"), ("A", "K"), ("B", "G")),
                "A", "G", "K", ("A", "B"))
    assert "B" not in maze.moves(State("A", False))
    assert "A" not in maze.moves(State("B", False))
    assert maze.moves(State("A", True))["B"] == State("B", True)
    assert maze.moves(State("B", True))["A"] == State("A", True)
    assert maze.replay(["K", "A", "B", "A"]) == State("A", True)
    assert maze.moves(State("G", True)) == {}


def test_transfer_layouts_and_plain_control():
    for maze in held_out():
        full = evaluate(maze, "(room, has_key)")
        assert full["solvable"] and full["solved"] and full["optimal"]
        assert full["sufficient_on_reachable_states"]
    plain = held_out()[-1]
    assert evaluate(plain, "room")["sufficient_on_reachable_states"]
    assert evaluate(plain, "room")["solved"]
    for maze in held_out()[:2]:
        assert not evaluate(maze, "room")["solved"]
    # An alternative route can rescue one query despite an invalid abstraction.
    cycle = held_out()[2]
    assert evaluate(cycle, "room")["solved"]
    assert not evaluate(cycle, "room")["sufficient_on_reachable_states"]


@pytest.mark.parametrize("expression", ["room.upper()", "__import__('os')", "room.__class__", "room[0]"])
def test_model_expression_cannot_execute_arbitrary_code(expression):
    with pytest.raises(ValueError):
        encoder(expression)


def test_bad_world_is_rejected():
    maze = held_out()[0]
    with pytest.raises(ValueError):
        replace(maze, locked_passage=(maze.key_room, maze.goal))
