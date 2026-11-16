from functools import lru_cache

import pytest

from selftrap.planning_lab import Arena, World, make_world, records, verify


def test_ordered_dots_cannot_skip_numbers_revisit_or_finish_early():
    world = World("dots", 3, 3, points=(0, 2, 8))
    before_last = world.replay(["D", "D", "R"])
    assert "R" not in world.moves(before_last)  # Cannot visit 3 before 2.
    state = world.replay(["R", "R", "D", "D"])
    assert state[0] == 8 and not world.goal(state)
    assert world.moves(state) == {}
    with pytest.raises(ValueError):
        world.replay(["R", "L"])


def test_numberlink_can_leave_an_entire_disconnected_empty_region():
    world = World("numberlink", 4, 3, pairs=((4, 7), (8, 11)))
    state = world.replay(["R", "R", "R"])
    assert state[0] == 8 and state[2] == 1
    assert not any(state[1] & (1 << i) for i in range(4))
    assert world.goal(world.replay(["R", "R", "R"], state))
    assert Arena(world).completion(state) == ["R", "R", "R"]


def test_robots_slide_to_obstacles_and_helper_positions_matter():
    world = World("ricochet", 4, 4, robots=(0, 2, 15), target=5)
    assert world.moves(world.initial)["0:R"] == (1, 2, 15)
    assert world.moves(world.initial)["0:D"] == (12, 2, 15)
    blocked = World("ricochet", 4, 4, walls=((0, 4),), robots=(0, 2, 15), target=5)
    assert "0:D" not in blocked.moves(blocked.initial)
    arena = Arena(world)
    route = arena.completion(world.initial)
    assert route and world.goal(world.replay(route))


@pytest.mark.parametrize("game", ["dots", "ricochet", "numberlink"])
def test_exact_reverse_distances_match_independent_depth_limited_search(game):
    arena = Arena(make_world(game, 11, 3, 3))
    world = arena.world

    @lru_cache(None)
    def possible(state, left):
        if world.goal(state):
            return True
        return left > 0 and any(possible(child, left - 1) for child in world.moves(state).values())

    for state in list(arena.parent)[:25]:
        for budget in (0, 1, 2, 3):
            assert possible(state, budget) == (state in arena.distance and arena.distance[state] <= budget)
    probes = records(arena, 8)
    assert len({p["id"] for p in probes}) == len(probes)
    assert all(verify(p["certificate"]) for p in probes)


def test_world_validation_rejects_nonadjacent_walls_and_duplicate_endpoints():
    with pytest.raises(ValueError):
        World("ricochet", 4, 4, walls=((0, 15),), robots=(0, 1, 2), target=5)
    with pytest.raises(ValueError):
        World("numberlink", 3, 3, pairs=((0, 1), (1, 2)))
