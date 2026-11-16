from selftrap.concept_audit import audit_world
from selftrap.planning_lab import World, verify


def test_offline_search_carries_real_short_completions_for_unsafe_pruning():
    world = World("numberlink", 3, 3, pairs=((0, 1), (6, 7)))
    result = audit_world(world, 'def generate(batch):\n return [False for obs in batch]', budgets=(1, 2, 3))
    assert result["unsafe"] == result["winning"] > 0
    assert result["examples"] and all(verify(e["certificate"]) for e in result["examples"])
    assert all(len(e["certificate"]["completion"]) <= e["certificate"]["observation"]["moves_left"] for e in result["examples"])


def test_no_rejections_are_not_counted_as_a_new_method():
    world = World("dots", 3, 3, points=(0, 2, 6))
    result = audit_world(world, 'def generate(batch):\n return [True for obs in batch]', budgets=(1, 3))
    assert result["unsafe"] == result["dead_pruned"] == 0
    assert result["kept"] == result["checked"]
