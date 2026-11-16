from itertools import permutations, product

import pytest

from selftrap.sortnet import CANONICAL, certify, discovery_panel, execute, pool


def test_known_sort_and_a_concrete_failure():
    assert certify(CANONICAL)["valid"]
    bad = certify(CANONICAL[:-1])
    assert not bad["valid"]
    witness = bad["counterexample"]
    assert execute(CANONICAL[:-1], witness["input"]) == witness["output"]
    assert witness["output"] != sorted(witness["input"])


def test_backward_comparison_is_not_automatically_wrong():
    steps = ((3, 0),) + CANONICAL
    assert certify(steps)["valid"]
    assert all(execute(steps, values) == sorted(values) for values in permutations(range(4)))
    assert execute(((3, 0),), [0, 1, 2, 3]) == [3, 1, 2, 0]


def test_binary_oracle_agrees_with_independent_permutation_and_tie_check():
    for row in pool(19, 3, 3)["cases"]:
        steps = row["public"]["steps"]
        distinct = all(execute(steps, values) == sorted(values) for values in permutations(range(4)))
        including_ties = all(execute(steps, values) == sorted(values) for values in product(range(4), repeat=4))
        assert row["certificate"]["valid"] == distinct == including_ties


def test_discovery_and_held_out_programs_are_distinct_and_labels_hidden():
    discovery = discovery_panel()["cases"]
    assert len(discovery) == 4
    assert sum(r["certificate"]["valid"] for r in discovery) == 2
    assert all(set(r["public"]) == {"id", "steps"} for r in discovery)
    hidden = pool(59, 3, 3, exclude=[r["public"]["id"] for r in discovery])["cases"]
    assert not {r["public"]["id"] for r in discovery} & {r["public"]["id"] for r in hidden}


def test_malformed_comparison_is_rejected():
    with pytest.raises(ValueError):
        execute(((0, 0),), [0, 1, 0, 1])
