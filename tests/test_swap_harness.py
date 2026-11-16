import pytest

from selftrap.swap_harness import lattice, render, validate_policy


def test_edit_bank_can_change_style_without_changing_protected_words_or_numbers():
    source='“It may not hold,” she said. Among the neighbors, only 3.14 was recorded.'
    problem=lattice(source)
    assert render(problem,[0]*len(problem['sites']))==source
    altered=render(problem,[len(s['options'])-1 for s in problem['sites']])
    assert 'may not hold' in altered and 'only 3.14' in altered
    assert 'Amongst' in altered and 'neighbours' in altered
    assert altered!=source
    for plan in ([99]*len(problem['sites']), [True]*len(problem['sites']), []):
        with pytest.raises(ValueError):render(problem,plan)


def test_policy_cannot_import_os_or_access_classifier_internals():
    for code in ('import os\ndef generate(x): return []',
                 'def generate(x): return score.__globals__',
                 'def generate(x): return x.__class__'):
        with pytest.raises(ValueError):validate_policy(code)
