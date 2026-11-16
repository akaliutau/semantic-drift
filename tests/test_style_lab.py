import pytest

pytest.importorskip("numpy")
from selftrap.style_lab import normalize_typography, split_groups, surface_features


def test_all_responses_to_a_prompt_stay_in_one_split():
    rows=[{'id':f'{i}-{m}','prompt_id':str(i),'text':f'topic {i}','model':m}
          for i in range(25) for m in ('A','B')]
    split=split_groups(rows)
    groups={name:{r['prompt_id'] for r in part} for name,part in split.items()}
    assert groups['train'].isdisjoint(groups['test'])
    assert groups['train'].isdisjoint(groups['calibration'])
    assert groups['test'].isdisjoint(groups['calibration'])
    assert sum(map(len,split.values()))==len(rows)


def test_surface_perturbation_preserves_probability_and_negation_tokens():
    text='“It may not hold”—probability 0.6; range 0.4–0.8.'
    normalized=normalize_typography(text)
    assert 'may not hold' in normalized and '0.6' in normalized and '0.4-0.8' in normalized
    assert surface_features('') is not None
