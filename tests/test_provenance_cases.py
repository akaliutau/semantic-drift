from copy import deepcopy

import pytest

from selftrap.provenance_cases import ARMS, cases, oracle, score, visible
from selftrap.storage import dumps


def test_provenance_treatments_leave_claims_origins_and_observations_unchanged():
    for case in cases():
        originals = visible(case, 'plain')
        for arm in ARMS:
            shown = visible(case, arm)
            for label, row in shown.items():
                assert dumps({k:v for k,v in row.items() if k != 'provenance'}) == dumps(originals[label])
                for step in row['pipeline']:
                    if step['operation'] == 'copy_exact':
                        assert step['input_sha256'] == step['output_sha256']
                        assert step['new_test_access'] is False


def test_editing_does_not_add_truth_evidence_but_an_actual_check_does():
    for case in cases():
        gold = oracle(case)
        expected = ('B' if case['weak_origin_label']=='A' else 'A') if case['split']=='positive_control' else case['weak_origin_label']
        assert gold['audit'] == expected
        answer = {**gold, 'why':'oracle'}
        answer.pop('residual_expected_errors')
        assert score(case, answer)['expected_error_regret'] == pytest.approx(0)
        wrong = deepcopy(answer)
        wrong['audit'] = 'B' if answer['audit']=='A' else 'A'
        assert score(case, wrong)['expected_error_regret'] > .15


def test_no_scalar_confidence_or_fake_numeric_response_is_silently_repaired():
    case = cases()[0]
    for value in (True, float('nan'), '0.2', 1.1):
        with pytest.raises(ValueError):
            score(case, {'p_correct':{'A':value,'B':.8},'audit':'A','why':'bad'})
