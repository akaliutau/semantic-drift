import pytest

from pipeline.artifacts import strict_json
from pipeline.protocol import candidates,construction_check,load_cases,protected_sentence,score


def test_frozen_topics_controls_and_construction_guard():
    cases=load_cases('data/cases.jsonl',40,2026111605)
    assert len({c['cluster'] for c in cases})==40
    for case in cases:
        p=protected_sentence(case)
        assert case['protected']==p
        assert construction_check(case,{'text':p})[0]
        assert not construction_check(case,{'text':'“'+p+'”'})[0]
        bare=candidates(case,p,'bare',1);masked=candidates(case,p,'rewritten',1)
        assert [(c['id'],c['kind']) for c in bare]==[(c['id'],c['kind']) for c in masked]
        assert [c for c in bare if c['kind']!='changed']==[c for c in masked if c['kind']!='changed']
    with pytest.raises(ValueError):load_cases('data/cases.jsonl',41,1)


def test_ambiguous_or_nonfinite_submissions_are_rejected():
    for value in ['{"A":true,"A":false}','1e999','NaN']:
        with pytest.raises(ValueError):strict_json(value)
    rows=candidates(load_cases('data/cases.jsonl',1,1)[0],'rewrite','bare',1)
    with pytest.raises(ValueError):score(rows,{'A':True})
