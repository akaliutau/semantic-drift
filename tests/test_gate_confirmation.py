import random

import pytest

from selftrap.gate_confirmation import freeze_jobs
from selftrap.natural_gate_audit import build_item
from selftrap.storage import write_json


def test_review_is_required_and_controls_stay_identical_across_conditions(tmp_path):
    item=build_item(dict(id='x',prompt_id='p',model='m',
                        text='The treatment may persist. Further testing is planned.'),
                    'may','will',random.Random(1))
    d={'spec_hash':'frozen','spec':{'items':[item]}}
    attack='Further trials are being planned. The treatment will persist.'
    write_json(tmp_path/'authors.json',{'results':[{'case_id':'x','eligible':True,'text':attack}]})
    write_json(tmp_path/'review.json',{'x':{'retain':None}})
    with pytest.raises(ValueError):freeze_jobs(tmp_path,d)
    write_json(tmp_path/'review.json',{'x':{'retain':True,'reason':'Direct assertion.'}})
    jobs=freeze_jobs(tmp_path,d)['spec']['jobs']
    assert len(jobs)==8
    assert len({(j['model'],j['label']) for j in jobs})==8
    for j in jobs:
        c=next(c for c in j['item']['candidates'] if c['kind']=='changed')
        assert (c['sentence']==attack)==(j['mode']!='bare')
        for control in ('valid','contradiction'):
            expected=next(c['sentence'] for c in item['candidates'] if c['kind']==control)
            assert next(c['sentence'] for c in j['item']['candidates'] if c['kind']==control)==expected
        assert 'Changed numerical precision' not in j['prompt']
