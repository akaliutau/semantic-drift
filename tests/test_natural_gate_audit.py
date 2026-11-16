import random

from selftrap.natural_gate_audit import build_item, prompt, word_diff


def test_edits_are_exact_and_diff_contains_no_semantic_label():
    row=dict(id='x',prompt_id='p',model='m',text='The effect may persist. Testing will continue.')
    item=build_item(row,'may','will',random.Random(1))
    c=next(x for x in item['candidates'] if x['kind']=='changed')
    assert c['sentence']=='The effect will persist. Testing will continue.'
    assert word_diff(row['text'],c['sentence'])==[{'source':'may','candidate':'will'}]
    assert item['reference']['span']==[11,14]
    assert 'AUTOMATIC WORD DIFFS' in prompt(item,'diff')
    assert 'Strengthens uncertainty' not in prompt(item,'diff')
