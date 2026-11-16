from selftrap.claim_worlds import make_panel, oracle, verify_witness
from selftrap.gate_audit import build_cases, evaluate, prompt


def test_new_gold_controls_and_all_prompts_preserve_the_same_target():
    previous = make_panel(2026111601, 20)
    items = build_cases(42, 10, [c['semantic_id'] for c in previous])
    assert {x['case']['semantic_id'] for x in items}.isdisjoint(c['semantic_id'] for c in previous)
    for item in items:
        case = item['case']
        assert verify_witness(case, oracle(case)['counterexample'])
        assert item['valid_oracle']['equivalent']
        for mode in ('document', 'localized', 'checklist'):
            p = prompt(item, mode)
            assert item['source_sentence'] in p
            assert all(c['sentence'] in p for c in item['candidates'])


def test_scoring_keeps_all_three_controls_separate():
    item = build_cases(31, 1)[0]
    # Build the actual shuffled candidate list before any live calls.
    item['candidates'] = [
        {'id':'K','kind':'changed','gold':False},
        {'id':'M','kind':'valid','gold':True},
        {'id':'L','kind':'contradiction','gold':False}]
    obj = {'judgments': [{'id':k, 'admissible':v, 'reason':'x'}
                         for k,v in [('L',False),('K',True),('M',True)]]}
    rows = evaluate(item,obj)
    assert [r['correct'] for r in rows] == [True,False,True]
