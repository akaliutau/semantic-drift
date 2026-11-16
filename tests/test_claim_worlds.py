from copy import deepcopy

from selftrap.claim_worlds import canonical, make_panel, oracle, statements, variants, verify_witness


def test_every_negative_has_a_replayable_counterexample_and_valid_controls_are_equivalent():
    cases=make_panel(11,30)
    for c in cases:
        g=oracle(c)
        assert not g['equivalent'] and verify_witness(c,g['counterexample'])
        assert oracle(c,'valid')['equivalent']
        assert len(variants(c))==12 and statements(c,'valid')
        damaged=deepcopy(g['counterexample']);damaged['rewrite']=damaged['source']
        assert not verify_witness(c,damaged)


def test_discovery_and_confirmation_have_no_semantic_cluster_overlap():
    a=make_panel(11,15);b=make_panel(12,60,exclude=[c['semantic_id'] for c in a])
    assert {c['semantic_id'] for c in a}.isdisjoint(c['semantic_id'] for c in b)
    assert len({c['semantic_id'] for c in a+b})==75


def test_renaming_interchangeable_locks_does_not_create_a_new_sample():
    c={'family':'quantifier_order','n':2,'m':2,'k':0,'facts':{'0':True,'3':False}}
    renamed={**c,'facts':{'2':True,'1':False}}
    assert canonical(c)==canonical(renamed)
