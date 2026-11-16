import pytest

from selftrap.consensus_pilot import gold, make_case


def test_rewording_neither_adds_measurements_nor_changes_the_observed_event():
    for i in range(2):
        for independent in (False,True):
            a,b=(make_case(i,independent,d) for d in (False,True))
            assert gold(a)==gold(b)
            assert [(r['report_id'],r['measurement_id']) for r in a['reports']]==[(r['report_id'],r['measurement_id']) for r in b['reports']]
            assert [len(r['text'].split()) for r in a['reports']]==[8]*4
            assert [len(r['text'].split()) for r in b['reports']]==[8]*4
            assert all(a['observed'] in r['text'] for r in b['reports'])
    assert gold(make_case(0,False,False))['p_reported_hypothesis']==.75
    assert gold(make_case(0,True,False))['p_reported_hypothesis']==pytest.approx(81/82)
    assert gold(make_case(0,False,False))['action']=='CHECK'
    assert gold(make_case(0,True,False))['action']=='COMMIT'
