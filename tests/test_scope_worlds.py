from selftrap.scope_worlds import candidates,oracle,panel,predicate,ORDERS


def test_composed_counterexamples_replay_and_positive_controls_are_equivalent():
    for c in panel(21,16):
        g=oracle(c);assert not g['equivalent'] and g['permitted_worlds']>=16
        b=[x['value'] for x in g['counterexample']['assignments']]
        a,z=ORDERS[c['mode']]
        assert predicate(c,b,a)!=predicate(c,b,z)
        assert all(b[int(i)]==v for i,v in c['facts'].items())
        assert oracle(c,True)['equivalent']
        assert len(candidates(c))==16
