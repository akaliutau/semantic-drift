from selftrap import worlds


def game():
    def a(name,lo,hi,delta):return dict(name=name,lo=lo,hi=hi,delta=delta)
    return {'title':'Keys','variables':['coin','key','gate','dummy'],'start':[2,0,0,0],
        'goal':[0,1,1,0],'actions':[
            a('buy',[1,0,0,0],[5,0,0,0],[-1,1,0,0]),
            a('burn',[1,0,0,0],[5,5,5,5],[-1,0,0,0]),
            a('open',[1,1,0,0],[5,5,0,0],[-1,0,1,0]),
            a('dummy_up',[0,0,0,0],[5,5,5,4],[0,0,0,1]),
            a('dummy_down',[0,0,0,1],[5,5,5,5],[0,0,0,-1])],
        'solution':['buy','open'],'p_no_softlocks':.9}


def test_softlock_can_have_legal_moves_and_is_not_a_failed_strategy():
    g=game();r=worlds.analyse(g)
    assert r['goal_reachable'] and r['witness_valid']
    assert not r['no_softlocks']
    assert worlds.counterexample_valid(g,['burn','burn'])
    assert worlds.step(g,[0,0,0,0],'dummy_up') is not None
    assert not worlds.counterexample_valid(g,['buy'])
    assert not worlds.counterexample_valid(g,['open'])
    assert worlds.independent_replay(g,['buy','open'])
