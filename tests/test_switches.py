from selftrap.switches import turn,inverse,worlds,distances

def test_inverse_and_horizon_certificates():
    moves=[{'name':str(i),'rotate':r,'mask':m} for i,(r,m) in enumerate([(1,53),(3,417),(5,783),(2,207),(7,969)])]
    for s in [0,13,1023]:
        for move in moves:assert inverse(turn(s,move),move)==s
    w=worlds({'title':'Fixture','moves':moves})[0]
    dist=distances(moves)
    assert dist[turn(w['start'],moves[w['safe_action']])]<=w['budget']-1
    assert dist.get(turn(w['start'],moves[w['greedy_action']]),999)>w['budget']-1
