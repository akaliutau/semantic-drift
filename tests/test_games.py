from selftrap import games

def test_compiled_goal_and_softlock_are_independently_verified():
    obj={'title':'Resources','resources':['fuel','key','position'],'start':[2,0,0],
         'actions':[{'name':'trade','cost':[1,0,0],'gain':[0,1,0]},
                    {'name':'advance','cost':[0,1,0],'gain':[0,0,1]},
                    {'name':'lose fuel','cost':[1,0,0],'gain':[0,0,0]},
                    {'name':'add position','cost':[0,0,0],'gain':[0,0,1]}]}
    game,oracle=games.compile_mechanics(obj)
    assert oracle['goal_reachable'] and games.independent_win(game,oracle['winning_witness'])
    assert oracle['softlock_states']
    assert games.valid_softlock(game,oracle['shortest_softlock'])
