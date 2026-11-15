"""Model-invented finite resource games with compiler-certified goals."""
from collections import deque

SEMANTICS='''Three resources are integers in 0..3. An action has cost[3] and gain[3].
It is legal if each current resource >= its cost, and each result
current-cost+gain is in 0..3. All changes are simultaneous. Names are descriptive
only; arrays are the complete rules. Goal equality immediately ends the game.
No extra rules, clamping, history, or randomness. A softlock is a reachable
non-goal state from which NO legal sequence can reach the goal. Having legal
actions does not rule out a softlock.'''

def transition(game,s,index):
    if list(s)==game.get('goal'):return None
    a=game['actions'][index]
    if any(v<c for v,c in zip(s,a['cost'])):return None
    nxt=tuple(v-c+g for v,c,g in zip(s,a['cost'],a['gain']))
    return nxt if all(0<=v<=3 for v in nxt) else None

def graph(game):
    start=tuple(game['start']);paths={start:[]};edges={};queue=deque([start])
    while queue:
        s=queue.popleft();edges[s]=[]
        for i in range(len(game['actions'])):
            t=transition(game,s,i)
            if t is None:continue
            edges[s].append((i,t))
            if t not in paths:paths[t]=paths[s]+[i];queue.append(t)
    return paths,edges

def vector(a):return isinstance(a,list) and len(a)==3 and all(type(v) is int and 0<=v<=3 for v in a)

def compile_mechanics(obj):
    if not isinstance(obj,dict) or set(obj)!={'title','resources','start','actions'}:raise ValueError('Wrong fields')
    if not isinstance(obj['resources'],list) or len(obj['resources'])!=3 or not vector(obj['start']):raise ValueError('Three resources required')
    if not isinstance(obj['actions'],list) or not 4<=len(obj['actions'])<=6:raise ValueError('4..6 actions required')
    for a in obj['actions']:
        if not isinstance(a,dict) or set(a)!={'name','cost','gain'} or not vector(a['cost']) or not vector(a['gain']):raise ValueError('Wrong action fields')
        if a['cost']==a['gain']:raise ValueError('No-op excluded')
    paths,_=graph(obj)
    if len(paths)<5:raise ValueError('Fewer than five reachable states')
    goal=max(paths,key=lambda s:(len(paths[s]),s));game={**obj,'goal':list(goal)};oracle=analyse(game)
    if not oracle['goal_reachable'] or not independent_win(game,oracle['winning_witness']):raise RuntimeError('Invalid compiled goal')
    return game,oracle

def analyse(game):
    paths,edges=graph(game);goal=tuple(game['goal']);winning={goal} if goal in paths else set()
    while True:
        added={s for s,es in edges.items() if s not in winning and any(t in winning for _,t in es)}
        if not added:break
        winning|=added
    losing=set(paths)-winning;traps=[]
    for s in sorted(winning):
        safe=[i for i,t in edges[s] if t in winning];bad=[i for i,t in edges[s] if t in losing]
        if safe and bad:traps.append({'state':list(s),'safe':safe,'bad':bad,'path':paths[s]})
    return {'reachable_states':len(paths),'goal_reachable':goal in paths,'winning_witness':paths.get(goal),
        'no_softlocks':not losing,'softlock_states':[list(s) for s in sorted(losing)],
        'looping_softlocks':[list(s) for s in sorted(losing) if edges[s]],'trap_states':traps,
        'shortest_softlock':min((paths[s] for s in losing),key=lambda p:(len(p),p),default=None)}

def replay(game,actions,start=None):
    s=tuple(game['start'] if start is None else start);states=[list(s)]
    if not isinstance(actions,list) or len(actions)>30 or any(type(i) is not int or not 0<=i<len(game['actions']) for i in actions):raise ValueError('Action indices required')
    for i in actions:
        t=transition(game,s,i)
        if t is None:return {'legal':False,'states':states,'wins':False}
        s=t;states.append(list(s))
    return {'legal':True,'states':states,'wins':list(s)==game['goal']}

def independent_win(game,trace):
    s=game['start'][:]
    for i in trace:
        if s==game['goal'] or not 0<=i<len(game['actions']):return False
        a=game['actions'][i];nxt=[]
        for j in range(3):
            if s[j]<a['cost'][j]:return False
            v=s[j]-a['cost'][j]+a['gain'][j]
            if not 0<=v<=3:return False
            nxt.append(v)
        s=nxt
    return s==game['goal']

def valid_softlock(game,trace):
    r=replay(game,trace)
    return r['legal'] and r['states'][-1] in analyse(game)['softlock_states']
