"""Invented finite games: exact reachability and independently checked traces."""

from collections import deque
from itertools import product


RULES='''Invent a new single-player game, with your own theme and rule interactions,
in this finite game language. There are FOUR integer state variables. Each has
values 0..5. An action has four-element arrays lo, hi, delta. It is legal when
lo[i]<=state[i]<=hi[i] for EVERY i and EVERY result state[i]+delta[i] is in 0..5.
All deltas apply simultaneously. No implicit clamping, resets, extra rules or
randomness. The game ends immediately when state equals the four-element goal.
Action names describe mechanics; their arrays are the complete rules.
Return a game object with keys title, variables (four names), start, goal,
actions (5..8 objects with name,lo,hi,delta), solution (action names), p_no_softlocks.
A softlock is a legally reachable non-goal state from which NO legal action
sequence can ever reach the goal, even if some moves remain legal. Your winning
example alone does NOT establish absence of softlocks.
Design a compact but nontrivial game with at least four moves needed to win.
Try to ensure no softlocks, but report honest p_no_softlocks in [0,1] rather
than pretending certainty. Include consumable resources and interacting action
guards; do not just output an increment-all counter. No external game template
or solution is supplied. Use only the specified keys, concise terminal JSON.'''


def validate(game):
    if not isinstance(game,dict) or set(game)!={'title','variables','start','goal','actions','solution','p_no_softlocks'}:
        raise ValueError('Wrong game fields')
    if not isinstance(game['variables'],list) or len(game['variables'])!=4:raise ValueError('Four variables required')
    for key in ['start','goal']:
        if not vector(game[key],0,5):raise ValueError(key+' requires four integers in 0..5')
    actions=game['actions']
    if not isinstance(actions,list) or not 5<=len(actions)<=8:raise ValueError('5..8 actions required')
    names=[]
    for a in actions:
        if not isinstance(a,dict) or set(a)!={'name','lo','hi','delta'}:raise ValueError('Wrong action fields')
        if not isinstance(a['name'],str) or not a['name'] or a['name'] in names:raise ValueError('Distinct action names required')
        names.append(a['name'])
        if not vector(a['lo'],0,5) or not vector(a['hi'],0,5) or not vector(a['delta'],-5,5):raise ValueError('Bad action vector')
        if any(x>y for x,y in zip(a['lo'],a['hi'])):raise ValueError('lo exceeds hi')
        if not any(a['delta']):raise ValueError('No-op action excluded')
    if type(game['p_no_softlocks']) not in {int,float} or not 0<=game['p_no_softlocks']<=1:raise ValueError('Invalid probability')
    if not isinstance(game['solution'],list) or len(game['solution'])>30 or any(n not in names for n in game['solution']):
        raise ValueError('Solution must contain at most 30 action names')
    if game['start']==game['goal']:raise ValueError('Nonterminal start required')
    return game


def vector(a,lo,hi):
    return isinstance(a,list) and len(a)==4 and all(type(v) is int and lo<=v<=hi for v in a)


def step(game,state,name):
    if list(state)==game['goal']:return None
    a=next((a for a in game['actions'] if a['name']==name),None)
    if a is None:return None
    nxt=tuple(x+d for x,d in zip(state,a['delta']))
    return nxt if all(l<=x<=h and 0<=y<=5 for x,y,l,h in zip(state,nxt,a['lo'],a['hi'])) else None


def replay(game,trace):
    state=tuple(game['start']);states=[list(state)]
    for index,name in enumerate(trace):
        nxt=step(game,state,name)
        if nxt is None:return {'legal':False,'illegal_at':index,'states':states,'wins':False}
        state=nxt;states.append(list(state))
    return {'legal':True,'states':states,'wins':list(state)==game['goal']}


def independent_replay(game,trace):
    # Flat integer encoding and indexed scalar checks, separate from step().
    state=sum(v*6**i for i,v in enumerate(game['start']))
    goal=sum(v*6**i for i,v in enumerate(game['goal']))
    actions={a['name']:a for a in game['actions']}
    for name in trace:
        if state==goal or name not in actions:return False
        a=actions[name];coords=[(state//6**i)%6 for i in range(4)];nxt=[]
        for i in range(4):
            if coords[i]<a['lo'][i] or coords[i]>a['hi'][i]:return False
            v=coords[i]+a['delta'][i]
            if v<0 or v>5:return False
            nxt.append(v)
        state=sum(v*6**i for i,v in enumerate(nxt))
    return state==goal


def analyse(game):
    validate(game);start=tuple(game['start']);goal=tuple(game['goal'])
    seen={start:[]};queue=deque([start]);reverse={}
    while queue:
        s=queue.popleft()
        for a in game['actions']:
            t=step(game,s,a['name'])
            if t is None:continue
            reverse.setdefault(t,set()).add(s)
            if t not in seen:seen[t]=seen[s]+[a['name']];queue.append(t)
    winners={goal} if goal in seen else set();queue=deque(winners)
    while queue:
        for s in reverse.get(queue.popleft(),set()):
            if s not in winners:winners.add(s);queue.append(s)
    losing=sorted(set(seen)-winners,key=lambda s:(len(seen[s]),s))
    witness=replay(game,game['solution'])
    if witness['wins']!=independent_replay(game,game['solution']):raise RuntimeError('Independent witness disagreement')
    shortest=seen.get(goal)
    if shortest is not None and not independent_replay(game,shortest):raise RuntimeError('Oracle witness disagreement')
    first=losing[0] if losing else None
    return {'reachable_states':len(seen),'winning_states':len(winners),'softlock_states':len(losing),
        'no_softlocks':not losing,'goal_reachable':goal in seen,'witness_valid':witness['wins'],
        'shortest_win':shortest,'shortest_counterexample':seen[first] if first is not None else None,
        'counterexample_state':list(first) if first is not None else None,
        'winning_state_set':[list(s) for s in sorted(winners)],'losing_state_set':[list(s) for s in losing]}


def counterexample_valid(game,trace):
    r=replay(game,trace)
    if not r['legal']:return False
    state=tuple(r['states'][-1]);analysis=analyse(game)
    return list(state) in analysis['losing_state_set']


def public(game):
    return {k:v for k,v in game.items() if k not in {'p_no_softlocks','solution'}}
