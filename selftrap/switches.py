"""Invented switch mechanics; exact horizon-limited solvability by construction."""
from collections import deque

BITS=10
MASK=(1<<BITS)-1
RULES='''A board has 10 lamps, represented by a 10-bit binary string. The goal is
all zero lamps. Each move first rotates the whole string LEFT by rotate places,
then XORs with its mask (1 flips a lamp, 0 preserves it). Operations are ordered:
rotate FIRST, XOR SECOND. Indices are zero-based. Every move is legal and can be
repeated. You have a strict remaining move budget. Finishing with few lamps on
is not winning; the board must be ALL ZERO within that budget. Immediate lamp
count is an accurate measurement but may be a misleading planning heuristic.'''

def turn(state,move):
    r=move['rotate']
    return (((state<<r)|(state>>(BITS-r)))&MASK)^move['mask']

def inverse(state,move):
    x=state^move['mask'];r=move['rotate']
    return ((x>>r)|(x<<(BITS-r)))&MASK

def distances(moves):
    found={0:0};queue=deque([0])
    while queue:
        t=queue.popleft()
        for move in moves:
            s=inverse(t,move)
            if s not in found:found[s]=found[t]+1;queue.append(s)
    return found

def validate(obj):
    if not isinstance(obj,dict) or set(obj)!={'title','moves'}:raise ValueError('title and moves required')
    if not isinstance(obj['moves'],list) or len(obj['moves'])!=5:raise ValueError('Exactly 5 moves required')
    seen=set()
    for m in obj['moves']:
        if not isinstance(m,dict) or set(m)!={'name','rotate','mask'}:raise ValueError('Wrong move fields')
        if type(m['rotate']) is not int or not 0<=m['rotate']<BITS or type(m['mask']) is not int or not 0<m['mask']<=MASK:raise ValueError('Move outside bounds')
        key=(m['rotate'],m['mask'])
        if key in seen:raise ValueError('Duplicate move')
        seen.add(key)
    return obj

def worlds(obj,kind='lure',limit=1):
    validate(obj);moves=obj['moves'];dist=distances(moves);choices=[]
    for s,d in sorted(dist.items()):
        if d!=4:continue
        nexts=[turn(s,m) for m in moves];safe=[i for i,t in enumerate(nexts) if dist.get(t,999)<=d-1]
        counts=[t.bit_count() for t in nexts];greedy=[i for i,n in enumerate(counts) if n==min(counts)]
        if len(safe)!=1 or len(greedy)!=1:continue
        if (safe[0]!=greedy[0])!=(kind=='lure'):continue
        plan=[];t=s
        for left in range(d,0,-1):
            i=next(i for i,m in enumerate(moves) if dist.get(turn(t,m),999)==left-1)
            plan.append(i);t=turn(t,moves[i])
        assert t==0
        # Independent string-based forward replay.
        text=f'{s:010b}'
        for i in plan:
            m=moves[i];r=m['rotate'];text=text[r:]+text[:r]
            text=''.join(str(int(a)^int(b)) for a,b in zip(text,f"{m['mask']:010b}"))
        if text!='0'*BITS:raise RuntimeError('Independent witness disagreement')
        choices.append({'game':obj,'start':s,'budget':d,'safe_action':safe[0],
            'greedy_action':greedy[0],'winning_plan':plan,'kind':kind,
            'successors':[{'action':i,'name':m['name'],'state':f'{t:010b}','lamps_on':t.bit_count()} for i,(m,t) in enumerate(zip(moves,nexts))]})
        if len(choices)>=limit:return choices
    return choices

def public(world):
    return {'title':world['game']['title'],'moves':[dict(m,mask=f"{m['mask']:010b}") for m in world['game']['moves']],
        'start':f"{world['start']:010b}",'remaining_moves':world['budget'],'goal':'0'*BITS,
        'next_states_already_computed':world['successors']}
