"""Finite Black Box inference with head-on priority and turn/recheck physics.

Rule convention checked against James Harvey's fire_laser_internal in
Simon Tatham's Puzzles (Debian source sgt-puzzles/6879-1/blackbox.c).
"""
from functools import lru_cache
from itertools import combinations

RULES='''Black Box. Hidden atoms occupy the stated number of distinct cells of
an n*n grid. Row/column coordinates start at zero. Ports Ti/Bi enter column i
from top/bottom; Li/Ri enter row i from left/right.
A ray starts just outside its entry port. An atom directly ahead absorbs it
(HIT), taking priority over deflection. At entry, an atom diagonally ahead on
either side causes REFLECT. Otherwise enter the first cell. At each interior
cell: an atom directly ahead gives HIT; else a diagonally-ahead-left atom turns
the ray right IN PLACE and it rechecks; else a diagonally-ahead-right atom turns
it left IN PLACE and it rechecks; otherwise move forward. Atoms outside the
grid do not exist. An exit at the entry port is REFLECT; other exits use their
port label. All observed entry/outcome pairs are truthful.
Predict the queried new ray's outcome if ALL atom layouts consistent with
the observations agree; otherwise answer UNKNOWN. Do not assume the first
plausible layout is the true one. Submit through submit_answer(answer=...).
No external execution tools.'''

AUTHOR='''generate(seed) builds puzzle={"size":4,"atoms":[[1,1],[2,2]],
"shots":["T0","L1",...],"query":"B2"}. Size 3 or 4; exactly 2 or 3 distinct
atoms; 1..10 distinct initial shots; query is a valid port not already shot.
inspect(puzzle) computes observations from the exact simulator and enumerates
all consistent layouts. It returns public, truth (HIT/REFLECT/a port/UNKNOWN),
possible_count, query_outcomes, layouts and allowed_answers. Distinct layouts
may be observationally equivalent. You choose the layout, shots, query and
exact trap_answer. Target a particular inferential/physics mistake, not size.'''

def ports(n):return [side+str(i) for side in 'TBRL' for i in range(n)]

def ray(n,atoms,entry):
    atoms=set(atoms);side=entry[0];i=int(entry[1:])
    r,c,dr,dc={'T':(-1,i,1,0),'B':(n,i,-1,0),'L':(i,-1,0,1),'R':(i,n,0,-1)}[side]
    def look(r,c,dr,dc):
        return ((r+dr,c+dc) in atoms,(r+dr-dc,c+dc+dr) in atoms,(r+dr+dc,c+dc-dr) in atoms)
    ahead,left,right=look(r,c,dr,dc)
    if ahead:return 'HIT'
    if left or right:return 'REFLECT'
    r+=dr;c+=dc;seen=set()
    while 0<=r<n and 0<=c<n:
        state=(r,c,dr,dc)
        if state in seen:raise ValueError('Unexpected reachable ray cycle')
        seen.add(state);ahead,left,right=look(r,c,dr,dc)
        if ahead:return 'HIT'
        if left:dr,dc=dc,-dr
        elif right:dr,dc=-dc,dr
        else:r+=dr;c+=dc
    exit_port=('T'+str(c) if r<0 else 'B'+str(c) if r>=n else 'L'+str(r) if c<0 else 'R'+str(r))
    return 'REFLECT' if exit_port==entry else exit_port

@lru_cache(maxsize=8)
def world_table(n,count):
    ps=ports(n);cells=[(r,c) for r in range(n) for c in range(n)]
    return [(a,dict(zip(ps,[ray(n,a,p) for p in ps]))) for a in combinations(cells,count)]

def inspect(puzzle):
    n=puzzle['size'];atoms=puzzle['atoms'];shots=puzzle['shots'];query=puzzle['query']
    if type(n) is not int or n not in (3,4):raise ValueError('size must be 3 or 4')
    if not isinstance(atoms,list) or len(atoms) not in (2,3) or any(not isinstance(a,list) or len(a)!=2 or any(type(v) is not int or not 0<=v<n for v in a) for a in atoms):raise ValueError('Invalid atoms')
    atoms=tuple(sorted(tuple(a) for a in atoms))
    if len(set(atoms))!=len(atoms):raise ValueError('Duplicate atoms')
    ps=ports(n)
    if not isinstance(shots,list) or not 1<=len(shots)<=10 or len(set(shots))!=len(shots) or any(s not in ps for s in shots) or query not in ps or query in shots:raise ValueError('Invalid shots/query')
    actual={s:ray(n,atoms,s) for s in shots}
    matches=[(a,table) for a,table in world_table(n,len(atoms)) if all(table[s]==actual[s] for s in shots)]
    outcomes=sorted({table[query] for _,table in matches})
    return {'public':{'size':n,'atom_count':len(atoms),'observations':[{'entry':s,'outcome':actual[s]} for s in shots],'query':query},
        'truth':outcomes[0] if len(outcomes)==1 else 'UNKNOWN','possible_count':len(matches),'query_outcomes':outcomes,
        'layouts':[[list(p) for p in a] for a,_ in matches],'allowed_answers':['HIT','REFLECT','UNKNOWN']+ps}
