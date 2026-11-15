"""Minesweeper claims are entailed over every board matching the revealed clues."""
from itertools import combinations

RULES='''Minesweeper. Exactly mine_count distinct cells on the size*size grid
are mines. Coordinates [row,column] start at zero. Each revealed cell is SAFE
and its clue counts mines in its up to eight neighboring cells, diagonals
included. All other cells are unrevealed. The total mine count is known.
For the queried unrevealed cell, submit SAFE if every consistent board makes
it safe, MINE if every consistent board makes it a mine, otherwise UNKNOWN.
A good guess or the most likely board does not establish a forced value.
Use submit_answer(answer=...). No external execution tools.'''

AUTHOR='''generate(seed) builds puzzle={"size":5,"mines":[[0,1],[2,3],[4,4]],
"revealed":[[0,0],[1,2],...],"query":[2,2]}. Size 4 or 5; 2 or 3 distinct
mines; 1..12 distinct revealed cells, none a mine; query unrevealed.
inspect(puzzle) computes truthful neighbor clues, all consistent boards and
the entailed query answer SAFE/MINE/UNKNOWN. It returns public, truth,
possible_count, query_outcomes, layouts and allowed_answers. Your program
chooses the pattern, reveals, query and exact trap_answer. Target a specific
local-versus-joint deduction, total-mine-count error, or false certainty.'''

def neighbors(n,p):
    r,c=p
    return {(a,b) for a in range(max(0,r-1),min(n,r+2)) for b in range(max(0,c-1),min(n,c+2)) if (a,b)!=p}

def inspect(puzzle):
    n=puzzle['size'];mines=puzzle['mines'];revealed=puzzle['revealed'];query=puzzle['query']
    if type(n) is not int or n not in (4,5):raise ValueError('size must be 4 or 5')
    for name,cells in [('mines',mines),('revealed',revealed),('query',[query])]:
        if not isinstance(cells,list) or any(not isinstance(p,list) or len(p)!=2 or any(type(v) is not int or not 0<=v<n for v in p) for p in cells):raise ValueError('Invalid '+name)
        if len({tuple(p) for p in cells})!=len(cells):raise ValueError('Duplicate '+name)
    if len(mines) not in (2,3) or not 1<=len(revealed)<=12:raise ValueError('Invalid mine/reveal count')
    mines=set(map(tuple,mines));revealed=list(map(tuple,revealed));query=tuple(query)
    if any(p in mines for p in revealed) or query in revealed:raise ValueError('A revealed cell is mined or query is revealed')
    constraints=[(p,neighbors(n,p)) for p in revealed]
    clues=[(p,ns,len(ns&mines)) for p,ns in constraints]
    unknown=[(r,c) for r in range(n) for c in range(n) if (r,c) not in revealed]
    layouts=[]
    for candidate in combinations(unknown,len(mines)):
        world=set(candidate)
        if all(len(ns&world)==value for _,ns,value in clues):layouts.append(candidate)
    outcomes={('MINE' if query in a else 'SAFE') for a in layouts}
    return {'public':{'size':n,'mine_count':len(mines),'revealed':[{'cell':list(p),'count':v} for p,_,v in clues],'query':list(query)},
        'truth':next(iter(outcomes)) if len(outcomes)==1 else 'UNKNOWN','possible_count':len(layouts),'query_outcomes':sorted(outcomes),
        'layouts':[[list(p) for p in a] for a in layouts],'allowed_answers':['SAFE','MINE','UNKNOWN']}
