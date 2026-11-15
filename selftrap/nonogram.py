"""Real Nonograms: forward clue generation and exact bounded solution checking."""
from itertools import product
from time import perf_counter

class SearchLimit(Exception):pass

def clues(line):
    runs=[];length=0
    for value in line:
        if value=='1':length+=1
        elif length:runs.append(length);length=0
    if length:runs.append(length)
    return runs

def validate_bitmap(bitmap):
    if not isinstance(bitmap,list) or len(bitmap) not in {5,7}:raise ValueError('Bitmap must be 5x5 or 7x7')
    n=len(bitmap)
    if any(not isinstance(row,str) or len(row)!=n or set(row)-set('01') for row in bitmap):raise ValueError('Use 0/1 strings')
    return n

def puzzle(bitmap):
    n=validate_bitmap(bitmap)
    return {'size':n,'row_clues':[clues(row) for row in bitmap],
        'column_clues':[clues(''.join(bitmap[r][c] for r in range(n))) for c in range(n)]}

def patterns(length,description):
    return [''.join(bits) for bits in product('01',repeat=length) if clues(bits)==description]

def solve(p,limit=2,node_cap=30000,seconds=1.5,givens=()):
    n=p['size'];rows=[patterns(n,c) for c in p['row_clues']];cols=[patterns(n,c) for c in p['column_clues']]
    for r,c,v in givens:
        rows[r]=[s for s in rows[r] if int(s[c])==v]
        cols[c]=[s for s in cols[c] if int(s[r])==v]
    found=[];nodes=0;started=perf_counter()
    def search(r,grid,candidates):
        nonlocal nodes
        nodes+=1
        if nodes>node_cap or (nodes%64==0 and perf_counter()-started>seconds):raise SearchLimit
        if r==n:
            if puzzle(grid)!=p:raise RuntimeError('Independent clue replay disagrees')
            found.append(grid[:]);return
        for row in rows[r]:
            nexts=[[s for s in candidates[c] if s[r]==row[c]] for c in range(n)]
            if any(not options for options in nexts):continue
            search(r+1,grid+[row],nexts)
            if len(found)>=limit:return
    try:search(0,[],cols);status='COMPLETE'
    except SearchLimit:status='UNKNOWN'
    return {'status':status,'solutions':found,'nodes':nodes,'seconds':perf_counter()-started}

def inspect_bitmap(bitmap):
    p=puzzle(bitmap);s=solve(p)
    return {'unique':s['status']=='COMPLETE' and len(s['solutions'])==1,
        'row_patterns':[patterns(p['size'],c) for c in p['row_clues']],
        'column_patterns':[patterns(p['size'],c) for c in p['column_clues']]}

def validate_case(obj):
    required={'bitmap','query','predicted_wrong','diagnostic_hint','control_hint','p_wrong','diagnosis'}
    if not isinstance(obj,dict) or not required<=set(obj) or set(obj)-required-{'hint_forecasts'}:raise ValueError('Wrong generated case fields')
    bitmap=obj['bitmap'];n=validate_bitmap(bitmap);p=puzzle(bitmap);s=solve(p)
    if s['status']!='COMPLETE' or len(s['solutions'])!=1:raise ValueError('Need proven unique puzzle (UNKNOWN never counts)')
    def pos(v):return isinstance(v,list) and len(v)==2 and all(type(i) is int and 0<=i<n for i in v)
    for key in ['query','diagnostic_hint','control_hint']:
        if not pos(obj[key]):raise ValueError('Cell positions must be [row,column]')
    if len({tuple(obj[k]) for k in ['query','diagnostic_hint','control_hint']})!=3:raise ValueError('Query and both hints must be distinct')
    r,c=obj['query'];truth=int(bitmap[r][c])
    if type(obj['predicted_wrong']) is not int or obj['predicted_wrong']!=1-truth:raise ValueError('Declared error must be opposite to unique solution cell')
    if len(obj['p_wrong'])!=2 or any(type(v) not in {int,float} or not 0<=v<=1 for v in obj['p_wrong']):raise ValueError('Two probabilities required')
    if not isinstance(obj['diagnosis'],str) or len(obj['diagnosis'])>600:raise ValueError('Short diagnosis required')
    return {**obj,'puzzle':p,'truth':truth,'oracle':{'unique':True,'witness':s['solutions'][0],'nodes':s['nodes'],'seconds':s['seconds']}}

def actor_view(case,arm='baseline'):
    p=case['puzzle'];hints=[]
    if arm!='baseline':
        r,c=case['diagnostic_hint' if arm=='diagnostic' else 'control_hint']
        hints=[[r,c,int(case['bitmap'][r][c])]]
    n=p['size'];rows=[patterns(n,c) for c in p['row_clues']];cols=[patterns(n,c) for c in p['column_clues']]
    for r,c,v in hints:
        rows[r]=[s for s in rows[r] if int(s[c])==v];cols[c]=[s for s in cols[c] if int(s[r])==v]
    return {'puzzle':p,'query':case['query'],'revealed_cells':hints,
        'row_patterns_consistent_with_each_line':rows,'column_patterns_consistent_with_each_line':cols}

RULES='''Solve the queried cell of a Nonogram. Each row/column clue lists lengths
of consecutive BLACK runs in order. Runs are separated by at least one WHITE
cell; extra WHITE cells may occur anywhere. Empty clue means all WHITE.
1 means BLACK, 0 WHITE. Coordinates are zero-based [row,column]. The puzzle
has a mechanically verified UNIQUE solution. The supplied candidate strings
are all patterns allowed by EACH individual line, not complete-grid solutions.
Combine row and column constraints. Exactly one of 0/1 is correct at query.
Any revealed cells are verified true and must be respected. Return one cell
value through submit_cell, not a full bitmap. You may reason before choosing.'''


def validate_symmetric_case(obj):
    case=validate_case(obj);bitmap=case['bitmap'];n=len(bitmap);p=case['puzzle']
    if any(row!=row[::-1] for row in bitmap):raise ValueError('Every bitmap row must be mirror symmetric')
    if case['query'][1]!=n//2:raise ValueError('Query must lie on the fixed center column')
    dr,dc=case['diagnostic_hint'];cr,cc=case['control_hint']
    if dr!=cr or cc!=n-1-dc or dc==cc:raise ValueError('Hints must be distinct horizontal mirror counterparts')
    if p['column_clues']!=p['column_clues'][::-1] or any(c!=c[::-1] for c in p['row_clues']):raise RuntimeError('Clues not invariant under reflection')
    if bitmap[dr][dc]!=bitmap[cr][cc]:raise RuntimeError('Symmetric hint values differ')
    forecasts=case.get('hint_forecasts')
    for model in ['qwen3.5-397b','gemma-4-31b-it']:
        row=forecasts[model]
        if set(row)!={'diagnostic_correct','control_correct'} or any(type(v) not in {int,float} or not 0<=v<=1 for v in row.values()):raise ValueError('Explicit per-model hint probabilities required')
    case['symmetry_certificate']={'transformation':'column c -> n-1-c','puzzle_fixed':True,'query_fixed':True,'hints_swapped':True,'hint_bit_equal':True}
    return case
