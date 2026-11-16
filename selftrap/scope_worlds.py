"""Composed resource claims: per-instance feasibility versus a stable resource.

Finite three-way capability relations give exact labels and replayable witnesses.
"""

from itertools import permutations, product
import random

from .storage import digest


MODES=('stable_per_job','shared_machine','common_shift','common_machine_per_shift')
ORDERS={
 'stable_per_job':(((0,'a'),(2,'a'),(1,'e')),((0,'a'),(1,'e'),(2,'a'))),
 'shared_machine':(((0,'a'),(1,'e'),(2,'a')),((1,'e'),(0,'a'),(2,'a'))),
 'common_shift':(((0,'a'),(2,'e'),(1,'e')),((2,'e'),(0,'a'),(1,'e'))),
 'common_machine_per_shift':(((2,'a'),(0,'a'),(1,'e')),((2,'a'),(1,'e'),(0,'a')))}


def predicate(case,bits,order):
    dims=case['dims'];assignment=[0,0,0]
    def step(level):
        if level==3:return bits[(assignment[0]*dims[1]+assignment[1])*dims[2]+assignment[2]]
        axis,kind=order[level]
        values=[]
        for value in range(dims[axis]):
            assignment[axis]=value;found=step(level+1)
            if kind=='a' and not found:return False
            if kind=='e' and found:return True
            values.append(found)
        return all(values) if kind=='a' else any(values)
    return step(0)


def decode(case,index):
    n,m,s=case['dims'];job=index//(m*s);machine=(index//s)%m;shift=index%s
    return f'machine M{machine+1} can execute job J{job+1} in shift S{shift+1}'


def facts(case):
    return [f"{'TRUE' if v else 'FALSE'}: {decode(case,int(i))}." for i,v in sorted(case['facts'].items(),key=lambda p:int(p[0]))]


def context(case):
    n,m,s=case['dims']
    return (f'Exactly {n} jobs J1..J{n}, {m} machines M1..M{m}, and {s} shifts S1..S{s} exist. '
        'For each job/machine/shift triple, capability is a Boolean fact. There are no capacity '
        'conflicts or unstated constraints. Execute/can execute/capable of executing refer to '
        'this same capability relation, not to an actual allocation.\nKnown facts:\n'+'\n'.join(facts(case)))


def texts(case):
    mode=case['mode']
    src={
      'stable_per_job':'For every job and every shift, at least one machine can execute that job in that shift.',
      'shared_machine':'For every job, there is a machine that can execute that job in every shift.',
      'common_shift':'For every job, there is a shift in which at least one machine can execute that job.',
      'common_machine_per_shift':'In every shift, every job is executable by at least one machine.'}[mode]
    wrong={
      'stable_per_job':['For every job, there is a machine that can execute that job in every shift.',
                        'Every job has a machine capable of executing it throughout all shifts.',
                        'For each job, one machine suffices across the entire set of shifts.',
                        'For each job, all its shifts are supported by a common capable machine.'],
      'shared_machine':['There is a machine that can execute every job in every shift.',
                        'One common machine is capable of every job throughout all shifts.',
                        'A machine exists that suffices for each job and each shift.',
                        'All jobs and shifts are covered by one machine capable of executing them.'],
      'common_shift':['There is a shift in which every job is executable by at least one machine.',
                       'In one common shift, every job has at least one capable machine.',
                       'A shift exists that supports each job on some capable machine.',
                       'All jobs have a capable machine within one shared shift.'],
      'common_machine_per_shift':['In every shift, there is a machine capable of executing every job.',
                       'Each shift has a common machine capable of all jobs in that shift.',
                       'For each shift, one machine suffices for every job.',
                       'Every shift has a machine whose capabilities cover all jobs.']}[mode]
    valid={
      'stable_per_job':'Given any job and any shift, some machine is capable of that job in that shift.',
      'shared_machine':'For each job, at least one machine is capable of that job across the entire set of shifts.',
      'common_shift':'Each job has at least one machine/shift combination in which it is executable.',
      'common_machine_per_shift':'For every shift and every job, a capable machine exists for that job in that shift.'}[mode]
    return src,wrong,valid


def candidates(case):
    _,wrong,_=texts(case);known=' '.join(facts(case));result=[]
    for claim in wrong:
        result.extend([claim,'Summary:\n'+claim,known+' '+claim,known+' '+claim+' '+known])
    return result


def canonical(case):
    n,m,s=case['dims'];known={int(i):v for i,v in case['facts'].items()};forms=[]
    for a in permutations(range(n)):
      for b in permutations(range(m)):
       for c in permutations(range(s)):
        forms.append(tuple(known.get((i*m+j)*s+k,-1) for i in a for j in b for k in c))
    # Two modes have the same universal/universal/existential source. Treat
    # an isomorphic context as the same cluster even when its target differs.
    source_class='each_job_each_shift' if case['mode'] in ('stable_per_job','common_machine_per_shift') else case['mode']
    return digest([source_class,case['dims'],min(forms)])


def oracle(case,valid=False):
    n,m,s=case['dims'];a,b=ORDERS[case['mode']];worlds=0;witness=None
    for bits in product((False,True),repeat=n*m*s):
        if any(bits[int(i)]!=v for i,v in case['facts'].items()):continue
        worlds+=1;x=predicate(case,bits,a);y=x if valid else predicate(case,bits,b)
        if x!=y and witness is None:
            witness={'assignments':[{'fact':decode(case,i),'value':v} for i,v in enumerate(bits)],'source':x,'rewrite':y}
    return {'equivalent':witness is None,'permitted_worlds':worlds,'counterexample':witness}


def panel(seed,size,*,exclude=(),large=False):
    rng=random.Random(seed);seen=set(exclude);rows=[];attempts=0
    while len(rows)<size:
        attempts+=1
        if attempts>size*3000:raise ValueError('Not enough distinct capability contexts')
        mode=MODES[len(rows)%len(MODES)];dims=[rng.choice((2,3)) if large else 2,2,2]
        total=dims[0]*4;count=rng.randrange(1,total//2+1)
        c={'id':f'{seed}-{len(rows):03d}','mode':mode,'dims':dims,
            'facts':{str(i):bool(rng.randrange(2)) for i in rng.sample(range(total),count)},
            'random_variant':rng.randrange(16)}
        c['semantic_id']=canonical(c)
        if c['semantic_id'] in seen or oracle(c)['equivalent']:continue
        seen.add(c['semantic_id']);rows.append(c)
    return rows
