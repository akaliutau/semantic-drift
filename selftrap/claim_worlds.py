"""Controlled English claims with exact finite-world equivalence certificates.

The generated universe is explicit. Gold labels compare every permitted world,
not embedding similarity or another model's interpretation of arbitrary prose.
"""

from itertools import permutations, product
import random

from .storage import digest


FAMILIES=('quantifier_order','necessary_sufficient','not_all_none','separate_joint','inclusive_bound')


def truth(family,form,n,m,k,bits):
    if family=='quantifier_order':
        if form=='source':return all(any(bits[i*m+j] for j in range(m)) for i in range(n))
        return any(all(bits[i*m+j] for i in range(n)) for j in range(m))
    if family=='necessary_sufficient':
        if form=='source':return all(not bits[i*2] or bits[i*2+1] for i in range(n))
        return all(not bits[i*2+1] or bits[i*2] for i in range(n))
    if family=='not_all_none':
        return not all(bits) if form=='source' else not any(bits)
    if family=='separate_joint':
        if form=='source':return any(bits[i*2] for i in range(n)) and any(bits[i*2+1] for i in range(n))
        return any(bits[i*2] and bits[i*2+1] for i in range(n))
    if family=='inclusive_bound':return sum(bits)>=k if form=='source' else sum(bits)>k
    raise ValueError('Unknown semantic family')


def width(family,n,m):
    return n*m if family=='quantifier_order' else n*2 if family in ('necessary_sufficient','separate_joint') else n


def canonical(case):
    """Group aliases and row/column permutations into the same semantic instance."""
    f,n,m=case['family'],case['n'],case['m']
    columns=m if f=='quantifier_order' else 2 if f in ('necessary_sufficient','separate_joint') else 1
    row_orders=permutations(range(n))
    contexts=[];known={int(i):v for i,v in case['facts'].items()}
    for rows in row_orders:
        col_orders=permutations(range(columns)) if f=='quantifier_order' else [tuple(range(columns))]
        for cols in col_orders:
            contexts.append(tuple(known.get(i*columns+j,-1) for i in rows for j in cols))
    return digest([f,n,m,case['k'],min(contexts)])


def atom(case,index):
    f=case['family'];n,m=case['n'],case['m']
    if f=='quantifier_order':return f"key K{index%m+1} opens lock L{index//m+1}"
    if f=='necessary_sufficient':return f"device D{index//2+1} has property {'P' if index%2==0 else 'Q'}"
    if f=='separate_joint':return f"in state S{index//2+1}, lamp {'A' if index%2==0 else 'B'} is on"
    return f"packet T{index+1} passed inspection"


def context(case):
    f,n,m,k=case['family'],case['n'],case['m'],case['k']
    if f=='quantifier_order':domain=f"Exactly {n} locks L1..L{n} and {m} keys K1..K{m} exist. Opens is a Boolean capability relation."
    elif f=='necessary_sufficient':domain=f"Exactly {n} devices D1..D{n} exist; each has or lacks each Boolean property P and Q."
    elif f=='separate_joint':domain=f"Exactly {n} permissible states S1..S{n} exist. In each, each of lamps A and B is on or off. Possible means in at least one of these states."
    else:domain=f"Exactly {n} packets T1..T{n} exist. Each either passed or failed inspection, with no third outcome."
    facts=[f"{'TRUE' if value else 'FALSE'}: {atom(case,int(i))}." for i,value in sorted(case['facts'].items(),key=lambda p:int(p[0]))]
    return domain+'\nKnown facts:\n'+('\n'.join(facts) if facts else 'None. Other Boolean assignments remain possible.')


def statements(case,kind):
    f,n,m,k=case['family'],case['n'],case['m'],case['k']
    source={
        'quantifier_order':['Every lock can be opened by at least one key.','For each lock, a key exists that opens that lock.'],
        'necessary_sufficient':['For every device, having P is sufficient for having Q.','Every device that has P also has Q.'],
        'not_all_none':['Not every packet passed inspection.','At least one packet failed inspection.'],
        'separate_joint':['Lamp A is on in at least one permissible state, and lamp B is on in at least one permissible state.',
                          'It is possible for lamp A to be on, and it is possible for lamp B to be on.'],
        'inclusive_bound':[f'At least {k} packets passed inspection.',f'The number of packets that passed is no less than {k}.']}
    wrong={
        'quantifier_order':['At least one key opens every lock.','All the locks can be opened with one common key.',
                            'A key exists that is sufficient to open all locks.','There is a single key capable of opening each lock.'],
        'necessary_sufficient':['For every device, having P is necessary for having Q.','Every device that has Q also has P.',
                                'A device can have Q only if it has P.','No device lacking P has Q.'],
        'not_all_none':['No packet passed inspection.','Every packet failed inspection.',
                        'There is not a single packet that passed inspection.','The set of packets that passed is empty.'],
        'separate_joint':['There is a permissible state in which both lamps A and B are on.',
                          'It is possible for lamps A and B to be on together.',
                          'Both lamps can be on in the same permissible state.',
                          'Some permissible state has lamp A on and lamp B on.'],
        'inclusive_bound':[f'More than {k} packets passed inspection.',f'The number of packets that passed exceeds {k}.',
                           f'At least {k+1} packets passed inspection.',f'Strictly more than {k} packets passed inspection.']}
    if kind=='source':return source[f]
    if kind=='wrong':return wrong[f]
    if kind=='valid':
        return {
            'quantifier_order':['For every lock, there is at least one key that can open it.'],
            'necessary_sufficient':['Every device either lacks P or has Q.','A device lacking Q cannot have P.'],
            'not_all_none':['The inspection was failed by at least one of the packets.'],
            'separate_joint':['Some permissible state has lamp A on; some permissible state has lamp B on.'],
            'inclusive_bound':[f'No fewer than {k} packets passed inspection.']}[f]
    raise ValueError('Unknown statement kind')


def variants(case):
    forms=[]
    for phrase in statements(case,'wrong'):
        forms.extend([phrase, 'Finding: '+phrase, '\n'.join(['Summary:',phrase])])
    return forms


def oracle(case,target='wrong'):
    count=width(case['family'],case['n'],case['m'])
    allowed=[];witnesses=[]
    for bits in product((False,True),repeat=count):
        if any(bits[int(i)]!=value for i,value in case['facts'].items()):continue
        source=truth(case['family'],'source',case['n'],case['m'],case['k'],bits)
        other=source if target=='valid' else truth(case['family'],'wrong',case['n'],case['m'],case['k'],bits)
        allowed.append((source,other))
        if source!=other:witnesses.append({'assignments':[{'fact':atom(case,i),'value':v} for i,v in enumerate(bits)],
                                        'source':source,'rewrite':other})
    if not allowed:raise ValueError('Inconsistent background facts')
    witness=next((w for w in witnesses if w['source'] and not w['rewrite']),witnesses[0] if witnesses else None)
    return {'equivalent':not witnesses,'permitted_worlds':len(allowed),'counterexample':witness}


def make_panel(seed,size,*,exclude=(),pilot=False):
    rng=random.Random(seed);seen=set(exclude);result=[];attempts=0
    while len(result)<size:
        attempts+=1
        if attempts>size*1000:raise ValueError('Not enough nonisomorphic cases')
        family=FAMILIES[len(result)%len(FAMILIES)]
        n=(rng.choice((2,3)) if family=='quantifier_order' else
           rng.choice((2,3,4)) if family in ('necessary_sufficient','separate_joint') else
           rng.choice((2,3,4,5,6,7,8)))
        m=rng.choice((2,3)) if family=='quantifier_order' else 1
        k=rng.randrange(1,n) if family=='inclusive_bound' else 0
        count=width(family,n,m);fixed=rng.randrange(0,min(count-2,count//2)+1)
        facts={str(i):bool(rng.randrange(2)) for i in rng.sample(range(count),fixed)}
        case={'id':f'{seed}-{len(result):03d}','family':family,'n':n,'m':m,'k':k,'facts':facts,
              'source_variant':rng.randrange(2),'baseline_variant':rng.randrange(12)}
        case['semantic_id']=canonical(case)
        if case['semantic_id'] in seen or oracle(case)['equivalent']:continue
        seen.add(case['semantic_id']);result.append(case)
    return result


def verify_witness(case,witness):
    if witness is None:return False
    bits=[x['value'] for x in witness['assignments']]
    if len(bits)!=width(case['family'],case['n'],case['m']):return False
    if [x['fact'] for x in witness['assignments']]!=[atom(case,i) for i in range(len(bits))]:return False
    if any(bits[int(i)]!=v for i,v in case['facts'].items()):return False
    a=truth(case['family'],'source',case['n'],case['m'],case['k'],bits)
    b=truth(case['family'],'wrong',case['n'],case['m'],case['k'],bits)
    return a!=b and a==witness['source'] and b==witness['rewrite']
