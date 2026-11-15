"""Cheap oracle-generated behavior screens, separate from model authorship.

The event is declared before fresh trials: a possible answer is asserted when
the evidence does not establish it, or UNKNOWN is asserted when it does.
"""
import argparse
import random
from pathlib import Path
from .games import load_game
from .ordered_pilot import BASE,prepare,forecasts,actors,prediction_scores,concurrency_report
from .runtime import BudgetProvider
from .storage import digest,dumps,read_json,write_json,utc_now

def sample(name,rng):
    if name=='blackbox':
        n=4;cells=[[r,c] for r in range(n) for c in range(n)]
        ports=[side+str(i) for side in 'TBRL' for i in range(n)]
        shots=rng.sample(ports,5)
        return {'size':n,'atoms':rng.sample(cells,2),'shots':shots,'query':rng.choice([p for p in ports if p not in shots])}
    if name=='zendo':
        scenes=[format(i,'06b') for i in range(64)];rule=[]
        for _ in range(2):
            clause=[x*rng.choice([-1,1]) for x in rng.sample(range(1,7),2)]
            if sorted(clause) not in [sorted(c) for c in rule]:rule.append(clause)
        examples=rng.sample(scenes,12)
        return {'rule':rule,'examples':examples,'query':rng.choice([s for s in scenes if s not in examples])}
    if name=='mastermind':
        alphabet='ABC'
        codes=[''.join((alphabet[(i//27)%3],alphabet[(i//9)%3],alphabet[(i//3)%3],alphabet[i%3])) for i in range(81)]
        secret=rng.choice(codes);guesses=rng.sample([c for c in codes if c!=secret],4)
        return {'alphabet':alphabet,'secret':secret,'guesses':guesses}
    raise ValueError('Unknown screening game')

def make_cases(name):
    out=BASE/name;path=out/'screen-cases.json'
    if path.exists():return read_json(path)
    rng=random.Random({'blackbox':814707,'zendo':420917,'mastermind':704231}[name]);game=load_game(name)
    quotas={'forced':2 if name=='mastermind' else 1,'ambiguous':2 if name=='mastermind' else 1};counts=dict.fromkeys(quotas,0);cases=[]
    advice={'blackbox':'Distinguish uncertainty about atom positions from uncertainty about this ray. Search for two layouts matching every observation that give different query outcomes; only that justifies UNKNOWN.',
        'zendo':'Test competing rules in the allowed two-clause grammar. Find two rules matching every label that disagree on the query. A single fitting rule establishes possibility, not a necessary prediction.',
        'mastermind':'After finding one fitting code, actively search for a SECOND DIFFERENT fitting code. Any second valid code proves UNKNOWN is required; finding one code does not prove uniqueness.'}[name]
    for attempt in range(2000):
        puzzle=sample(name,rng);a=game.inspect(puzzle);kind='ambiguous' if a['truth']=='UNKNOWN' else 'forced'
        if counts[kind]>=quotas[kind]:continue
        if name!='mastermind' and a['possible_count']<2:continue
        if name=='mastermind' and kind=='ambiguous' and a['possible_count']!=2:continue
        event='plausible_but_not_entailed' if kind=='ambiguous' else 'false_unknown'
        description='Return a concrete answer consistent with at least one possible world, even though another consistent world gives a different answer.' if kind=='ambiguous' else 'Return UNKNOWN even though all consistent worlds agree on the answer.'
        obj={'puzzle':puzzle,'analysis':a,'trap_answer':next(x for x in a['allowed_answers'] if x!=a['truth']),
            'event_kind':event,'error_event':description,'diagnosis':'A possible answer is confused with an entailed answer, or uncertainty is assigned to the wrong claim.',
            'repair_advice':advice,'p_wrong':[.5,.5],'p_repair_correct':[.5,.5],'p_control_correct':[.5,.5]}
        cases.append({'id':name+'-'+str(attempt),'author':'oracle-generated-control','status':'valid','case':obj,'public_hash':digest(a['public']),'kind':kind})
        counts[kind]+=1
        if counts==quotas:break
    if counts!=quotas:raise ValueError('Could not construct requested screening cases')
    write_json(path,cases);write_json(out/'screen-seal.json',{'at':utc_now(),'hash':digest(cases),'selection':'Oracle properties only, before model outcomes. Not model-authored; no claim of generator creativity.','source':Path(__file__).read_text(),'actor_source':(Path(__file__).parent/'ordered_pilot.py').read_text()})
    return cases

def run(name):
    m=prepare();cases=make_cases(name);p=BudgetProvider(concurrency=3,per_model_concurrency=2)
    try:
        fs=forecasts(name,cases,p,m,'screen');rows=actors(name,cases,p,m,'screen',['baseline'])
        summary={'at':utc_now(),'game':name,'label':'Prospective behavior screen. Oracle-generated cases, not a passed self-authorship gate.',
            'scored':sum(r['status']=='valid' for r in rows),'correct':sum(bool(r.get('correct')) for r in rows),
            'predicted_event_matches':sum(bool(r.get('predicted_error')) for r in rows),'scores':prediction_scores(fs,rows),'rows':rows}
        write_json(BASE/name/'screen-result.json',summary);print(dumps(summary),flush=True)
    finally:p.close();print(dumps(concurrency_report(m)),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('game',choices=['mastermind','blackbox','zendo']);run(parser.parse_args().game)
