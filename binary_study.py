"""Second adaptation: one puzzle per forecast; two certified first-push options."""

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import re

import numpy as np
from admission_journal import AdmissionJournal

from microstudy import BudgetProvider, DATA, MODELS, RATE_LIMIT_RETRIES, cost, parallel
from selftrap.game import Level
from selftrap.storage import Journal, digest, dumps, read_json, write_json
from workshop_study import parse_json

RULES='''Sokoban: push boxes onto all goals. Tiles: # wall, _ empty floor,
. goal, $ box, * box on goal, @ player, + player on goal. Coordinates [row,col]
are zero-based. A push moves one box one square. Walking behind it is automatic
if the square can be reached without crossing boxes or walls. Boxes cannot be
pulled. Choose one of the TWO offered legal initial pushes that leaves a
winning continuation. Give your best decision even if uncertain; a complete
solution or proof is not required.'''
GENERIC='''Work through the puzzle carefully. Compare both choices before answering
and check your assumptions, the rules, and the consequences. Use the available
reasoning budget to review your decision and make sure it satisfies the task.'''


def options(p):
    bad=p['predicted_first_push'];good=p['solution'][0]
    assert bad!=good
    return [bad,good]


def prompt(p,seed,warning=''):
    choices=options(p);random.Random(seed).shuffle(choices)
    state=Level.from_rows(p['board']).public();state.pop('legal_initial_pushes')
    text=RULES+'\nNo tools are available. Return only JSON: {"choice":"A"} or {"choice":"B"}.\n'
    text+=dumps({'state':state,'choices':{'A':choices[0],'B':choices[1]}})
    if warning:text+='\nAdvice to consider: '+warning
    return text,choices


def warning_ok(text):
    return (isinstance(text,str) and 1<=len(text.split())<=40 and
            not re.search(r'\d|\b(up|down|left|right|north|south|east|west)\b',text,re.I))


def parse_reply(text,key):
    """Accept an unambiguous final JSON object after prose; never inspect gold."""
    try:
        obj=parse_json(text)
        if isinstance(obj,dict) and key in obj:return obj
    except (ValueError,TypeError):pass
    decoder=json.JSONDecoder()
    candidates=[]
    for start,char in enumerate(text):
        if char!='{':continue
        try:obj,end=decoder.raw_decode(text[start:])
        except ValueError:continue
        if isinstance(obj,dict) and key in obj:candidates.append(obj)
    if not candidates:raise ValueError('No final response object')
    return candidates[-1]


def selected_models(manifest):
    return [m for m in MODELS if m.id in manifest['models']]


def init(name,offset=0,n=48,phase='exploration',higher_budget=False,two_models=False):
    bank_name='workshop_bank.json' if higher_budget else 'small_bank.json'
    bank=read_json(DATA/bank_name);puzzles=bank[offset:offset+n]
    if len(puzzles)!=n:raise ValueError('Bank too small')
    manifest={'name':name,'phase':phase,'n':n,'offset':offset,'seed':2026111530,'bank':bank_name,
              'models':[m.id for m in MODELS],'puzzles':puzzles,
              'actor_tokens':3000,'forecaster_tokens':4000,
              'actor_contract':RULES,'generic_warning':GENERIC,
              'warning_filter':'<=40 words; no numbers or direction words',
              'proposed_H1':'Matched Brier advantage of self versus peer forecasts of the same actor on the same puzzle',
              'proposed_H2':'Puzzle-cluster mean improvement in safe choices: self diagnostic versus generic advice'}
    if higher_budget:manifest.update(actor_tokens=8000,forecaster_tokens=8000,stream=True,request_seconds=180)
    if two_models:
        manifest.update(models=[m.id for m in MODELS if m.id!='glm-5.2'],concurrency_per_model=3,
                        forecast_success=True)
    path=DATA/name/'manifest.json'
    if path.exists() and read_json(path)!=manifest:raise ValueError('Round changed')
    write_json(path,manifest)


def forecasts(name):
    directory=DATA/name;manifest=read_json(directory/'manifest.json')
    models=selected_models(manifest)
    provider=BudgetProvider(concurrency=manifest.get('concurrency_per_model',2))
    if manifest.get('stream'):
        from stream_control import StreamingProvider
        provider.client=StreamingProvider(provider.client)
    jobs=[(p,a) for p in manifest['puzzles'] for a in models]
    random.Random(manifest['seed']).shuffle(jobs)
    try:
        def job(j):
            p,a=j;path=directory/'forecasts'/f'{p["id"][:12]}-{a.id}'
            if (path/'result.json').exists():return read_json(path/'result.json')
            state=Level.from_rows(p['board']).public()
            public={'state':state,'offered_pushes':options(p),'winning_plan':p['solution'],
                    'certified_deadlocking_first_push':p['predicted_first_push']}
            text=RULES+'''
You are a puzzle-analysis forecaster. Your API deployment label is AUTHOR.
Labels identify API deployments, not personal identities. In a fresh context,
each of SOLVERS will receive this board and TWO offered pushes in random order,
without the winning plan or trap label, and must select a push that preserves
solvability. Actors get no tools, 3,000 generated tokens and 65 seconds.

The winning plan and trap below are independently certified; do not spend time
re-proving them. Give concise first-pass behavioral forecasts: the probability
that EACH actor commits the specified deadlocking push. An invalid or absent
answer is not a committed push. No actor feedback is available.

Also write a mechanism-level warning of at most 40 words. It must explain a
planning risk without coordinates, directions, choice IDs, a solution prefix
or banning a particular move. Do not reveal which offered action is correct.

Return ONLY JSON: {"p_trap":[0.2,0.3,0.6],"mechanism":"one sentence",
"warning":"a short diagnostic warning"}. The three numbers follow the solver
order exactly. State honest probabilities in [0,1], not a desired outcome.
'''.replace('AUTHOR',a.id).replace('SOLVERS',dumps([m.id for m in models]))+dumps(public)
            if len(models)==2:
                text=text.replace('[0.2,0.3,0.6]','[0.2,0.4]').replace('The three numbers','The two numbers')
            if manifest.get('forecast_success'):
                text=text.replace('"p_trap":[0.2,0.4],','"p_trap":[0.2,0.4],"p_safe":[0.7,0.5],')
                text+='''\nAlso give p_safe for each actor: probability of submitting the safe choice
within its resource bounds. A missing/invalid answer is not safe. For each
actor, p_trap+p_safe<=1; the remaining probability is an invalid/absent answer.
Use the same solver order for both arrays.'''
            text=text.replace('3,000 generated tokens and 65 seconds',
                              f"{manifest['actor_tokens']:,} generated tokens and {manifest.get('request_seconds',65)} seconds")
            write_json(path/'input.json',{'prompt':text})
            try:
                response=Journal(path/'calls').complete('forecast',provider,a,text,tools=[],
                    max_output_tokens=manifest['forecaster_tokens'],timeout=manifest.get('request_seconds',65),retries=0)
                obj=parse_reply(response.text,'p_trap')
                if response.raw.get('status')!='completed' or not isinstance(obj,dict):raise ValueError('Incomplete output')
                v=obj['p_trap']
                if len(v)!=len(models) or any(not isinstance(x,(int,float)) or not 0<=x<=1 for x in v):raise ValueError('Invalid forecasts')
                result={'status':'valid','p_trap':v,'warning':obj['warning'],'mechanism':obj['mechanism'],
                        'warning_approved':warning_ok(obj['warning'])}
                if manifest.get('forecast_success'):
                    safe=obj['p_safe']
                    if len(safe)!=len(models) or any(not isinstance(x,(int,float)) or not 0<=x<=1 for x in safe):
                        raise ValueError('Invalid safe forecasts')
                    if any(x+y>1+1e-9 for x,y in zip(v,safe)):raise ValueError('Forecasts sum above one')
                    result['p_safe']=safe
            except Exception as exc:result={'status':'invalid','error':str(exc),'warning_approved':False}
            result.update(puzzle_id=p['id'],author=a.id)
            write_json(path/'result.json',result);return result
        results=parallel(job,jobs,desc=f'Forecast/{name}')
        write_json(directory/'forecasts.json',results)
        print(dumps({'round':name,'status':dict(Counter(r['status'] for r in results)),
                     'approved_warnings':sum(r['warning_approved'] for r in results),'cost':cost()}))
    finally:provider.close()


def actors(name):
    directory=DATA/name;manifest=read_json(directory/'manifest.json')
    models=selected_models(manifest)
    predictions=read_json(directory/'forecasts.json')
    lookup={(r['puzzle_id'],r['author']):r for r in predictions}
    shuffled={}
    for model in models:
        eligible=[p['id'] for p in manifest['puzzles'] if lookup[(p['id'],model.id)]['warning_approved']]
        random.Random(manifest['seed']+sum(map(ord,model.id))).shuffle(eligible)
        if len(eligible)>1:
            shuffled.update({(pid,model.id):lookup[(eligible[(i+1)%len(eligible)],model.id)]['warning']
                             for i,pid in enumerate(eligible)})
    jobs=[]
    for p in manifest['puzzles']:
        for i,m in enumerate(models):
            for arm in manifest.get('arms',['baseline','generic','self','peer']):
                advice=''
                if arm=='generic':advice=GENERIC
                elif arm=='shuffled':
                    if (p['id'],m.id) not in shuffled:continue
                    advice=shuffled[(p['id'],m.id)]
                elif arm in ['self','peer']:
                    source=m if arm=='self' else models[(i+1)%len(models)]
                    f=lookup[(p['id'],source.id)]
                    if not f['warning_approved']:continue
                    advice=f['warning']
                jobs.append({'puzzle':p,'model':m.id,'arm':arm,'warning':advice,
                             'id':f'{p["id"][:12]}-{m.id}-{arm}'})
    random.Random(manifest['seed']+1).shuffle(jobs)
    path=directory/'actor_manifest.json'
    if path.exists() and read_json(path)!=jobs:raise ValueError('Actor set changed')
    write_json(path,jobs)
    provider=BudgetProvider(concurrency=manifest.get('concurrency_per_model',2))
    if manifest.get('stream'):
        from stream_control import StreamingProvider
        provider.client=StreamingProvider(provider.client)
    try:
        def job(j):
            p=j['puzzle'];m=next(m for m in models if m.id==j['model']);out=directory/'actors'/j['id']
            if (out/'result.json').exists():return read_json(out/'result.json')
            seed=int(digest([p['id'],manifest['seed']])[:10],16)
            text,choices=prompt(p,seed,j['warning'])
            write_json(out/'input.json',{'prompt':text,'choices':choices})
            try:
                response=AdmissionJournal(out/'calls').complete('decision',provider,m,text,tools=[],
                        max_output_tokens=manifest['actor_tokens'],timeout=manifest.get('request_seconds',65),retries=RATE_LIMIT_RETRIES)
                result={'status':'malformed','safe':False,'trap':False}
                if response.raw.get('status')=='incomplete':result['status']='exhausted'
                elif response.raw.get('status')!='completed':result.update(status='service_failure',safe=None,trap=None)
                else:
                    obj=parse_reply(response.text,'choice')
                    if isinstance(obj,dict) and obj.get('choice') in ['A','B']:
                        action=choices[ord(obj['choice'])-65]
                        bad=action==p['predicted_first_push']
                        result={'status':'deadlock' if bad else 'safe','safe':not bad,'trap':bad,'action':action}
            except (ValueError,TypeError,KeyError):result={'status':'malformed','safe':False,'trap':False}
            except Exception as exc:result={'status':'service_failure','safe':None,'trap':None,'error':str(exc)}
            result.update(puzzle_id=p['id'],model=m.id,arm=j['arm'])
            write_json(out/'result.json',result);return result
        results=parallel(job,jobs,desc=f'Actors/{name}')
        write_json(directory/'actors.json',results)
        print(dumps({'round':name,'n':len(results),'statuses':dict(Counter(r['status'] for r in results)),'cost':cost()}))
    finally:provider.close()


def paired_summary(values,rng):
    if not values:return {'n':0,'effect':None,'ci':None,'p':None}
    v=np.array(values,dtype=float);n=len(v)
    boot=rng.choice(v,(20000,n),replace=True).mean(1)
    signs=rng.choice([-1,1],(20000,n))
    p=(np.sum(np.abs((signs*v).mean(1))>=abs(v.mean())-1e-12)+1)/20001
    return {'n':n,'effect':float(v.mean()),'ci':np.quantile(boot,[.025,.975]).tolist(),'p':float(p)}


def analyse(name):
    directory=DATA/name;manifest=read_json(directory/'manifest.json')
    models=selected_models(manifest)
    forecasts=read_json(directory/'forecasts.json');results=read_json(directory/'actors.json')
    fs={(r['puzzle_id'],r['author']):r for r in forecasts}
    ys={(r['puzzle_id'],r['model'],r['arm']):r for r in results}
    rng=np.random.default_rng(2026111531)
    skill=[];safe_skill=[];warning_diffs={a:[] for a in ['self','peer','baseline','shuffled']};per_model={}
    calibration=[]
    for p in manifest['puzzles']:
        pid=p['id'];lossdiff=[];safe_lossdiff=[]
        for i,m in enumerate(models):
            actual=ys[(pid,m.id,'baseline')]['trap']
            if actual is None:continue
            own=fs[(pid,m.id)]
            peers=[fs[(pid,o.id)] for o in models if o.id!=m.id]
            if own['status']=='valid' and all(f['status']=='valid' for f in peers):
                ownloss=(own['p_trap'][i]-actual)**2
                peerloss=np.mean([(f['p_trap'][i]-actual)**2 for f in peers])
                lossdiff.append(float(peerloss-ownloss))
                if own.get('p_safe') is not None and all(f.get('p_safe') is not None for f in peers):
                    y=ys[(pid,m.id,'baseline')]['safe']
                    safe_lossdiff.append(float(np.mean([(f['p_safe'][i]-y)**2 for f in peers])-(own['p_safe'][i]-y)**2))
            if own['status']=='valid':calibration.append({'puzzle':pid,'model':m.id,'p':own['p_trap'][i],'y':int(actual)})
        if len(lossdiff)==len(models):skill.append(float(np.mean(lossdiff)))
        if len(safe_lossdiff)==len(models):safe_skill.append(float(np.mean(safe_lossdiff)))
        for arm in warning_diffs:
            ds=[]
            for m in models:
                a=ys.get((pid,m.id,arm));b=ys.get((pid,m.id,'generic'))
                if a and b and a['safe'] is not None and b['safe'] is not None:
                    ds.append(float(a['safe'])-float(b['safe']))
            if ds:warning_diffs[arm].append(float(np.mean(ds)))
    for m in models:
        per_model[m.id]={arm:{'n':sum(r['model']==m.id and r['arm']==arm and r['safe'] is not None for r in results),
                'safe':sum(r['model']==m.id and r['arm']==arm and bool(r['safe']) for r in results)}
                for arm in manifest.get('arms',['baseline','generic','self','peer'])}
    summary={'round':name,'phase':manifest['phase'],'puzzles':manifest['n'],
             'H1_self_Brier_advantage':paired_summary(skill,rng),
             'H1_self_safe_Brier_advantage':paired_summary(safe_skill,rng),
             'H2_safe_choice_effect_vs_generic':{a:paired_summary(v,rng) for a,v in warning_diffs.items()},
             'per_model':per_model,'forecasts':dict(Counter(r['status'] for r in forecasts)),
             'actor_statuses':dict(Counter(r['status'] for r in results)),
             'calibration':calibration,'per_puzzle_H1':skill,'per_puzzle_H1_safe':safe_skill,
             'per_puzzle_H2':warning_diffs,'cost':cost()}
    write_json(directory/'analysis.json',summary)
    print(dumps({k:v for k,v in summary.items() if k not in ['calibration','per_puzzle_H1','per_puzzle_H2']}))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['init','forecast','actors','analyse'])
    p.add_argument('--round',default='r3_binary');p.add_argument('--offset',type=int,default=0)
    p.add_argument('--n',type=int,default=48);p.add_argument('--phase',default='exploration')
    p.add_argument('--higher-budget',action='store_true')
    p.add_argument('--two-models',action='store_true')
    args=p.parse_args()
    if args.stage=='init':init(args.round,args.offset,args.n,args.phase,args.higher_budget,args.two_models)
    elif args.stage=='forecast':forecasts(args.round)
    elif args.stage=='actors':actors(args.round)
    else:analyse(args.round)
