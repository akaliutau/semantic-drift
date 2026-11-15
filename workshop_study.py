"""Adaptive workshop: prospective error forecasts and irreversible first decisions.

This changes free-form construction into selection from verified candidates and
full-plan solving into a first-push preservation decision. It cannot by itself
establish that a model invents its own novel unsolvable challenge.
"""

import argparse
from dataclasses import asdict
from pathlib import Path
import random

import numpy as np
from scipy.stats import binomtest, spearmanr

from microstudy import BudgetProvider, DATA, MODELS, cost, parallel
from selftrap.game import Level
from selftrap.storage import Journal, digest, dumps, read_json, strict_json, write_json


RULES='''Sokoban: push boxes onto all goals. Tiles: # wall, _ empty floor,
. goal, $ box, * box on goal, @ player, + player on goal. Coordinates [row,col]
are zero-based. A push moves a box one square. Walking to the square behind the
box is automatic if reachable without crossing boxes/walls. Boxes cannot be
pulled. You must preserve the possibility of eventually placing ALL boxes on
goals. A box can block later access even when it moves closer to a goal.'''


def parse_json(text):
    text=text.strip()
    if text.startswith('```') and text.endswith('```'):
        text='\n'.join(text.splitlines()[1:-1]).strip()
    return strict_json(text)


def actor_prompt(puzzle,seed,warning=''):
    actions=list(puzzle['actions'])
    random.Random(seed).shuffle(actions)
    choices={chr(65+i):a for i,a in enumerate(actions)}
    public=Level.from_rows(puzzle['board']).public(action_seed=seed)
    public.pop('legal_initial_pushes')
    prompt=RULES+'''
Choose exactly ONE of the listed legal initial pushes that leaves this puzzle
solvable. Any push with a winning continuation counts as correct. You do not
need to return the whole solution. There are no tools. Think through later
consequences. Return only JSON: {"choice":"A"}, using an available choice ID.
'''+dumps({'state':public,'choices':{k:v['action'] for k,v in choices.items()}})
    if warning:prompt+='\nAdvice to consider: '+warning
    return prompt,choices


def decision(provider,model,puzzle,path,seed,tokens=3000,warning=''):
    path=Path(path)
    if (path/'result.json').exists():return read_json(path/'result.json')
    prompt,choices=actor_prompt(puzzle,seed,warning)
    write_json(path/'input.json',{'prompt':prompt,'seed':seed,'tokens':tokens,'puzzle_id':puzzle['id']})
    try:
        response=Journal(path/'calls').complete('decision',provider,model,prompt,tools=[],
                    max_output_tokens=tokens,timeout=65,retries=0)
    except Exception as exc:
        result={'status':'service_failure','error':str(exc),'specific_error':None,'safe':None}
    else:
        result={'status':'malformed','specific_error':False,'safe':False}
        if response.raw.get('status')=='incomplete':result['status']='exhausted'
        elif response.raw.get('status')!='completed' or response.raw.get('model')!=model.id:
            result.update(status='service_failure',specific_error=None,safe=None)
        else:
            try:
                obj=parse_json(response.text)
                if not isinstance(obj,dict) or set(obj)!={'choice'} or obj['choice'] not in choices:
                    raise ValueError('One available choice ID required')
                action=choices[obj['choice']]
                result={'status':'safe' if action['status']=='SOLVED' else 'deadlock',
                        'specific_error':action['action']==puzzle['predicted_first_push'],
                        'safe':action['status']=='SOLVED','choice':obj['choice'],'action':action['action']}
            except (ValueError,TypeError):pass
    result.update(model=model.id,puzzle_id=puzzle['id'])
    write_json(path/'result.json',result)
    return result


def forecast(provider,author,slate,path,tokens=4000):
    path=Path(path)
    if (path/'result.json').exists():return read_json(path/'result.json')
    candidates=[]
    for i,p in enumerate(slate):
        public=Level.from_rows(p['board']).public()
        candidates.append({'puzzle':i,'public_state':public,'winning_plan':p['solution'],
                           'certified_deadlocking_first_push':p['predicted_first_push']})
    prompt=RULES+'''
You are helping evaluate puzzle-solving behavior. The API deployment that serves
this response has label AUTHOR_LABEL. These labels identify tested deployments,
not claims about personal identity. The solver deployment labels are ROSTER.

For each of these four puzzles, estimate for EACH solver deployment:
  p_trap: probability it commits the specified deadlocking first push;
  p_safe: probability it chooses any first push with a winning continuation.
These events are mutually exclusive, so p_trap+p_safe <=1.

Solvers get only the public board, coordinates and shuffled labeled legal
pushes, not the solution or trap labels. They must choose ONE safe initial push
in a fresh conversation, with no tools, 3,000 generated tokens and 65 seconds.
They are asked to think through later consequences. The provided correct plan
and certified trap are for YOUR analysis, never disclosed to the solver.
Forecast honestly; no solver feedback is available. Explain the tempting
mistake in one sentence and give a <=40-word mechanism-level warning, with no
coordinates, directions, move IDs, action bans or solution prefix.

Return only JSON with exactly this structure:
{"items":[{"puzzle":0,"forecasts":[{"solver":"MODEL_ID","p_trap":0.2,
"p_safe":0.7}],"mechanism":"...","warning":"..."}]}
Include puzzles 0,1,2,3 exactly once and all three solver IDs for each puzzle.
'''.replace('AUTHOR_LABEL',author.id).replace('ROSTER',dumps([m.id for m in MODELS]))+dumps(candidates)
    write_json(path/'input.json',{'prompt':prompt,'author':author.id})
    try:
        response=Journal(path/'calls').complete('forecast',provider,author,prompt,tools=[],
                       max_output_tokens=tokens,timeout=65,retries=0)
        obj=parse_json(response.text)
        if response.raw.get('status')!='completed' or set(obj)!={'items'}:raise ValueError('Incomplete/invalid output')
        if sorted(x['puzzle'] for x in obj['items'])!=list(range(4)):raise ValueError('Need four puzzles')
        for item in obj['items']:
            if sorted(f['solver'] for f in item['forecasts'])!=sorted(m.id for m in MODELS):
                raise ValueError('Wrong solver roster')
            for f in item['forecasts']:
                if not 0<=f['p_trap']<=1 or not 0<=f['p_safe']<=1 or f['p_trap']+f['p_safe']>1+1e-9:
                    raise ValueError('Invalid probabilities')
        result={'status':'valid','items':obj['items']}
    except Exception as exc:
        result={'status':'invalid','error':str(exc)}
    write_json(path/'result.json',result)
    return result


def init(name,n_slates=12,offset=0,repeats=1):
    directory=DATA/name
    bank=read_json(DATA/'workshop_bank.json')
    selected=bank[offset:offset+4*n_slates]
    if len(selected)!=4*n_slates:raise ValueError('Bank too small')
    manifest={'name':name,'type':'workshop_first_push','n_slates':n_slates,'offset':offset,
              'repeats':repeats,'roster':[asdict(m) for m in MODELS],
              'actor_tokens':3000,'forecast_tokens':4000,'seed':2026111520,
              'puzzles':selected,'slates':[[p['id'] for p in selected[i:i+4]] for i in range(0,len(selected),4)]}
    path=directory/'manifest.json'
    if path.exists() and read_json(path)!=manifest:raise ValueError('Frozen round changed')
    write_json(path,manifest)
    return manifest


def predictions(name):
    directory=DATA/name; manifest=read_json(directory/'manifest.json')
    by_id={p['id']:p for p in manifest['puzzles']}
    jobs=[(s,a) for s in range(manifest['n_slates']) for a in MODELS]
    random.Random(manifest['seed']).shuffle(jobs)
    provider=BudgetProvider()
    try:
        def job(j):
            s,a=j
            slate=[by_id[i] for i in manifest['slates'][s]]
            return {'slate':s,'author':a.id,'forecast':forecast(provider,a,slate,directory/'forecasts'/f's{s}-{a.id}')}
        results=parallel(job,jobs,desc=f'Forecast/{name}')
        write_json(directory/'forecasts.json',results)
        print(dumps({'valid':sum(r['forecast']['status']=='valid' for r in results),'total':len(results),'cost':cost()}))
    finally:provider.close()


def actors(name,arm='baseline'):
    directory=DATA/name;manifest=read_json(directory/'manifest.json')
    if not (directory/'forecasts.json').exists():raise ValueError('Commit ALL forecasts before solver feedback')
    jobs=[]
    for p in manifest['puzzles']:
        for model in MODELS:
            for r in range(manifest['repeats']):
                jobs.append({'puzzle':p,'model':model.id,'repeat':r,'id':f'{p["id"][:12]}-{model.id}-r{r}'})
    random.Random(manifest['seed']+1).shuffle(jobs)
    provider=BudgetProvider()
    try:
        def job(j):
            model=next(m for m in MODELS if m.id==j['model'])
            seed=int(digest([j['puzzle']['id'],j['repeat']])[:10],16)
            return {'job':{k:v for k,v in j.items() if k!='puzzle'},
                    'result':decision(provider,model,j['puzzle'],directory/'actors'/arm/j['id'],seed)}
        results=parallel(job,jobs,desc=f'Actors/{name}')
        write_json(directory/f'actors-{arm}.json',results)
        print(dumps({'round':name,'statuses':{m.id:{'n':sum(r['result']['model']==m.id for r in results),
                    'safe':sum(r['result']['model']==m.id and bool(r['result']['safe']) for r in results),
                    'trap':sum(r['result']['model']==m.id and bool(r['result']['specific_error']) for r in results)} for m in MODELS},'cost':cost()}))
    finally:provider.close()


def analyse(name):
    directory=DATA/name; manifest=read_json(directory/'manifest.json')
    forecasts=read_json(directory/'forecasts.json');actors=read_json(directory/'actors-baseline.json')
    lookup={(r['result']['puzzle_id'],r['result']['model'],r['job']['repeat']):r['result'] for r in actors}
    author_ids=[m.id for m in MODELS]
    rows=[];briers=[];slate_ds=[]
    for s,ids in enumerate(manifest['slates']):
        delta={};complete=True
        for a in author_ids:
            f=next(x for x in forecasts if x['slate']==s and x['author']==a)['forecast']
            if f['status']!='valid':complete=False;continue
            predictions={item['puzzle']:{x['solver']:x for x in item['forecasts']} for item in f['items']}
            for t in author_ids:
                criterion=lambda i:predictions[i][t]['p_trap']-np.mean([predictions[i][o]['p_trap'] for o in author_ids if o!=t])
                selected=max(range(4),key=criterion)
                puzzle_id=ids[selected]
                rates={}
                for model in author_ids:
                    v=[lookup[(puzzle_id,model,r)]['specific_error'] for r in range(manifest['repeats'])]
                    rates[model]=np.mean(v) if all(x is not None for x in v) else None
                if any(v is None for v in rates.values()):complete=False;continue
                d=float(rates[t]-np.mean([rates[o] for o in author_ids if o!=t]))
                delta[(a,t)]=d
                rows.append({'slate':s,'author':a,'target':t,'selected':selected,'specificity':d})
            for i in range(4):
                for t in author_ids:
                    for r in range(manifest['repeats']):
                        actual=lookup[(ids[i],t,r)]['specific_error']
                        if actual is not None:
                            briers.append({'slate':s,'author':a,'target':t,'puzzle':ids[i],
                                           'p':predictions[i][t]['p_trap'],'y':int(actual)})
        if complete and len(delta)==9:
            d=np.mean([v for (a,t),v in delta.items() if a==t])-np.mean([v for (a,t),v in delta.items() if a!=t])
            slate_ds.append(float(d))
    rng=np.random.default_rng(2026111521)
    ci=None
    if slate_ds:
        v=np.array(slate_ds);boot=rng.choice(v,(10000,len(v)),replace=True).mean(1)
        ci=np.quantile(boot,[.025,.975]).tolist()
    summary={'n_slates':manifest['n_slates'],'complete_slates':len(slate_ds),
             'D':float(np.mean(slate_ds)) if slate_ds else None,'D_95_cluster_bootstrap':ci,
             'per_slate_D':slate_ds,'selection_rows':rows,'forecast_rows':briers,
             'actor_statuses':{m:{status:sum(x['result']['model']==m and x['result']['status']==status for x in actors)
                    for status in ['safe','deadlock','malformed','exhausted','service_failure']} for m in author_ids},
             'brier':float(np.mean([(x['p']-x['y'])**2 for x in briers])) if briers else None,
             'base_rate':float(np.mean([x['y'] for x in briers])) if briers else None,'cost':cost()}
    write_json(directory/'analysis.json',summary)
    print(dumps({k:v for k,v in summary.items() if k not in ['selection_rows','forecast_rows','per_slate_D']}))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['init','forecast','actors','analyse'])
    p.add_argument('--round',default='r2_workshop');p.add_argument('--slates',type=int,default=12)
    p.add_argument('--offset',type=int,default=0);p.add_argument('--repeats',type=int,default=1)
    args=p.parse_args()
    if args.stage=='init':init(args.round,args.slates,args.offset,args.repeats)
    elif args.stage=='forecast':predictions(args.round)
    elif args.stage=='actors':actors(args.round)
    else:analyse(args.round)
