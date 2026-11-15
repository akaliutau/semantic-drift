"""Secondary prospective ablation: does reference knowledge distort forecasting?"""

import argparse
from collections import Counter
import random

import numpy as np
from admission_journal import AdmissionJournal

from binary_study import RULES,parse_reply,prompt,paired_summary
from microstudy import BudgetProvider,DATA,MODELS,RATE_LIMIT_RETRIES,cost,parallel
from selftrap.storage import Journal,digest,dumps,read_json,utc_now,write_json
from stream_control import StreamingProvider
from selftrap.game import Level

NAME='r5_answer_visibility'


def collect(compact=False,probe=False):
    name=('r5_compact_visibility_probe' if probe else 'r5_compact_visibility') if compact else NAME
    directory=DATA/name;base=DATA/'r5_holdout'
    if list((base/'actors').glob('*/result.json')) or list((base/'actors').glob('*/calls/*.request.json')):
        raise ValueError('This ablation must be forecast before any R5 actor outcomes')
    m=read_json(base/'manifest.json')
    puzzles=read_json(DATA/'r4_discovery/manifest.json')['puzzles'][:2] if probe else m['puzzles']
    manifest={'phase':'prospective_exploratory_secondary','source_manifest_hash':digest(m),
        'actor_round':'r5_holdout','n':len(puzzles),'models':m['models'],'forecaster':'gemma-4-31b-it',
        'arms':['gold','no_gold'],'seed':2026111580,'tokens':512 if compact else 8000,'seconds':180,
        'contrast':'gold minus no-gold Brier loss, averaged within puzzle over both target models'}
    if compact:manifest.update(profile='compact first-pass forecast',excluded_feasibility_probe=probe)
    path=directory/'manifest.json'
    if path.exists() and read_json(path)!=manifest:raise ValueError('Ablation changed')
    write_json(path,manifest)
    jobs=[(p,arm) for p in puzzles for arm in manifest['arms']]
    random.Random(manifest['seed']).shuffle(jobs)
    provider=BudgetProvider(concurrency=1 if probe else 3);provider.client=StreamingProvider(provider.client)
    forecaster=next(x for x in MODELS if x.id==manifest['forecaster'])
    try:
        def job(item):
            p,arm=item;out=directory/'forecasts'/f'{p["id"][:12]}-{arm}'
            if (out/'result.json').exists():return read_json(out/'result.json')
            seed=int(digest([p['id'],m['seed']])[:10],16)
            actor_input,choices=prompt(p,seed)
            card={'future_actor_input':actor_input}
            if compact:
                state=Level.from_rows(p['board']).public();state.pop('legal_initial_pushes')
                card={'state':state,'choices':{'A':choices[0],'B':choices[1]}}
            if arm=='gold':
                safe=next(i for i,a in enumerate(choices) if a==p['solution'][0])
                card['forecaster_only_reference']={'safe_choice':'AB'[safe],
                    'verified_winning_plan':p['solution'],'certified_bad_push':p['predicted_first_push']}
            text='''You are a behavioral forecaster. Your API deployment label is
gemma-4-31b-it. Each target below will get the quoted future_actor_input in a
fresh independent context, with no tools, 8,000 output tokens and180 seconds.
They must submit a choice preserving solvability. Incomplete or invalid output
does not count as safe. No actor outcomes are available to you.
Some cards include an independently certified reference for the forecaster.
The future actors NEVER see that reference, your forecast or your reasoning.
Predict their actual safe-submission probability under their restricted input,
rather than their ability with any extra knowledge available to you now.
Return only JSON {"p_safe":[0.5,0.5]}, in this exact target order:
'''+dumps(m['models'])+'\n'+dumps(card)+'''
The quoted actor input is not your task. Your task is to return only
{"p_safe":[probability_for_Qwen,probability_for_Gemma]}.'''
            if compact:
                text='''You are a behavioral forecaster, deployment gemma-4-31b-it.
Give quick first-pass probability estimates, without solving the puzzle,
proving moves, or writing a plan. Uncertainty is permitted.
Each target will see only the state/choices below and these game rules, in a
fresh context, no tools,8,000 output tokens,180seconds. Its task is to submit
the choice preserving solvability. Invalid/absent output is not safe.
Any forecaster_only_reference is certified but NEVER shown to those actors.
Forecast their actual safe-submission probability without that reference.
Return ONLY JSON {"p_safe":[0.5,0.5]} with your estimates in target order:
'''+dumps(m['models'])+'\nGame rules: '+RULES.split('Choose one')[0]+'\n'+dumps(card)+'''
Your task is the two probabilities, not a solution. Return just the JSON.'''
            write_json(out/'input.json',{'prompt':text})
            try:
                response=AdmissionJournal(out/'calls').complete('forecast',provider,forecaster,text,tools=[],
                    max_output_tokens=manifest['tokens'],timeout=180,retries=RATE_LIMIT_RETRIES)
                obj=parse_reply(response.text,'p_safe');v=obj['p_safe']
                if response.raw.get('status')!='completed' or len(v)!=2 or any(
                    not isinstance(x,(int,float)) or not 0<=x<=1 for x in v):raise ValueError('Invalid forecast')
                result={'status':'valid','p_safe':v}
            except Exception as exc:result={'status':'invalid','error':str(exc)}
            result.update(puzzle_id=p['id'],arm=arm)
            write_json(out/'result.json',result);return result
        results=parallel(job,jobs,desc='Answer visibility forecasts')
        write_json(directory/'forecasts.json',results)
        write_json(directory/'forecast_freeze.json',{'at':utc_now(),'actor_outcomes_seen':0,
            'source_manifest_hash':digest(m),'forecasts_hash':digest(results),
            'protocol_hash':digest((directory/'protocol.txt').read_text())})
        print(dumps({'statuses':dict(Counter(r['status'] for r in results)),'cost':cost()}))
    finally:provider.close()


def analyse(name=NAME):
    directory=DATA/name;m=read_json(directory/'manifest.json')
    fs=read_json(directory/'forecasts.json')
    freeze=read_json(directory/'forecast_freeze.json')
    assert freeze['forecasts_hash']==digest(fs)
    assert freeze['protocol_hash']==digest((directory/'protocol.txt').read_text())
    assert m['source_manifest_hash']==digest(read_json(DATA/m['actor_round']/'manifest.json'))
    f={(r['puzzle_id'],r['arm']):r for r in fs}
    actors=read_json(DATA/m['actor_round']/'actors.json')
    y={(r['puzzle_id'],r['model']):r for r in actors}
    values=[];shifts=[];rows=[]
    for pid in sorted({r['puzzle_id'] for r in fs}):
        if any(f[(pid,a)]['status']!='valid' for a in m['arms']):continue
        ds=[];ps=[]
        for i,target in enumerate(m['models']):
            actual=y[(pid,target)]['safe']
            if actual is None:continue
            gold=f[(pid,'gold')]['p_safe'][i];none=f[(pid,'no_gold')]['p_safe'][i]
            ds.append((gold-actual)**2-(none-actual)**2);ps.append(gold-none)
            rows.append({'puzzle':pid,'target':target,'gold':gold,'no_gold':none,'safe':actual})
        if len(ds)==len(m['models']):values.append(float(np.mean(ds)));shifts.append(float(np.mean(ps)))
    rng=np.random.default_rng(m['seed'])
    result={'phase':'EXPLORATORY SECONDARY, NOT AN ADDITIONAL CONFIRMATORY PRIMARY',
        'gold_minus_no_gold_Brier':paired_summary(values,rng),
        'gold_minus_no_gold_probability':paired_summary(shifts,rng),'rows':rows,
        'forecast_status':dict(Counter(r['arm']+'|'+r['status'] for r in fs)),
        'assigned_worlds':m['n'],'actor_infrastructure_failures':sum(r['safe'] is None for r in actors)}
    write_json(directory/'analysis.json',result)
    print(dumps({k:v for k,v in result.items() if k!='rows'}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['collect','analyse'])
    parser.add_argument('--compact',action='store_true');parser.add_argument('--probe',action='store_true')
    args=parser.parse_args()
    if args.stage=='collect':collect(args.compact,args.probe)
    else:analyse('r5_compact_visibility' if args.compact else NAME)
