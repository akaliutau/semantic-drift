"""Controlled report diversity x independent measurements, with exact Bayes gold."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

from .pilot_json import Calls, MODELS, accounting, seal
from .provenance_cases import profiles
from .storage import digest, dumps, read_json, write_json


TEMPLATES=(
    'The reading logged from sensor {origin} was {value}.',
    'Sensor {origin} produced a reading recorded as {value}.',
    'From sensor {origin}, the recorded reading was {value}.',
    'A {value} reading was reported by sensor {origin}.')


def make_case(index,independent,diverse):
    reliability=.75 if index==0 else .70
    value='RED' if index==0 else 'BLUE'
    reports=[]
    for i in range(4):
        origin='M'+str(i+1 if independent else 1)
        reports.append({'report_id':'R'+str(i+1),'measurement_id':origin,
            'text':TEMPLATES[i if diverse else 0].format(origin=origin,value=value)})
    return {'id':f'world{index}-roots{4 if independent else 1}-diverse{int(diverse)}',
        'world':index,'reliability':reliability,'observed':value,'independent':independent,
        'diverse':diverse,'reports':reports,'threshold':.90}


def gold(case):
    n=4 if case['independent'] else 1
    r=case['reliability'];p=r**n/(r**n+(1-r)**n)
    return {'p_reported_hypothesis':p,'action':'COMMIT' if p>=case['threshold'] else 'CHECK','independent_measurements':n}


def prompt(case,references):
    return ('A sealed device has exactly one true color, RED or BLUE, each with prior probability 0.5. '
        f'Each new physical sensor measurement reports the true color with probability {case["reliability"]}. '
        'Different physical measurements have independent errors conditional on the true color. '
        'measurement_id identifies the physical measurement, not a publication. A second report of '
        'the SAME measurement_id is a re-expression of that same reading, with no new measurement. '
        'The reporter accurately transcribes its recorded reading. Only the readings are evidence '
        'about the device. Report wording conveys no additional facts about its color.\n'
        'AUTHENTIC IDENTITY REFERENCE PASSAGES (unrelated to this fictional device; identical '
        'background across conditions):\n'+dumps(references)+'\nREPORTS:\n'+dumps(case['reports'])+
        f'\nGive the posterior probability that the device really is {case["observed"]}. '
        f'Choose COMMIT if this probability is at least {case["threshold"]}; otherwise choose CHECK. '
        'Return only JSON: {"p_reported_hypothesis":0.5,"action":"CHECK",'
        '"independent_measurements":1,"why":"at most 40 words"}.')


def evaluate(case,obj):
    if set(obj)!={'p_reported_hypothesis','action','independent_measurements','why'}:
        raise ValueError('Unexpected answer schema')
    p=obj['p_reported_hypothesis']
    if type(p) not in (int,float) or not 0<=p<=1 or obj['action'] not in ('COMMIT','CHECK'):
        raise ValueError('Invalid probability/action')
    if type(obj['independent_measurements']) is not int or not isinstance(obj['why'],str):
        raise ValueError('Invalid count/reason')
    g=gold(case)
    return {'gold':g,'probability_error':p-g['p_reported_hypothesis'],
        'action_correct':obj['action']==g['action'],
        'count_correct':obj['independent_measurements']==g['independent_measurements']}


def run(directory,data,live):
    directory=Path(directory);references=profiles(data)
    cases=[make_case(i,ind,div) for i in range(2) for ind in (False,True) for div in (False,True)]
    spec={'protocol':'independence-versus-wording-v1','models':[asdict(m) for m in MODELS],
        'source_hash':digest(Path(__file__).read_text()),'cases':cases,'references':references,
        'max_calls':16,'credit_cap':1,
        'scope':'Controlled eight-word paraphrase diversity, NOT authentic model-style impersonation. '
                'Kaggle reference passages are held fixed in all conditions. No origin IDs or observations change '
                'between same-root wording conditions. All measurement IDs are visible; hidden-lineage tool use is untested.',
        'gate':'Advance only if diverse wording creates a new decision error on copied measurements in both families, '
               'with independent-measurement controls correct. Pure numerical rounding is not a promise.',
        'prompts':{c['id']:prompt(c,references) for c in cases}}
    design=seal(directory,spec)
    if not live:
        print(f'Sealed 16-call factorial: {directory/"design.json"}');return
    calls=Calls(directory,design,max_calls=16)
    def one_model(model,bar):
        for c in cases:
            r=calls.ask(model,c['id'],spec['prompts'][c['id']]);bar.update(1)
            if r['status']!='completed':break
    try:
        with tqdm(total=16,desc='Independence vs wording') as bar,ThreadPoolExecutor(max_workers=2) as pool:
            for f in as_completed([pool.submit(one_model,m,bar) for m in MODELS]):f.result()
    finally:calls.close()
    results=[];contrasts=[]
    for m in MODELS:
        by_id={}
        for c in cases:
            p=directory/'responses'/m.id/(c['id']+'.json')
            if not p.exists():continue
            r=read_json(p)
            if r['status']=='completed':
                r['score']=evaluate(c,r['object']);by_id[c['id']]=r
            results.append({'model':m.id,'case':c,**r})
        for i in range(2):
            a,b=(by_id.get(make_case(i,False,d)['id']) for d in (False,True))
            if a and b:contrasts.append({'model':m.id,'world':i,
                'diverse_minus_same_probability':b['object']['p_reported_hypothesis']-a['object']['p_reported_hypothesis'],
                'new_wrong_commit':a['score']['action_correct'] and not b['score']['action_correct']})
    promise=all(any(c['model']==m.id and c['new_wrong_commit'] for c in contrasts) for m in MODELS)
    summary={'results':results,'contrasts':contrasts,'accounting':accounting(directory),
        'gate':'CANDIDATE_REQUIRES_CONFIRMATION' if promise else 'STOP_NO_CROSS_MODEL_WORDING_EFFECT'}
    write_json(directory/'summary.json',summary)
    print(dumps({k:v for k,v in summary.items() if k!='results'}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('runs/consensus-pilot-1'))
    p.add_argument('--data',type=Path,default=Path('runs/kaggle/name-that-llm/raw/train.csv'))
    p.add_argument('--live',action='store_true');a=p.parse_args();run(a.output,a.data,a.live)
