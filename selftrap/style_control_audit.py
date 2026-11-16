"""Audit standard versus commitment-explicit validation of existing style masks.

This is a pipeline feasibility screen, not a replication or refutation of a
published style-bias estimate. The masks were generated in an earlier pilot.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
import random

from tqdm import tqdm

from .pilot_json import Calls, MODELS, accounting, seal
from .storage import digest, dumps, read_json, write_json


ISSUES={
    'p001':{'S1':'Drops often from the beauty generalization and seems from growth remembering arithmetic.',
            'S2':'Changes growth seems to remember into growth adheres and adds eternal to the closing generalization.'},
    'p007':{'S1':'Could fail becomes would; water-pump and aurora-brightness details are omitted.',
            'S2':'Possible satellite, GPS and transformer failures become definite consequences.'},
    'p013':{'S1':'An efficient optical arrangement becomes the only way to see; navigation is omitted.',
            'S2':'Similar challenges and designs become identical and inevitably rewarded; adjustable focus is omitted.'}}


def audit_cases(previous):
    design=read_json(previous/'design.json');result=[]
    for i,item in enumerate(design['spec']['items']):
        if item['prompt_id'] not in ISSUES:continue
        response=read_json(previous/'responses'/(item['id']+'.json'))
        if response['status']!='completed':raise ValueError('Earlier rewrite did not complete')
        candidates=[{'text':item['text'],'gold_admissible':True,'source_condition':'verbatim','issue':None}]
        candidates += [{'text':response['rewrites'][s],'gold_admissible':False,'source_condition':s,
                        'issue':ISSUES[item['prompt_id']][s]} for s in ('S1','S2')]
        random.Random(20261116+i).shuffle(candidates)
        candidates=[{'id':chr(75+j),**c} for j,c in enumerate(candidates)]
        result.append({'id':item['prompt_id'],'source':item['text'],'candidates':candidates})
    if len(result)!=3:raise ValueError('Expected three nonfiction/technical source passages')
    return result


def prompt(case,mode):
    instruction=('Assess whether each candidate preserves the meaning of SOURCE. A meaning-preserving '
        'candidate is admissible as a style-only experimental rewrite. Do not fact-check the source '
        'against outside knowledge. Judge the relationship between the two texts.')
    if mode=='commitments':
        instruction+=(' Check every factual proposition, qualification, negation, possibility versus necessity, '
            'frequency quantifier, scope, and causal claim. Added or dropped propositions and changed '
            'certainty make a candidate inadmissible even if the overall topic and gist match. '
            'Poetic wording alone is not a factual difference; identify the precise changed commitment.')
    return (instruction+'\nSOURCE:\n'+case['source']+'\nCANDIDATES:\n'
        +dumps([{'id':c['id'],'text':c['text']} for c in case['candidates']])
        +'\nReturn only JSON: {"judgments":[{"id":"K","admissible":true,'
        '"reason":"at most 35 words"}]}. Include exactly one judgment per candidate.')


def evaluate(case,obj):
    if not isinstance(obj,dict) or set(obj)!={'judgments'} or not isinstance(obj['judgments'],list):
        raise ValueError('Expected judgments list')
    rows=obj['judgments'];expected={c['id']:c for c in case['candidates']}
    if len(rows)!=3 or {r.get('id') for r in rows}!=expected.keys():raise ValueError('Missing/duplicate candidate IDs')
    result=[]
    for row in rows:
        if set(row)!={'id','admissible','reason'} or type(row['admissible']) is not bool or not isinstance(row['reason'],str):
            raise ValueError('Malformed judgment')
        gold=expected[row['id']]
        result.append({**row,'gold_admissible':gold['gold_admissible'],'source_condition':gold['source_condition'],
                       'reference_issue':gold['issue'],'correct':row['admissible']==gold['gold_admissible']})
    return result


def run(directory,previous,live):
    directory=Path(directory);previous=Path(previous);cases=audit_cases(previous)
    spec={'protocol':'style-only-control-audit-v1','source_hash':digest(Path(__file__).read_text()),
        'models':[asdict(m) for m in MODELS],'cases':cases,'max_calls':12,'credit_cap':1,
        'gold':'Single unblinded assistant reference review with explicit textual issues, not independent human annotation.',
        'scope':'Three already-generated source groups, six masks and three verbatim controls. Judgments within '
                'a call/source are dependent. This does NOT test a published rewriting implementation or estimate a style-bias effect.',
        'gate':'A validator-blind-spot signal requires both models to admit at least two known-altered masks '
               'under the standard screen, reject more under the explicit screen, and accept all verbatim controls. '
               'Even a positive signal requires independent annotation and a published-method replication for novelty.',
        'prompts':{c['id']:{m:prompt(c,m) for m in ('standard','commitments')} for c in cases}}
    design=seal(directory,spec)
    if not live:print(f'Sealed 12-call validator audit: {directory/"design.json"}');return
    calls=Calls(directory,design,max_calls=12)
    def one_model(model,bar):
        for case in cases:
            for mode in ('standard','commitments'):
                row=calls.ask(model,case['id']+'-'+mode,spec['prompts'][case['id']][mode]);bar.update(1)
                if row['status']!='completed':return
    try:
        with tqdm(total=12,desc='Meaning-preservation controls') as bar,ThreadPoolExecutor(max_workers=2) as pool:
            for f in as_completed([pool.submit(one_model,m,bar) for m in MODELS]):f.result()
    finally:calls.close()
    results=[];rates={}
    for model in MODELS:
        rates[model.id]={}
        for mode in ('standard','commitments'):
            judgments=[]
            for case in cases:
                p=directory/'responses'/model.id/(case['id']+'-'+mode+'.json')
                if not p.exists():continue
                r=read_json(p)
                if r['status']=='completed':
                    r['judgments']=evaluate(case,r['object']);judgments.extend(r['judgments'])
                results.append({'model':model.id,'case':case['id'],'mode':mode,**r})
            rates[model.id][mode]={'masks_seen':sum(not r['gold_admissible'] for r in judgments),
                'altered_masks_accepted':sum(not r['gold_admissible'] and r['admissible'] for r in judgments),
                'verbatim_controls_rejected':sum(r['gold_admissible'] and not r['admissible'] for r in judgments),
                'controls_seen':sum(r['gold_admissible'] for r in judgments)}
    signal=all(v['standard']['altered_masks_accepted']>=2 and
               v['commitments']['altered_masks_accepted']<v['standard']['altered_masks_accepted'] and
               all(x['controls_seen']==3 and x['verbatim_controls_rejected']==0 for x in v.values())
               for v in rates.values())
    summary={'rates':rates,'results':results,'accounting':accounting(directory),
        'gate':'VALIDATOR_BLIND_SPOT_REQUIRES_INDEPENDENT_VALIDATION' if signal else 'NO_CROSS_MODEL_VALIDATOR_BLIND_SPOT',
        'novelty':'No published bias estimate was reproduced or overturned; broad uncertainty drift is known.'}
    write_json(directory/'summary.json',summary)
    print(dumps({k:v for k,v in summary.items() if k!='results'}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('runs/style-control-audit-1'))
    p.add_argument('--previous',type=Path,default=Path('runs/kaggle/name-that-llm/mask-pilot-1'))
    p.add_argument('--live',action='store_true');a=p.parse_args();run(a.output,a.previous,a.live)
