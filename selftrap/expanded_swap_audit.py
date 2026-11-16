"""Offline feasibility and frozen-policy audit with context-reviewed synonyms.

Pairs are researcher proposals, not general semantic-equivalence certificates.
Existing author programs are frozen; no outcome feedback is sent to models.
"""

import argparse
import math
from pathlib import Path
import re

from tqdm import tqdm

from .pilot_json import accounting, seal
from .storage import digest, dumps, read_json, write_json
from .swap_harness import GREEDY, PROTECTED, RANDOM, execute, render


PAIRS={
 'p004':{'rich enough':['sufficiently rich'],'built-in limits':['inherent limits'],
         'falsehoods':['false statements'],'in effect':['in essence']},
 'p012':{'enormous':['immense'],'sped up':['accelerated'],'neared':['approached'],
         'announce':['signal'],'simply':['just'],'squeezing':['compressing'],'ordinary':['usual'],'final':['last']},
 'p017':{'merely':['simply'],'according to':['based on'],'yet':['but'],'persistent':['enduring'],
         'arranged with exquisite care':['meticulously arranged'],'emulate':['simulate'],'thus':['therefore'],'uncanny':['eerie']},
 'p020':{'infant cosmos':['early universe'],'space itself ballooning':['space itself expanding'],
         'vast':['immense'],'eerily':['strangely']},
 'p022':{'rewrite':['alter'],'loosen':['relax'],'tune':['modulate'],'primed':['prepared']},
 'p023':{'colossal':['enormous'],'newborn':['young'],'scattering':['dispersing'],'flinging':['throwing'],'veered':['turned']},
 'p027':{'immense':['vast'],'upon':['on'],'voyage':['journey'],'vanish':['disappear'],'keep sliding':['continue sliding']},
 'p028':{'wreck':['shipwreck'],'at first':['initially'],'remnant':['relic'],'fractured':['broken'],
         'disclose':['reveal'],'astonishing':['remarkable'],'hand-cut teeth':['teeth cut by hand'],
         'once turned':['formerly turned'],'portable':['transportable'],'vanished':['disappeared'],'artifact':['artefact']},
}


def expand(task):
    old=task['problem'];text=old['original'];spans=[]
    # Recover positions of all old slots, preserving their exact options.
    cursor=0
    for i,site in enumerate(old['sites']):
        cursor+=len(old['pieces'][i]);spans.append((cursor,cursor+len(site['original']),site['options'],site['kind']))
        cursor+=len(site['original'])
    for source,alts in PAIRS[task['prompt_id']].items():
        for m in re.finditer(r'\b'+re.escape(source)+r'\b',text):
            if any(m.start()<b and m.end()>a for a,b,_,_ in spans):continue
            options=[source]+alts
            if any(PROTECTED.findall(o)!=PROTECTED.findall(source) for o in options):
                raise ValueError('A proposed synonym would alter protected tokens')
            spans.append((m.start(),m.end(),options,'context_reviewed_synonym'))
    spans.sort();pieces=[];sites=[];last=0
    for a,b,options,kind in spans:
        pieces.append(text[last:a]);last=b
        sites.append({'id':len(sites),'original':text[a:b],'options':options,'kind':kind})
    pieces.append(text[last:])
    result={**task,'problem':{'original':text,'pieces':pieces,'sites':sites}}
    assert render(result['problem'],[0]*len(sites))==text
    return result


EXACT='''def generate(problem):
    plans = [[]]
    for site in problem['sites']:
        plans = [prefix + [j] for prefix in plans for j in range(len(site['options']))]
    best = plans[0]
    value = score(best)
    for plan in plans[1:]:
        v = score(plan)
        if v > value:
            best, value = plan, v
    return best
'''


def run(directory,parent,repair):
    directory,parent,repair=Path(directory),Path(parent),Path(repair)
    original=read_json(parent/'design.json')['spec']
    tasks=[expand(t) for t in original['tasks']]
    programs={}
    for name in ('gemma-4-31b-it','glm-5.3-flash'):
        row=read_json(repair/'responses'/name/'construction.json')
        if row['status']=='valid':programs[name]=row['object']['code']
    spec={'protocol':'expanded-contextual-synonym-audit-v1','source_hash':digest(Path(__file__).read_text()),
        'harness_hash':digest(Path(__file__).with_name('swap_harness.py').read_text()),
        'pairs':PAIRS,'tasks':tasks,'program_hashes':{k:digest(v) for k,v in programs.items()},
        'scope':'Explicit post-narrow-bank feasibility follow-up. Bank curated after inspecting texts but '
                'before its scores. Authors never see expanded bank or test scores. The SAME frozen programs '
                'run on it. Protected-token invariance is mechanical; synonym equivalence requires external review. '
                'This is not an unbiased population attack-rate estimate.',
        'narrow_space_sizes':{t['id']:math.prod(len(s['options']) for s in t['problem']['sites']) for t in original['tasks']}}
    design=seal(directory,spec)
    artifact=Path(original['baseline'])/'char_tfidf_ovr_logistic.joblib';target=original['target']
    narrow=[{**t,'budget':math.prod(len(s['options']) for s in t['problem']['sites'])} for t in original['tasks']]
    exact=execute(EXACT,narrow,artifact,target);write_json(directory/'narrow-exhaustive.json',exact)
    outputs={}
    for name,code in tqdm({'random':RANDOM,'greedy':GREEDY,**programs}.items(),desc='Expanded bank: frozen policies'):
        rows=execute(code,tasks,artifact,target);outputs[name]=rows
        write_json(directory/'offline'/(name+'.json'),rows)
    transfer_ids={t['id'] for t in tasks if t['split']=='transfer'}
    overview={}
    for name,rows in outputs.items():
        transfer=[r for r in rows if r['id'] in transfer_ids and r['ok']]
        overview[name]={'completed_transfer':len(transfer),
            'new_target_top1':sum(r['top1']==target and r['original_top1']!=target for r in transfer),
            'mean_margin_gain':sum(r['margin']-r['original_margin'] for r in transfer)/len(transfer) if transfer else None,
            'mean_queries':sum(r['queries'] for r in transfer)/len(transfer) if transfer else None}
    summary={'overview':overview,'narrow_reachable_target_transfer':sum(r['top1']==target and r['id'] in transfer_ids for r in exact),
        'expanded_spaces':{t['prompt_id']:math.prod(len(s['options']) for s in t['problem']['sites']) for t in tasks},
        'narrow_exact_queries':sum(r.get('queries',0) for r in exact),
        'repair_accounting':accounting(repair),'scope':spec['scope']}
    write_json(directory/'summary.json',summary);print(dumps(summary))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('runs/expanded-swap-audit-1'))
    p.add_argument('--parent',type=Path,default=Path('runs/executable-mask-1'))
    p.add_argument('--repair',type=Path,default=Path('runs/executable-mask-repair-1'))
    a=p.parse_args();run(a.output,a.parent,a.repair)
