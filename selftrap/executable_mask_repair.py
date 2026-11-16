"""Explicit construction repair for the two noncompleted program authorings.

Original outcomes remain intact. No transfer feedback is sent to authors.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

from .executable_mask_pilot import TARGET, trust_prompt
from .pilot_json import MODELS, accounting, seal
from .runner import request
from .runtime import BudgetProvider
from .storage import digest, dumps, read_json, write_json
from .style_mask_pilot import style_scores
from .swap_harness import execute, validate_policy


TOOL={'type':'function','name':'submit_policy','description':'Submit the final executable Python edit-selection policy.',
    'parameters':{'type':'object','properties':{'code':{'type':'string'},'method':{'type':'string'}},
                  'required':['code','method'],'additionalProperties':False}}


def run(directory,parent,live):
    directory,parent=Path(directory),Path(parent)
    prior=read_json(parent/'design.json');spec0=prior['spec']
    prompts={}
    for m in MODELS:
        old=read_json(parent/'responses'/m.id/'author.json')
        if old['status']=='completed':raise ValueError('Repair is only for failed authorings')
        prompts[m.id]=(spec0['author_prompt']+'\nCONSTRUCTION REPAIR: the prior response did not submit '
            'an executable final program. Use the submit_policy function now. Provide actual Python source '
            'as its code STRING; do not double-encode a JSON object or add literal backslash-n outside '
            'strings. Exactly define generate(problem). Use a bounded FOR loop, not while. '
            'A simple valid adaptive mutation policy is sufficient; do not spend the whole response '
            'debating algorithms. No transfer examples or scores are available. '
            'Every evaluation must go through score(plan) and count toward the 64-call limit.')
    spec={'protocol':'executable-mask-construction-repair-v1','parent_design_hash':digest(prior),
        'parent_source_hashes':spec0['source_hashes'],'source_hash':digest(Path(__file__).read_text()),
        'models':[asdict(m) for m in MODELS],'prompts':prompts,'tools':[TOOL],
        'tokens':{'gemma-4-31b-it':3000,'glm-5.3-flash':16384},'timeout':180,'max_calls':12,
        'scope':'Two explicit construction repairs; no repetition of a valid failed attack. Same lattice, '
                'same development/transfer tasks, same target, same baselines and same 64-query budget. '
                'If GLM again truncates it is technical noncompletion, not a negative research result.'}
    design=seal(directory,spec)
    if not live:print(f'Sealed native-function repair: {directory/"design.json"}');return
    provider=BudgetProvider(concurrency=2);evaluations={};programs={};probe_rows=[]
    artifact=Path(spec0['baseline'])/'char_tfidf_ovr_logistic.joblib';tasks=spec0['tasks']
    try:
        def author(m):
            settings={'budget_before':design['budget_before'],'tokens':spec['tokens'][m.id],
                'seconds':180,'max_calls':12,'tool_choice':{'type':'function','name':'submit_policy'}}
            row=request(directory/'responses'/m.id,provider,m,prompts[m.id],'construction',settings,[TOOL])
            if row['status']!='valid':return m.id,None,row
            code=row['object']['code']
            try:
                validate_policy(code)
                dev=execute(code,[t for t in tasks if t['split']=='development'],artifact,TARGET)
                if not all(x['ok'] for x in dev):raise ValueError(dumps(dev))
                return m.id,code,row
            except Exception as exc:
                write_json(directory/'invalid'/(m.id+'.json'),{'error':str(exc),'submission':row})
                return m.id,None,row
        with ThreadPoolExecutor(max_workers=2) as pool:
            for f in tqdm(as_completed([pool.submit(author,m) for m in MODELS]),total=2,desc='Native code-submission repair'):
                name,code,row=f.result()
                if code:
                    programs[name]=code
                    evaluations[name]=execute(code,tasks,artifact,TARGET)
                    write_json(directory/'offline'/(name+'.json'),evaluations[name])
        policies=[name for name,rows in evaluations.items() if all(r['ok'] for r in rows)]
        dev_ids={t['id'] for t in tasks if t['split']=='development'}
        selected=max(policies,key=lambda name:sum(r['margin'] for r in evaluations[name] if r['id'] in dev_ids)/2) if policies else None
        successes=[r for r in evaluations.get(selected,[]) if r['id'] not in dev_ids and r['top1']==TARGET and r['original_top1']!=TARGET]
        if len(successes)>=2:
            chosen=sorted(successes,key=lambda r:r['id'])[:2]
            write_json(directory/'selected-probes.json',{'policy':selected,'selection':'first two target-top1 transfer successes; exploratory','items':chosen})
            for item in chosen:
                source=next(t['problem']['original'] for t in tasks if t['id']==item['id'])
                def one_judge(m):
                    result=[]
                    for version,text in (('original',source),('masked',item['text'])):
                        settings={'budget_before':design['budget_before'],'tokens':8192 if m.id.startswith('glm') else 2000,
                                  'seconds':120,'max_calls':12}
                        row=request(directory/'probes'/m.id,provider,m,
                            trust_prompt(item,text,version,spec0['reference_examples']),item['id']+'-'+version,settings)
                        result.append({'item':item['id'],'version':version,'judge':m.id,**row})
                    return result
                with ThreadPoolExecutor(max_workers=2) as pool:
                    for f in as_completed([pool.submit(one_judge,m) for m in MODELS]):probe_rows.extend(f.result())
    finally:provider.close()
    import joblib
    second=joblib.load(Path(spec0['baseline'])/'surface_features_logistic.joblib')
    overview={}
    for name,rows in evaluations.items():
        transfer=[r for r in rows if r['ok'] and r['id'] not in dev_ids]
        for r in transfer:r['unqueried_surface_top1']=style_scores(second,r['text'])['top1_style_label']
        overview[name]={'completed_transfer':len(transfer),'new_target_top1':sum(r['top1']==TARGET and r['original_top1']!=TARGET for r in transfer),
            'target_top1':sum(r['top1']==TARGET for r in transfer),
            'unqueried_surface_target_top1':sum(r['unqueried_surface_top1']==TARGET for r in transfer),
            'mean_margin_gain':sum(r['margin']-r['original_margin'] for r in transfer)/len(transfer) if transfer else None,
            'mean_queries':sum(r['queries'] for r in transfer)/len(transfer) if transfer else None}
    summary={'overview':overview,'selected_by_development':selected,'trust_probes':probe_rows,
        'accounting':accounting(directory),'gate':'NO_TARGETED_TRANSFER_PROMISE' if len(successes)<2 else 'TRANSFER_MASKS_REQUIRE_EPISTEMIC_CONFIRMATION',
        'parent_summary_hash':digest(read_json(parent/'summary.json'))}
    write_json(directory/'summary.json',summary)
    print(dumps({k:v for k,v in summary.items() if k!='trust_probes'}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('runs/executable-mask-repair-1'))
    p.add_argument('--parent',type=Path,default=Path('runs/executable-mask-1'))
    p.add_argument('--live',action='store_true');a=p.parse_args();run(a.output,a.parent,a.live)
