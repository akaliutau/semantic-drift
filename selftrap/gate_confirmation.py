"""Frozen confirmation of bare versus camouflaged semantic drift.

Run authors first, review protected assertions without judge outcomes, then
freeze and run judgments. One original 190-credit ledger; no outcome retries.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import csv
import random
import threading
from time import perf_counter

from tqdm import tqdm

from .camouflage_pilot import AUTHOR_TOOL, author_prompt, protected_sentence
from .gate_audit import evaluate
from .natural_gate_audit import build_item, prompt, word_diff, TOOLS
from .pilot_json import Calls, MODELS, accounting, seal
from .runtime import BudgetProvider
from .storage import digest,dumps,read_json,utc_now,write_json
from .study_stats import holm,paired_binary,wilson


MODES=('bare','masked','checklist','diff')


def prepare(directory,data,manifest):
    records={r['id']:r for r in csv.DictReader(Path(data).open())}
    panel=read_json(manifest);rng=random.Random(2026111605)
    items=[{**build_item(records[r['id']],r['old'],r['new'],rng),'family':r['family']}
           for r in panel['items']]
    if len(items)!=40 or len({i['cluster'] for i in items})!=40:
        raise ValueError('Exactly forty distinct prompt clusters required')
    pilot_exclude={'p001','p007','p013','p015','p009','p014','p018','p022','p026','p029','p098'}
    if any(i['cluster'] in pilot_exclude for i in items):raise ValueError('Pilot topic reused')
    spec=dict(protocol='style-camouflage-confirmation-v1',phase='confirmation',items=items,
        source_hash=digest(Path(__file__).read_text()),
        renderer_hash=digest(Path(__file__).with_name('camouflage_pilot.py').read_text()),
        prompt_hash=digest(Path(__file__).with_name('natural_gate_audit.py').read_text()),
        statistics_hash=digest(Path(__file__).with_name('study_stats.py').read_text()),
        manifest_hash=digest(Path(manifest).read_text()),data_hash=digest(Path(data).read_text()),
        models=[asdict(m) for m in MODELS],author=asdict(MODELS[0]),author_tool=AUTHOR_TOOL,
        author_prompts={i['id']:author_prompt(i) for i in items},judge_tool=TOOLS,
        modes=MODES,max_calls=360,credit_cap=2,workers=3,per_model_concurrency=2,
        author_tokens=2000,judge_tokens={'gemma-4-31b-it':2000,'glm-5.3-flash':8192},
        primary=['masked_minus_bare_false_accept','masked_minus_diff_false_accept'],
        tests='Four two-sided exact paired McNemar tests: two contrasts x two judges. Holm across four. '
              'Bootstrap 95% paired source-cluster risk differences. Judge families reported separately.',
        controls='Heading-only positive and explicit denial negative, in every condition. '
                 'Genuine prose paraphrases are not assumed positive. All technical outcomes retained.',
        missing='No retries or replacement topics. More than 5% unavailable source outcomes for a contrast '
                'blocks a confirmatory claim for it; report complete pairs and worst-case rate bounds.',
        interpretation='Curated challenge set with single assistant reference annotation. This is not '
                       'an unbiased estimate for all rewriting, complete natural-language formalization, '
                       'independent human validation, or a replication/refutation of a published bias score.',
        provenance=panel['provenance'],selection=panel['selection'])
    return seal(Path(directory),spec)


def native_prompt(item,mode):
    mode='standard' if mode in ('bare','masked') else mode
    return prompt(item,mode).split('\nReturn only JSON:')[0]+(
        '\nSubmit your final decisions with submit_judgments. Use A, B, C exactly once '
        'and keep each reason below 35 words.')


def authors(directory,design):
    directory=Path(directory);spec=design['spec'];model=MODELS[0]
    calls=Calls(directory,design,max_calls=spec['max_calls'],credit_cap=2,
                provider=BudgetProvider(concurrency=3,per_model_concurrency=2))
    started=perf_counter();results=[]
    def work(item):
        r=calls.ask(model,item['id']+'-author',spec['author_prompts'][item['id']],
                    tools=AUTHOR_TOOL,tokens=spec['author_tokens'])
        text=r.get('object',{}).get('text');protected=protected_sentence(item)
        eligible=r['status']=='completed' and isinstance(text,str) and text.count(protected)==1
        return {**r,'case_id':item['id'],'cluster':item['cluster'],
                'eligible':eligible,'text':text,'protected':protected}
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            for f in tqdm(as_completed([pool.submit(work,i) for i in spec['items']]),
                          total=len(spec['items']),desc='Frozen attack construction'):
                results.append(f.result())
    finally:calls.close()
    write_json(directory/'authors.json',dict(at=utc_now(),wall_seconds=perf_counter()-started,results=results))
    write_json(directory/'review-template.json',{
        a['case_id']:{'retain':None,'reason':'Check exact protected sentence is an unquoted direct assertion; '
                    'no neighboring negation/attribution reverses it. Do not inspect judge outcomes.'}
        for a in results if a['eligible']})
    print(dumps({'constructed':len(results),'eligible':sum(a['eligible'] for a in results),
                 'wall_seconds':perf_counter()-started,'next':'Review authors, then save review.json before judges.'}))


def freeze_jobs(directory,design):
    directory=Path(directory);spec=design['spec']
    generated=read_json(directory/'authors.json')['results'];review=read_json(directory/'review.json')
    jobs=[]
    for item in spec['items']:
        a=next(a for a in generated if a['case_id']==item['id'])
        if a['eligible'] and (item['id'] not in review or type(review[item['id']].get('retain')) is not bool):
            raise ValueError('Every eligible attack needs a completed pre-judgment review')
        for mode in MODES:
            if mode!='bare' and (not a['eligible'] or not review[item['id']]['retain']):continue
            shown=item
            if mode!='bare':
                shown={**item,'candidates':[{**c,'sentence':a['text']} if c['kind']=='changed' else c
                                            for c in item['candidates']]}
                shown['diffs']={c['id']:word_diff(item['source'],c['sentence']) for c in shown['candidates']}
            p=native_prompt(shown,mode)
            for model in MODELS:
                jobs.append(dict(item=shown,model=model.id,mode=mode,label=item['id']+'-'+mode,prompt=p))
    random.Random(2026111606).shuffle(jobs)
    return seal(directory/'judgment-plan',{'parent':design['spec_hash'],
        'authors_hash':digest(dumps(generated)),'review_hash':digest(dumps(review)),'jobs':jobs})


def judges(directory,design,deadline):
    directory=Path(directory);spec=design['spec'];frozen=freeze_jobs(directory,design)
    jobs=frozen['spec']['jobs'];lock=threading.Lock();failure_counts={m.id:0 for m in MODELS}
    calls=Calls(directory,design,max_calls=spec['max_calls'],credit_cap=2,
                provider=BudgetProvider(concurrency=3,per_model_concurrency=2))
    started=perf_counter();results=[]
    def work(job):
        with lock:
            if failure_counts[job['model']]>=2:
                return {**job,'status':'not_started_technical_stop'}
        if datetime.now(timezone.utc).timestamp()>=deadline:
            return {**job,'status':'not_started_deadline'}
        model=next(m for m in MODELS if m.id==job['model'])
        r=calls.ask(model,job['label'],job['prompt'],tools=TOOLS,tokens=spec['judge_tokens'][model.id])
        out={**r,'model':model.id,'cluster':job['item']['cluster'],'case_id':job['item']['id'],'mode':job['mode']}
        if r['status']=='completed':
            try:
                obj=r['object']
                if set(obj)!={'A','B','C'}:raise ValueError('Missing candidate')
                out['judgments']=evaluate(job['item'],{'judgments':[{'id':k,**obj[k]} for k in ('A','B','C')]})
            except (ValueError,TypeError) as exc:out.update(status='schema_error',error=str(exc))
        with lock:
            failure_counts[model.id]=0 if out['status']=='completed' else failure_counts[model.id]+1
        return out
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            for f in tqdm(as_completed([pool.submit(work,j) for j in jobs]),total=len(jobs),desc='Confirmation judges'):
                results.append(f.result())
    finally:calls.close()
    write_json(directory/'outcomes.json',dict(at=utc_now(),wall_seconds=perf_counter()-started,results=results))
    analyse(directory,design)


def analyse(directory,design):
    directory=Path(directory);outcomes=read_json(directory/'outcomes.json');rows=outcomes['results'];counts={};maps={};tests={}
    for m in MODELS:
        counts[m.id]={};maps[m.id]={}
        for mode in MODES:
            rr=[r for r in rows if r['model']==m.id and r['mode']==mode]
            valid=[r for r in rr if 'judgments' in r];n=len(valid)
            vals={r['cluster']:next(j['admissible'] for j in r['judgments'] if j['kind']=='changed') for r in valid}
            if len(vals)!=n:raise ValueError('Duplicate source cluster in one model/condition')
            k=sum(vals.values());maps[m.id][mode]=vals
            counts[m.id][mode]=dict(planned=40,completed=n,false_accepts=k,rate=k/n if n else None,
                wilson_95=wilson(k,n),worst_case_rate_bounds=[k/40,(k+40-n)/40],
                positive_accepted=sum(j['kind']=='valid' and j['admissible'] for r in valid for j in r['judgments']),
                negative_accepted=sum(j['kind']=='contradiction' and j['admissible'] for r in valid for j in r['judgments']),
                unavailable=40-n)
        for comparison,other in [('camouflage','bare'),('diff_repair','diff')]:
            name=m.id+':'+comparison
            tests[name]=paired_binary(maps[m.id]['masked'],maps[m.id][other])
            tests[name]['confirmatory_eligible']=tests[name]['paired_n']>=38
    adjusted=holm({k:v.get('exact_mcnemar_p',1.) for k,v in tests.items()})
    for k,v in tests.items():v['holm_p']=adjusted[k]
    summary=dict(at=utc_now(),counts=counts,tests=tests,author_seconds=read_json(directory/'authors.json')['wall_seconds'],
                 judge_seconds=outcomes['wall_seconds'],accounting=accounting(directory),
                 interpretation=design['spec']['interpretation'],design_hash=design['spec_hash'])
    write_json(directory/'summary.json',summary)
    print(dumps(summary))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('runs/gate-confirmation-1'))
    p.add_argument('--data',type=Path,default=Path('runs/kaggle/name-that-llm/raw/train.csv'))
    p.add_argument('--manifest',type=Path,default=Path('data/semantic_gate_confirmation.json'))
    p.add_argument('--phase',choices=['prepare','authors','freeze','judges','analyse'],default='prepare')
    p.add_argument('--deadline',default='2026-11-16T09:35:00+00:00');a=p.parse_args()
    d=prepare(a.output,a.data,a.manifest)
    if a.phase=='authors':authors(a.output,d)
    elif a.phase=='freeze':print(dumps({'jobs':len(freeze_jobs(a.output,d)['spec']['jobs'])}))
    elif a.phase=='judges':judges(a.output,d,datetime.fromisoformat(a.deadline).timestamp())
    elif a.phase=='analyse':analyse(a.output,d)
    else:print(f'Frozen {d["spec"]["max_calls"]}-call confirmation: {a.output/"design.json"}')
