"""Small, sealed test of whether editor identity is mistaken for verification."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
import random
import threading

from tqdm import tqdm

from .journal import AdmissionJournal
from .parsing import submission
from .providers import ModelSpec
from .provenance_cases import ARMS, cases, oracle, profiles, prompt, score
from .runtime import BudgetProvider, response_ok, usage
from .storage import digest, dumps, read_json, utc_now, write_json


MODELS = (ModelSpec('gemma-4-31b-it', input_per_million=.09, output_per_million=.34,
                    settings={'temperature':0}),
          ModelSpec('glm-5.3-flash', input_per_million=.15, output_per_million=.5,
                    settings={'temperature':0}))
TOKENS = {'gemma-4-31b-it':1500, 'glm-5.3-flash':8192}


def prepare(directory, data):
    references = profiles(data)
    all_cases = cases()
    spec = {'protocol':'correct-provenance-wrong-trust-v1', 'models':[asdict(m) for m in MODELS],
        'source_hashes':{p.name:digest(p.read_text()) for p in (Path(__file__), Path(__file__).with_name('provenance_cases.py'))},
        'data_sha256':sha256(Path(data).read_bytes()).hexdigest(), 'profiles':references,
        'cases':all_cases, 'tokens':TOKENS, 'timeout':120, 'retries':0,
        'max_calls':42, 'incremental_credit_cap':1.0,
        'discovery':'Two independent copy-only vignettes x four arms x two models, plus one real-check positive control per model: 18 calls.',
        'confirmation':'Only after discovery gate. Four fresh cases x plain/authenticated/typed x two models: 24 calls.',
        'gate':'Both models must show an authenticated-vs-plain audit reversal on at least one discovery case, '+
               'a typed-record repair on that case, and pass the actual-check positive control. Otherwise stop; '+
               'confidence-only movement does not meet the decision-quality promise gate.',
        'scope':'Fictional workflows with exact probabilistic decision oracle. Kaggle identity examples are authentic '+
                'but the processor roles and rates are stipulated, not measured. No claims about actual cryptographic '+
                'authentication or real model editing/verification behavior. Metadata is valid inside the simulator. '+
                'Claims, evidence and identity reference library are identical across arms; only the provenance display varies.',
        'unit':'Matched vignette; multiple arms and model families are not independent random samples.',
        'prompts':{c['id']:{a:prompt(c,a,references) for a in ARMS} for c in all_cases}}
    path = directory/'design.json'
    if path.exists():
        design = read_json(path)
        if design['spec_hash'] != digest(spec):
            raise ValueError('Protocol changed; use a new run directory')
        return design
    design = {'at':utc_now(), 'before_api':True, 'spec_hash':digest(spec), 'spec':spec,
              'budget_before':usage()}
    write_json(path,design)
    return design


def analyse(directory, design):
    results, contrasts, by_model = [], [], {}
    for model in MODELS:
        paired = []
        for case in design['spec']['cases']:
            found = {}
            for arm in ARMS:
                path = directory/'responses'/model.id/f"{case['id']}-{arm}.json"
                if path.exists():
                    r=read_json(path)
                    results.append({'model':model.id,'case':case['id'],'split':case['split'],'arm':arm,**r})
                    if r['status']=='completed':
                        found[arm]=r
            if case['split']!='positive_control' and {'plain','authenticated','typed'}<=found.keys():
                plain,auth,typed=(found[a] for a in ('plain','authenticated','typed'))
                weak=case['weak_origin_label']
                contrast={'model':model.id,'case':case['id'],'split':case['split'],
                    'auth_minus_plain_p_weak':auth['answer']['p_correct'][weak]-plain['answer']['p_correct'][weak],
                    'auth_new_audit_error':plain['score']['audit_correct'] and not auth['score']['audit_correct'],
                    'typed_repairs_auth_audit':not auth['score']['audit_correct'] and typed['score']['audit_correct'],
                    'auth_minus_plain_expected_error_regret':auth['score']['expected_error_regret']-plain['score']['expected_error_regret']}
                if 'editor' in found:
                    contrast['auth_minus_editor_p_weak']=auth['answer']['p_correct'][weak]-found['editor']['answer']['p_correct'][weak]
                    contrast['badge_new_error_vs_editor']=found['editor']['score']['audit_correct'] and not auth['score']['audit_correct']
                contrasts.append(contrast)
                if case['split']=='discovery':paired.append(contrast)
        control=[r for r in results if r['model']==model.id and r['split']=='positive_control' and r['status']=='completed']
        by_model[model.id]={'discovery_pairs':len(paired),
            'auth_new_audit_errors':sum(c['auth_new_audit_error'] for c in paired),
            'typed_repaired_new_errors':sum(c['auth_new_audit_error'] and c['typed_repairs_auth_audit'] for c in paired),
            'positive_control_pass':len(control)==1 and control[0]['score']['audit_correct'] and
                                    control[0]['answer']['p_correct']['A']>=.95}
    complete=all(v['discovery_pairs']==2 for v in by_model.values()) and all(v['positive_control_pass'] for v in by_model.values())
    gate=complete and all(v['typed_repaired_new_errors']>=1 for v in by_model.values())
    now,old=usage(),design['budget_before']
    summary={'at':utc_now(),'scope':design['spec']['scope'],'models':by_model,
        'gate':'ADVANCE_TO_CONFIRMATION' if gate else 'STOP_NO_CROSS_MODEL_DECISION_PROMISE' if complete else 'INCOMPLETE_OR_CONTROL_FAILED',
        'contrasts':contrasts,'results':results,'calls_delta':now['calls']-old['calls'],
        'reported_credits_delta':now['reported_credits']-old['reported_credits'],
        'uncertain_reserve_delta':now['uncertain_reserve']-old['uncertain_reserve']}
    write_json(directory/'summary.json',summary)
    print(dumps({k:v for k,v in summary.items() if k not in ('results','contrasts','scope')}))
    return summary


def run(directory, data, live=False, stage='discovery'):
    directory=Path(directory)
    design=prepare(directory,Path(data))
    if not live:
        print(f"Sealed experiment and held-out cases: {directory/'design.json'}")
        return
    if stage=='confirmation' and analyse(directory,design)['gate']!='ADVANCE_TO_CONFIRMATION':
        raise ValueError('Discovery gate did not pass; confirmation remains untouched')
    provider,lock=BudgetProvider(concurrency=2),threading.Lock()
    count=len(list((directory/'receipts').rglob('*.request.json')))
    def ask(model,case,arm):
        nonlocal count
        label=f"{case['id']}-{arm}"
        path=directory/'responses'/model.id/(label+'.json')
        text=design['spec']['prompts'][case['id']][arm]
        if path.exists():
            row=read_json(path)
            if row['prompt_hash']!=digest(text):raise ValueError('Cached prompt changed')
            return row
        receipt=directory/'receipts'/model.id/label
        restoring=bool(list(receipt.glob('*.request.json')))
        with lock:
            if not restoring:
                now,old=usage(),design['budget_before']
                spent=now['reported_credits']+now['uncertain_reserve']-old['reported_credits']-old['uncertain_reserve']
                reserve=2*((len(text.encode())+4096)*model.input_per_million+TOKENS[model.id]*model.output_per_million)/1e6
                if count>=42 or spent+reserve>1:
                    raise RuntimeError('42-call / one-credit local cap reached')
                count+=1
        row={'prompt_hash':digest(text)}
        try:
            response=AdmissionJournal(receipt).complete(label,provider,model,
                [{'role':'user','content':text}],tools=[],tool_choice='none',
                max_output_tokens=TOKENS[model.id],timeout=120,retries=0)
            row.update(api_status=response.raw.get('status'),served_model=response.raw.get('model'),
                       seconds=response.seconds,usage=response.usage)
            if not response_ok(response,model):raise ValueError('Incomplete response or wrong model')
            answer,encoding=submission(response,[])
            row.update(status='completed',answer=answer,score=score(case,answer),encoding=encoding)
        except Exception as exc:
            row.update(status='unavailable',error=f'{type(exc).__name__}: {exc}')
        write_json(path,row)
        return row
    def one_model(model,bar):
        if stage=='discovery':
            current=[c for c in design['spec']['cases'] if c['split']=='discovery']
            jobs=[(c,a) for c in current for a in ARMS]
            first=jobs.pop(0)
            random.Random(20261116).shuffle(jobs)
            jobs=[first]+jobs+[(next(c for c in design['spec']['cases'] if c['split']=='positive_control'),'typed')]
        else:
            jobs=[(c,a) for c in design['spec']['cases'] if c['split']=='confirmation'
                  for a in ('plain','authenticated','typed')]
            random.Random(20261117).shuffle(jobs)
        for case,arm in jobs:
            row=ask(model,case,arm)
            bar.update(1)
            if row['status']!='completed':
                print(f"Stopped {model.id} after technical noncompletion: {case['id']}/{arm}")
                break
    try:
        total=18 if stage=='discovery' else 24
        with tqdm(total=total,desc=f'Provenance {stage}') as bar, ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(one_model,m,bar) for m in MODELS]
            for f in as_completed(futures):f.result()
    finally:
        provider.close()
    analyse(directory,design)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('runs/provenance-trust-1'))
    parser.add_argument('--data',type=Path,default=Path('runs/kaggle/name-that-llm/raw/train.csv'))
    parser.add_argument('--stage',choices=('discovery','confirmation'),default='discovery')
    parser.add_argument('--live',action='store_true')
    args=parser.parse_args()
    run(args.output,args.data,args.live,args.stage)
