"""Finite-world false-certification study: pilot, frozen policies, confirmation.

Confirmation data are not sent until a separate protocol and sample size are
sealed. Original sponsor ledger remains binding across every phase.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from tqdm import tqdm

from .claim_worlds import FAMILIES, context, make_panel, oracle, statements, variants, verify_witness
from .pilot_json import Calls, MODELS, accounting, seal
from .runtime import BudgetProvider, LEDGER, usage
from .storage import digest, dumps, read_json, utc_now, write_json
from .throttle import file_lock


TOKENS={'gemma-4-31b-it':2000,'glm-5.3-flash':8192}
ARMS=('random','lexical','program','witness','valid','valid_checked')


def lexical(case):
    source=statements(case,'source')[case['source_variant']].lower()
    words=set(source.replace('.','').split())
    return max(range(12),key=lambda i:len(words&set(variants(case)[i].lower().replace('.','').split()))/
               max(1,len(words|set(variants(case)[i].lower().replace('.','').split()))))


def judge_prompt(case,candidate,*,evidence=None):
    s=statements(case,'source')[case['source_variant']]
    text=('You are a strict semantic validator. Decide whether SOURCE and REWRITE have exactly '
        'the same logical content UNDER the stated finite universe and known facts. '
        'EQUIVALENT means they have the same truth value in EVERY world satisfying the known facts. '
        'One permitted world where they differ is enough for DIFFERENT. Match of topic, gist, '
        'or plausibility is insufficient. Check quantifier scope, necessary/sufficient direction, '
        'negation, possibility/joint possibility, and inclusive/strict numeric bounds. '
        'Unknown facts may be true or false. Boolean properties are logical conditions, not causal '
        'claims. Passed/failed and on/off are exhaustive opposites. No facts may be inferred from '
        'real-world stereotypes. A rewrite may restate source wording, but it must not add or drop '
        'logical commitments relative to the background.\n\nUNIVERSE AND FACTS:\n'+context(case)
        +'\nSOURCE:\n'+s+'\nREWRITE:\n'+candidate)
    if evidence is not None:
        text+=('\nINDEPENDENT FINITE-WORLD CHECKER RECEIPT: the checker exhaustively enumerated '
               'the stated Boolean universe. Its data below are exact for this universe. '
               'Use the receipt to check your verdict; do not infer anything outside its scope.\n'+dumps(evidence))
    return text+'\nReturn JSON only: {"equivalent":false,"why":"at most 25 words"}.'


def evaluate(obj,gold):
    if not isinstance(obj,dict) or set(obj)!={'equivalent','why'} or type(obj['equivalent']) is not bool or not isinstance(obj['why'],str):
        raise ValueError('Expected equivalent boolean and short reason')
    return {'correct':obj['equivalent']==gold,'false_accept':obj['equivalent'] and not gold,
            'false_reject':not obj['equivalent'] and gold}


def paced_probe(directory):
    state=LEDGER.parent/'shared-admission/state.json';lock=state.with_name('state.lock')
    with file_lock(lock):
        data=read_json(state)
        if data['stopped']:raise ValueError('Shared collection circuit is stopped')
        old=data['interval'];new=data['config']['interval']
        write_json(Path(directory)/'pacing-probe.json',{'at':utc_now(),'old_interval':old,'new_interval':new,
            'scope':'Bounded return to configured admission interval; RPM/TPM/concurrency, circuit and credit limits unchanged. Any 429 triggers shared backoff.'})
        data['interval']=new
        data['events'].append({'at':utc_now(),'reason':'Bounded certificate-study pacing probe','old_interval':old,'new_interval':new})
        write_json(state,data)


def prepare_pilot(directory):
    cases=make_panel(2026111601,20)
    jobs=[]
    for i,c in enumerate(cases):
        # Three negatives and one positive per family, on separate semantic cases.
        positive=i//5==3
        index=(i//5)*4+(i%3)
        candidate=statements(c,'valid')[0] if positive else variants(c)[index%12]
        for model in MODELS:
            jobs.append({'case':c,'model':model.id,'arm':'valid' if positive else 'attack',
                'candidate':candidate,'gold':positive,'label':c['id']+'-'+('valid' if positive else 'attack'),
                'prompt':judge_prompt(c,candidate)})
    spec={'protocol':'semantic-certificate-pilot-v1','source_hash':digest(Path(__file__).read_text()),
        'oracle_hash':digest(Path(__file__).with_name('claim_worlds.py').read_text()),
        'models':[asdict(m) for m in MODELS],'tokens':TOKENS,'cases':cases,'jobs':jobs,
        'max_calls':40,'credit_cap':1,'workers':3,'per_model_concurrency':2,
        'scope':'Controlled English with finite interpretation fixed by renderer; not arbitrary-text formalization. '
                'Semantic contexts deduplicated under row/column renaming. Pilot cases excluded from confirmation.',
        'gate':'At least two false accepts on negative cases per judge, at least four of five valid controls accepted '
               'per judge, and no persistent API/schema failure. Otherwise no large confirmation.'}
    for c in cases:
        if not verify_witness(c,oracle(c)['counterexample']):raise ValueError('Invalid counterexample')
    return seal(directory,spec)


def execute_jobs(directory,design,jobs,*,max_calls,deadline=None):
    provider=BudgetProvider(concurrency=3,per_model_concurrency=2)
    calls=Calls(directory,design,max_calls=max_calls,credit_cap=5,provider=provider)
    started=perf_counter();results=[]
    def work(job):
        if deadline is not None and datetime.now(timezone.utc).timestamp()>deadline:
            return {**job,'status':'not_started_deadline'}
        model=next(m for m in MODELS if m.id==job['model'])
        row=calls.ask(model,job['label'],job['prompt'],tokens=TOKENS[model.id])
        result={**job,**row}
        if row['status']=='completed':
            try:result['score']=evaluate(row['object'],job['gold'])
            except ValueError as exc:result.update(status='schema_error',error=str(exc))
        return result
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            for f in tqdm(as_completed([pool.submit(work,j) for j in jobs]),total=len(jobs),desc='Certificate judgments'):
                results.append(f.result())
    finally:calls.close()
    return results,perf_counter()-started


def pilot(directory,live,probe):
    directory=Path(directory);design=prepare_pilot(directory)
    if not live:print(f'Sealed 40-call timing/signal pilot: {directory/"design.json"}');return
    if probe and not (directory/'pacing-probe.json').exists():paced_probe(directory)
    results,seconds=execute_jobs(directory,design,design['spec']['jobs'],max_calls=40)
    counts={}
    for m in MODELS:
        rows=[r for r in results if r['model']==m.id]
        negative=[r for r in rows if not r['gold'] and 'score' in r]
        positive=[r for r in rows if r['gold'] and 'score' in r]
        counts[m.id]={'negative_completed':len(negative),'false_accepts':sum(r['score']['false_accept'] for r in negative),
            'positive_completed':len(positive),'valid_accepted':sum(r['object']['equivalent'] for r in positive),
            'technical_failures':sum(r['status']!='completed' for r in rows)}
    gate=all(v['false_accepts']>=2 and v['valid_accepted']>=4 and v['technical_failures']==0 for v in counts.values())
    summary={'at':utc_now(),'results':results,'counts':counts,'wall_seconds':seconds,'accounting':accounting(directory),
        'gate':'ADVANCE' if gate else 'STOP_OR_DIAGNOSE_NO_SCALING','throttle':read_json(LEDGER.parent/'shared-admission/state.json')['interval']}
    write_json(directory/'summary.json',summary)
    print(dumps({k:v for k,v in summary.items() if k!='results'}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('runs/certificate-pilot-1'))
    p.add_argument('--live',action='store_true');p.add_argument('--probe-pacing',action='store_true')
    a=p.parse_args();pilot(a.output,a.live,a.probe_pacing)
