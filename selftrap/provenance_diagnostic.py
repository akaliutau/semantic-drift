"""Four-call diagnostic of one secondary instrument/processor confusion.

Exploratory follow-up to provenance-trust-1, not its untouched confirmation set.
The unaided-guess rate is deliberately separated from direct instrument output.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path

from tqdm import tqdm

from .provenance_cases import cases, profiles, prompt, score
from .provenance_pilot import MODELS, TOKENS
from .runner import request
from .runtime import BudgetProvider, usage
from .storage import digest, dumps, read_json, utc_now, write_json


def run(directory, data, live):
    directory=Path(directory)
    case=deepcopy(next(c for c in cases() if c['split']=='positive_control'))
    case['id']='direct-instrument-low-guess-accuracy'
    case['reports']['A']['fresh_answer_probability_correct']=.20
    case['reports']['B']['fresh_answer_probability_correct']=.98
    references=profiles(data)
    prompts={arm:prompt(case,arm,references).replace(
        'fresh_answer_probability_correct describes a processor when it independently answers '
        'a NEW question with test access.',
        'fresh_answer_probability_correct is the processor\'s historical accuracy for UNAIDED '
        'GUESSES BEFORE access to measurements. It is NOT an instrument or transcription error rate. '
        'The read_private_test receipt below is delivered DIRECTLY TO YOU from an independently '
        'authenticated, error-free instrument, not inferred or transcribed by the processor. '
        'Its reported PASS is the actual specimen result. The processor cannot alter this receipt.')
        for arm in ('plain','authenticated')}
    spec={'protocol':'direct-instrument-trust-diagnostic-v1',
        'reason':'A secondary discovery response used processor fresh-answer accuracy .98 instead of the '
                 'zero-error instrument. This phase disambiguates the definition and uses a .20 guesser, '
                 'so the same confusion would cause a decision error.',
        'case':case,'prompts':prompts,'tokens':TOKENS,'calls':4,
        'scope':'One exploratory vignette x two arms x two families. Not independent confirmation or a population estimate.',
        'source_hash':digest(Path(__file__).read_text())}
    manifest=directory/'design.json'
    if manifest.exists():
        design=read_json(manifest)
        if design['spec_hash']!=digest(spec):raise ValueError('Changed diagnostic; use a new directory')
    else:
        design={'at':utc_now(),'before_api':True,'spec':spec,'spec_hash':digest(spec),'budget_before':usage()}
        write_json(manifest,design)
    if not live:
        print(f'Sealed diagnostic: {manifest}')
        return
    provider=BudgetProvider(concurrency=2)
    def one_model(model):
        rows=[]
        for arm in prompts:
            settings={'budget_before':design['budget_before'],'tokens':TOKENS[model.id],'seconds':120,'max_calls':4}
            result=request(directory/'responses'/model.id,provider,model,prompts[arm],arm,settings)
            row={'model':model.id,'arm':arm,**result}
            if result['status']=='valid':row['score']=score(case,result['object'])
            rows.append(row)
            if result['status']!='valid':break
        return rows
    rows=[]
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            for f in tqdm(as_completed([pool.submit(one_model,m) for m in MODELS]),total=2,desc='Two diagnostic model families'):
                rows.extend(f.result())
    finally:
        provider.close()
    now,old=usage(),design['budget_before']
    summary={'at':utc_now(),'rows':rows,'calls_delta':now['calls']-old['calls'],
        'reported_credits_delta':now['reported_credits']-old['reported_credits'],
        'uncertain_reserve_delta':now['uncertain_reserve']-old['uncertain_reserve'],
        'scope':spec['scope']}
    write_json(directory/'summary.json',summary)
    print(dumps(summary))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('runs/provenance-diagnostic-1'))
    parser.add_argument('--data',type=Path,default=Path('runs/kaggle/name-that-llm/raw/train.csv'))
    parser.add_argument('--live',action='store_true')
    args=parser.parse_args()
    run(args.output,args.data,args.live)
