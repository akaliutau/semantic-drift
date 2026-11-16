"""Original no. 3 on fresh natural passages with pre-recorded, local edits.

Natural-language gold is a pre-call linguistic annotation, NOT a mechanical
formalization of arbitrary prose. Exact edit spans and unedited text are kept
for blind human review. The finite-world experiments provide separate gold.
"""

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import difflib
from pathlib import Path
import random
import re
from time import perf_counter

from tqdm import tqdm

from .gate_audit import evaluate
from .pilot_json import Calls, MODELS, accounting, seal
from .runtime import BudgetProvider
from .storage import digest, dumps, read_json, utc_now, write_json


MODES = ('standard', 'checklist', 'diff')
PILOT = (
    ('4a974f4a365c', 'approximately', 'exactly'),
    ('07f409f75b21', 'may', 'will'),
    ('0b3139e017df', 'may extend', 'will extend'),
    ('01b5eec535ac', 'may inherit', 'will inherit'),
    ('1a6913ea9402', 'could eliminate', 'will eliminate'),
    ('652bb0c1942b', 'could keep', 'will keep'),
    ('282c8c47afb7', 'often chase', 'always chase'),
)


def word_diff(source, candidate):
    a, b = source.split(), candidate.split()
    return [{'source': ' '.join(a[i:j]), 'candidate': ' '.join(b[k:l])}
            for op, i, j, k, l in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes()
            if op != 'equal']


def build_item(row, old, new, rng):
    source = row['text']
    if source.count(old) != 1:
        raise ValueError(f"Ambiguous edit in {row['id']}: {old}")
    changed = source.replace(old, new, 1)
    # The positive changes layout/heading only. No sentence is paraphrased.
    valid = '## Report\n\n' + source
    # A deliberately obvious negative control, kept distinct from subtle edits.
    contradiction = 'The following assertion is false: ' + '“' + source + '”'
    candidates = [dict(kind='valid', sentence=valid, gold=True),
                  dict(kind='changed', sentence=changed, gold=False),
                  dict(kind='contradiction', sentence=contradiction, gold=False)]
    rng.shuffle(candidates)
    candidates = [dict(id=chr(65+i), **r) for i, r in enumerate(candidates)]
    start = source.index(old)
    return dict(id=row['id'], cluster=row['prompt_id'], source_model=row['model'], source=source,
                old=old, new=new, candidates=candidates,
                reference=dict(old=old, new=new, span=[start, start+len(old)],
                    context=source[max(0,start-100):start+len(old)+140],
                    changed_commitment='Strengthens uncertainty, frequency, or numerical precision; pre-call annotation.'),
                diffs={c['id']: word_diff(source,c['sentence']) for c in candidates})


def prompt(item, mode):
    instruction = ('Assess whether each candidate preserves the meaning of SOURCE. A meaning-preserving '
        'candidate is admissible as a style-only experimental rewrite. Do not fact-check the source '
        'against outside knowledge. Judge the relationship between the two texts.')
    if mode == 'checklist':
        instruction += (' Check every factual proposition, qualification, negation, possibility versus necessity, '
            'frequency quantifier, scope, and causal claim. Added or dropped propositions and changed '
            'certainty make a candidate inadmissible even if the overall topic and gist match. '
            'Poetic wording alone is not a factual difference; identify the precise changed commitment.')
    text = instruction + '\nSOURCE:\n' + item['source'] + '\nCANDIDATES:\n' + dumps([
        dict(id=c['id'],text=c['sentence']) for c in item['candidates']])
    if mode == 'diff':
        text += ('\nAUTOMATIC WORD DIFFS (textual changes only, no semantic judgment; '
                 'layout or headings need not change meaning):\n' + dumps(item['diffs']))
    return text + ('\nReturn only JSON: {"judgments":[{"id":"A","admissible":true,'
                   '"reason":"at most 35 words"}]}. Include exactly one judgment per candidate.')


def prepare(directory, data):
    rows = {r['id']:r for r in csv.DictReader(Path(data).open())}
    rng = random.Random(2026111604)
    items = [build_item(rows[key], old, new, rng) for key,old,new in PILOT]
    if len({r['cluster'] for r in items}) != len(items):
        raise ValueError('Pilot must have one sample per source-prompt cluster')
    spec = dict(protocol='natural-style-validity-pilot-v1', phase='exploratory',
                source_hash=digest(Path(__file__).read_text()),
                scoring_hash=digest(Path(__file__).with_name('gate_audit.py').read_text()),
                data_hash=digest(Path(data).read_text()), models=[asdict(m) for m in MODELS],
                items=items, modes=MODES, max_calls=len(items)*6, credit_cap=2,
                gold='Pre-call assistant linguistic reference. No independent human gold; exact edit spans retained. '
                     'Not a mechanical semantic proof of arbitrary natural passages.',
                provenance='Yao Yan, Martyna Plomecka, Ashley Oldacre. Name That LLM, Kaggle 2026, CC BY-SA 4.0. '
                           'Synthetic edited texts are derivatives; no original prompts or factual truth labels.',
                prompts={i['id']:{m:prompt(i,m) for m in MODES} for i in items},
                gate='Both models approve at least two altered passages in standard mode, '
                     'accept at least 6/7 positive controls, reject all obvious negative controls; '
                     'no persistent technical failure. Confirmation must use new source clusters.')
    return seal(Path(directory), spec)


def execute(directory, design):
    directory=Path(directory);spec=design['spec'];items=spec['items']
    calls = Calls(directory,design,max_calls=spec['max_calls'],credit_cap=2,
                  provider=BudgetProvider(concurrency=3,per_model_concurrency=2))
    jobs=[(i,m,mode) for i in items for mode in spec['modes'] for m in MODELS]
    random.Random(2026111604).shuffle(jobs)
    results=[];started=perf_counter()
    def work(item,model,mode):
        r=calls.ask(model,item['id']+'-'+mode,spec['prompts'][item['id']][mode],
                    tokens=8192 if model.id.startswith('glm') else 3000)
        out={**r,'model':model.id,'case_id':item['id'],'cluster':item['cluster'],'mode':mode}
        if r['status']=='completed':
            try:out['judgments']=evaluate(item,r['object'])
            except ValueError as exc:out.update(status='schema_error',error=str(exc))
        return out
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            for f in tqdm(as_completed([pool.submit(work,*j) for j in jobs]),
                          total=len(jobs),desc='Natural passage gate audit'):
                results.append(f.result())
    finally:calls.close()
    counts={}
    for m in MODELS:
        counts[m.id]={}
        for mode in spec['modes']:
            rr=[r for r in results if r['model']==m.id and r['mode']==mode]
            jj=[j for r in rr for j in r.get('judgments',[])]
            counts[m.id][mode]=dict(completed=sum(r['status']=='completed' for r in rr),
                changed_accepted=sum(j['kind']=='changed' and j['admissible'] for j in jj),
                valid_accepted=sum(j['kind']=='valid' and j['admissible'] for j in jj),
                contradictions_accepted=sum(j['kind']=='contradiction' and j['admissible'] for j in jj),
                technical_failures=sum(r['status']!='completed' for r in rr))
    signal=all(v['standard']['changed_accepted']>=2 and v['standard']['valid_accepted']>=6
               and v['standard']['contradictions_accepted']==0 for v in counts.values())
    summary=dict(at=utc_now(),counts=counts,results=results,wall_seconds=perf_counter()-started,
                 accounting=accounting(directory),gate='ADVANCE' if signal else 'NO_FRESH_CROSS_MODEL_SIGNAL')
    write_json(directory/'summary.json',summary)
    print(dumps({k:v for k,v in summary.items() if k!='results'}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('runs/natural-gate-pilot-1'))
    p.add_argument('--data',type=Path,default=Path('runs/kaggle/name-that-llm/raw/train.csv'))
    p.add_argument('--live',action='store_true');a=p.parse_args()
    design=prepare(a.output,a.data)
    if a.live:execute(a.output,design)
    else:print(f'Sealed {design["spec"]["max_calls"]} calls: {a.output/"design.json"}')
