"""Fresh, exact-gold follow-up to the original style-only validator result.

The pilot is exploratory. Confirmation must use a new seed and sealed design.
Each request compares a real paraphrase, a changed commitment, and a direct
contradiction. The independent sampling unit is a finite semantic context.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
import random
from time import perf_counter

from tqdm import tqdm

from .claim_worlds import context, make_panel, oracle, statements, verify_witness
from .pilot_json import Calls, MODELS, accounting, seal
from .runtime import BudgetProvider
from .storage import digest, dumps, read_json, utc_now, write_json


MODES = ('document', 'localized', 'checklist')
INTRO = (
    'The laboratory has completed a small validation study. The report below concerns only '
    'the stated test universe, so it makes no claim about equipment outside the study. '
    'The team used a fixed protocol and kept the specimen identifiers in the original order. ',
    'A technical review has examined the following finite test system. The purpose of the '
    'review is to communicate the result clearly rather than recommend an operational change. '
    'Records were retained so that the analysis can be reviewed independently. ',
    'This note summarizes an engineering test. The team separated its observations from '
    'any proposed explanation and used the same definitions throughout the report. '
    'No conclusion about a larger population is intended. ',
)
OUTRO = (
    'The appendix contains the test protocol and the equipment register. The report does not '
    'attempt to explain the origin of the observed pattern. Further testing is planned before '
    'any deployment decision, and the reviewers recommend retaining the original observations.',
    'These results will be discussed at the next technical meeting. The experimental record '
    'will remain available to the reviewers. The next study will use the same naming convention '
    'so that the two reports can be compared without changing the underlying definitions.',
    'The team has archived the study record and will revisit the experiment during the next '
    'review cycle. Questions about maintenance, manufacturing cost, and practical deployment '
    'are outside the scope of this note. Those questions require a separate investigation.',
)


def wrap(case, sentence):
    return (INTRO[case['report_variant'] % len(INTRO)] + '\n\n' + sentence + '\n\n'
            + OUTRO[(case['report_variant'] // len(INTRO)) % len(OUTRO)])


def build_cases(seed, size, exclude=()):
    rng = random.Random(seed)
    result = []
    for c in make_panel(seed, size, exclude=exclude):
        c = {**c, 'report_variant': rng.randrange(9)}
        source = statements(c, 'source')[c['source_variant']]
        valid = statements(c, 'valid')[0]
        wrong = statements(c, 'wrong')[c['baseline_variant'] % 4]
        rows = [dict(kind='valid', sentence=valid, gold=True),
                dict(kind='changed', sentence=wrong, gold=False),
                dict(kind='contradiction', sentence='It is false that ' + source[0].lower() + source[1:], gold=False)]
        rng.shuffle(rows)
        rows = [dict(id=chr(65+i), **r) for i, r in enumerate(rows)]
        if not verify_witness(c, oracle(c)['counterexample']):
            raise ValueError('Unreplayable semantic change')
        result.append(dict(case=c, source_sentence=source, candidates=rows,
                           oracle=oracle(c), valid_oracle=oracle(c, 'valid')))
    return result


def prompt(item, mode):
    c = item['case']
    instruction = ('Assess whether each candidate preserves the meaning of SOURCE. A meaning-preserving '
        'candidate is admissible as a style-only experimental rewrite. Do not fact-check the source '
        'against outside knowledge. Judge the relationship between the two texts.')
    if mode == 'checklist':
        instruction += (' Check every factual proposition, qualification, negation, possibility versus necessity, '
            'frequency quantifier, scope, and causal claim. Added or dropped propositions and changed '
            'certainty make a candidate inadmissible even if the overall topic and gist match. '
            'Poetic wording alone is not a factual difference; identify the precise changed commitment.')
    source = item['source_sentence'] if mode == 'localized' else wrap(c, item['source_sentence'])
    candidates = [dict(id=r['id'], text=r['sentence'] if mode == 'localized' else wrap(c, r['sentence']))
                  for r in item['candidates']]
    return (instruction + '\nBACKGROUND (same for all versions):\n' + context(c)
            + '\nSOURCE:\n' + source + '\nCANDIDATES:\n' + dumps(candidates)
            + '\nReturn only JSON: {"judgments":[{"id":"A","admissible":true,'
              '"reason":"at most 35 words"}]}. Include exactly one judgment per candidate.')


def evaluate(item, obj):
    if not isinstance(obj, dict) or set(obj) != {'judgments'} or not isinstance(obj['judgments'], list):
        raise ValueError('Expected judgments')
    expected = {r['id']: r for r in item['candidates']}
    if len(obj['judgments']) != 3 or {r.get('id') for r in obj['judgments']} != expected.keys():
        raise ValueError('Missing/duplicate candidate')
    rows = []
    for r in obj['judgments']:
        if (set(r) != {'id', 'admissible', 'reason'} or type(r['admissible']) is not bool
                or not isinstance(r['reason'], str)):
            raise ValueError('Malformed judgment')
        g = expected[r['id']]
        rows.append({**r, 'kind': g['kind'], 'gold': g['gold'], 'correct': r['admissible'] == g['gold']})
    return rows


def run(directory, seed, size, live, exclude=()):
    directory = Path(directory)
    items = build_cases(seed, size, exclude)
    spec = dict(protocol='original-gate-fresh-followup-v1', seed=seed, size=size,
                source_hash=digest(Path(__file__).read_text()),
                oracle_hash=digest(Path(__file__).with_name('claim_worlds.py').read_text()),
                models=[asdict(m) for m in MODELS], items=items, modes=MODES,
                prompts={x['case']['id']: {m: prompt(x, m) for m in MODES} for x in items},
                max_calls=size*6, credit_cap=2,
                scope='Controlled English; five logical templates. Unchanged report paragraphs are '
                      'identical in source and candidates. Contexts deduplicated under renaming. '
                      'Localized is an information intervention, not a second independent sample.',
                gate='Both judges accept >=2 changed document claims, accept >=75% valid paraphrases, '
                     'and reject >=75% contradictions. Pilot is excluded from confirmation.')
    design = seal(directory, spec)
    if not live:
        print(f'Sealed {size*6}-call follow-up: {directory / "design.json"}')
        return
    calls = Calls(directory, design, max_calls=size*6, credit_cap=2,
                  provider=BudgetProvider(concurrency=3, per_model_concurrency=2))
    jobs = [(i, m, mode) for i in items for mode in MODES for m in MODELS]
    random.Random(seed+1).shuffle(jobs)
    started = perf_counter()
    results = []
    def work(item, model, mode):
        r = calls.ask(model, item['case']['id']+'-'+mode,
                      spec['prompts'][item['case']['id']][mode], tokens=8192 if model.id.startswith('glm') else 3000)
        row = {**r, 'model': model.id, 'case_id': item['case']['id'],
               'semantic_id': item['case']['semantic_id'], 'family': item['case']['family'], 'mode': mode}
        if r['status'] == 'completed':
            try:
                row['judgments'] = evaluate(item, r['object'])
            except ValueError as exc:
                row.update(status='schema_error', error=str(exc))
        return row
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            for f in tqdm(as_completed([pool.submit(work, *j) for j in jobs]),
                          total=len(jobs), desc='Original validator follow-up'):
                results.append(f.result())
    finally:
        calls.close()
    counts = {}
    for m in MODELS:
        counts[m.id] = {}
        for mode in MODES:
            rr = [r for r in results if r['model'] == m.id and r['mode'] == mode]
            jj = [j for r in rr for j in r.get('judgments', [])]
            counts[m.id][mode] = dict(completed=sum(r['status']=='completed' for r in rr),
                changed_accepted=sum(j['kind']=='changed' and j['admissible'] for j in jj),
                valid_accepted=sum(j['kind']=='valid' and j['admissible'] for j in jj),
                contradictions_accepted=sum(j['kind']=='contradiction' and j['admissible'] for j in jj),
                technical_failures=sum(r['status']!='completed' for r in rr))
    gate = all((v['document']['changed_accepted'] >= 2 and
                v['document']['valid_accepted'] >= .75*size and
                v['document']['contradictions_accepted'] <= .25*size and
                all(x['technical_failures']==0 for x in v.values())) for v in counts.values())
    summary = dict(at=utc_now(), counts=counts, results=results, wall_seconds=perf_counter()-started,
                   accounting=accounting(directory), gate='ADVANCE' if gate else 'NO_FRESH_CROSS_MODEL_SIGNAL')
    write_json(directory/'summary.json', summary)
    print(dumps({k:v for k,v in summary.items() if k!='results'}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('runs/original-gate-pilot-1'))
    p.add_argument('--seed', type=int, default=2026111603)
    p.add_argument('--size', type=int, default=10)
    p.add_argument('--exclude-design', type=Path, action='append', default=[])
    p.add_argument('--live', action='store_true')
    a = p.parse_args()
    excluded = []
    for path in a.exclude_design:
        old = read_json(path)['spec']
        old_cases = old.get('cases', [x['case'] for x in old.get('items', [])])
        excluded.extend(c['semantic_id'] for c in old_cases)
    run(a.output, a.seed, a.size, a.live, excluded)
