"""Four-passage style-mask feasibility pilot, not an epistemics benchmark.

Original generators, renderers, and target styles are recorded separately.
Classifier scores on rewritten text are style-likeness probes, not calibrated
authorship probabilities. No labels are inferred from the desired mask.
"""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
import re
import threading

from tqdm import tqdm

from .journal import AdmissionJournal
from .parsing import submission
from .providers import ModelSpec
from .runtime import BudgetProvider, response_ok, usage
from .storage import digest, read_json, utc_now, write_json
from .style_lab import load_rows, split_groups, surface_features


RENDERER = ModelSpec("gemma-4-31b-it", input_per_million=.09,
                     output_per_million=.34, settings={"temperature": 0})
SOURCE = "gpt-5.5"
TARGETS = {"S1": "claude-opus-5", "S2": "gemini-3.1-pro-preview"}
CONDITIONS = ("neutral", "S1", "S2")


def prompt_for(item, examples):
    return ("This is a controlled writing-style experiment. Return ONE JSON object "
        "with exactly three string fields: neutral, S1, S2. Each is a rewrite of "
        "the ORIGINAL. neutral uses plain, natural prose; S1 imitates the surface "
        "style of the S1 EXAMPLES; S2 imitates S2. Do not mention these labels or "
        "the examples' subjects. Preserve ALL original propositions, narrative "
        "events, names, numbers, qualifications, negations, and uncertainty. "
        "Do not correct, fact-check, add evidence, new factual claims, or new "
        "conclusions. Keep each rewrite within 10% of the original word count. "
        "Preserve digit-written numbers exactly. In fiction preserve events "
        "and sensory details. You may change rhythm, syntax, diction, paragraph "
        "breaks and punctuation. Output only the final JSON, no discussion.\n\n"
        + "STYLE EXAMPLES (reference only):\n" + str(examples)
        + "\n\nORIGINAL (" + str(len(item['text'].split())) + " words):\n"
        + item["text"])


def prepare(data, baseline, directory):
    split = split_groups(load_rows(data))
    groups = sorted({r['prompt_id'] for r in split['train']})[:3]
    examples = {s: [r['text'] for g in groups for r in split['train']
                    if r['prompt_id'] == g and r['model'] == model]
                for s, model in TARGETS.items()}
    test_groups = sorted({r['prompt_id'] for r in split['test']})[:4]
    items = [next(r for r in split['test'] if r['prompt_id'] == g
                  and r['model'] == SOURCE) for g in test_groups]
    spec = {"protocol": "style-mask-feasibility-v1", "renderer": asdict(RENDERER),
        "selection": "First four sorted held-out prompt groups; fixed GPT-5.5 source. "
                     "Two named target styles fixed before any rewriting, not selected by test scores.",
        "targets": TARGETS, "example_prompt_ids": groups, "examples": examples,
        "items": items, "source_hash": digest(Path(__file__).read_text()),
        "baseline_design_hash": digest(read_json(baseline / 'design.json')),
        "model_hashes": {p.name: digest(p.read_bytes().hex()) for p in sorted(baseline.glob('*.joblib'))},
        "tokens": 4000, "timeout": 120, "max_calls": 4, "local_credit_cap": .1,
        "retries": 0, "baseline": str(baseline.resolve()),
        "prompts": {r['id']: prompt_for(r, examples) for r in items},
        "scope": "Feasibility only. Rewrites have a new renderer, so original source labels "
                 "are NOT ground-truth final authorship. Style scores are OOD and not calibrated "
                 "probabilities. Manual semantic audit is required; numeric/length checks are insufficient. "
                 "There are no truth labels, judge decisions, or self-awareness outcomes in this pilot."}
    path = directory / 'design.json'
    if path.exists():
        design = read_json(path)
        if design['spec_hash'] != digest(spec):
            raise ValueError("Pilot changed; choose a fresh output directory")
        return design
    design = {"at": utc_now(), "before_api": True, "spec_hash": digest(spec),
              "spec": spec, "budget_before": usage()}
    write_json(path, design)
    return design


def mechanical_checks(original, rewritten):
    ratio = len(rewritten.split()) / max(1, len(original.split()))
    numbers = lambda text: Counter(re.findall(r"\d+(?:\.\d+)?", text))
    return {"word_count_ratio": ratio, "length_within_10_percent": .9 <= ratio <= 1.1,
            "digit_numbers_preserved": numbers(original) == numbers(rewritten),
            "semantic_fidelity": "UNREVIEWED: these checks do not certify meaning"}


def style_scores(artifact, text):
    raw = [text] if artifact['scaler'] is None else [surface_features(text)]
    x = artifact['vectorizer'].transform(raw)
    if artifact['scaler'] is not None:
        x = artifact['scaler'].transform(x)
    scores = artifact['classifier'].decision_function(x)[0]
    values = {str(c): float(v) for c, v in zip(artifact['classes'], scores)}
    return {"decision_scores": values, "top1_style_label": max(values, key=values.get)}


def analyse(directory, design):
    import joblib
    baseline = Path(design['spec']['baseline'])
    artifacts = {p.stem: joblib.load(p) for p in baseline.glob('*.joblib')}
    results = []
    for item in design['spec']['items']:
        path = directory / 'responses' / (item['id'] + '.json')
        if not path.exists():
            continue
        row = read_json(path)
        result = {"id": item['id'], "prompt_id": item['prompt_id'],
                  "semantic_source": SOURCE, "renderer": RENDERER.id,
                  "status": row['status']}
        if row['status'] == 'completed':
            texts = {"original": item['text'], **row['rewrites']}
            result['conditions'] = {c: {"checks": mechanical_checks(item['text'], t),
                "scores": {k: style_scores(a, t) for k, a in artifacts.items()}}
                for c, t in texts.items()}
            result['mask_probes'] = []
            for condition, target in TARGETS.items():
                for kind in artifacts:
                    get = lambda c: result['conditions'][c]['scores'][kind]['decision_scores']
                    margin = lambda c: get(c)[target] - get(c)[SOURCE]
                    result['mask_probes'].append({"condition": condition, "target": target,
                        "classifier": kind,
                        "target_minus_source_margin_original": margin('original'),
                        "target_minus_source_margin_neutral": margin('neutral'),
                        "target_minus_source_margin_masked": margin(condition),
                        "delta_vs_neutral": margin(condition) - margin('neutral')})
        results.append(result)
    now, before = usage(), design['budget_before']
    summary = {"scope": design['spec']['scope'], "passages": len(results),
        "completed": sum(r['status'] == 'completed' for r in results),
        "api_calls": now['calls'] - before['calls'],
        "reported_credits_delta": now['reported_credits'] - before['reported_credits'],
        "uncertain_reserve_delta": now['uncertain_reserve'] - before['uncertain_reserve'],
        "results": results}
    write_json(directory / 'summary.json', summary)
    print({k:v for k,v in summary.items() if k != 'results'})


def run(data, baseline, directory, live):
    directory = Path(directory)
    design = prepare(Path(data), Path(baseline), directory)
    if not live:
        print(f"Sealed four-call pilot: {directory / 'design.json'}; use --live to execute")
        return
    provider, lock = BudgetProvider(concurrency=2), threading.Lock()
    calls = len(list((directory / 'receipts').rglob('*.request.json')))
    def ask(item):
        nonlocal calls
        label = item['id']
        path = directory / 'responses' / (label + '.json')
        if path.exists():
            return read_json(path)
        prompt = design['spec']['prompts'][label]
        restoring = bool(list((directory / 'receipts' / label).glob('*.request.json')))
        with lock:
            if not restoring:
                now, old = usage(), design['budget_before']
                spent = now['reported_credits'] + now['uncertain_reserve'] - old['reported_credits'] - old['uncertain_reserve']
                reserve = ((len(prompt.encode()) + 4096)*RENDERER.input_per_million
                           + 4000*RENDERER.output_per_million) / 1e6
                if calls >= 4 or spent + 2*reserve > .1:
                    raise RuntimeError("Four-call / .1-credit pilot cap reached")
                calls += 1
        row = {"prompt_hash": digest(prompt), "renderer": RENDERER.id}
        try:
            response = AdmissionJournal(directory / 'receipts' / label).complete(
                label, provider, RENDERER, [{"role":"user", "content":prompt}],
                tools=[], tool_choice='none', max_output_tokens=4000, timeout=120, retries=0)
            row.update(api_status=response.raw.get('status'), served_model=response.raw.get('model'),
                       usage=response.usage, seconds=response.seconds)
            if not response_ok(response, RENDERER):
                raise ValueError("Incomplete response or wrong model")
            obj, encoding = submission(response, [])
            if set(obj) != set(CONDITIONS) or any(not isinstance(v, str) or not v.strip() for v in obj.values()):
                raise ValueError("Expected exactly three nonempty rewrite strings")
            row.update(status='completed', rewrites=obj, encoding=encoding)
        except Exception as exc:
            row.update(status='unavailable', error=f"{type(exc).__name__}: {exc}")
        write_json(path, row)
        return row
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(ask, item) for item in design['spec']['items']]
            for future in tqdm(as_completed(futures), total=len(futures), desc='Four style-mask passages'):
                future.result()
    finally:
        provider.close()
    analyse(directory, design)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('runs/kaggle/name-that-llm/raw/train.csv'))
    parser.add_argument('--baseline', type=Path, default=Path('runs/kaggle/name-that-llm/baseline-1'))
    parser.add_argument('--output', type=Path, default=Path('runs/kaggle/name-that-llm/mask-pilot-1'))
    parser.add_argument('--live', action='store_true')
    args = parser.parse_args()
    run(args.data, args.baseline, args.output, args.live)
