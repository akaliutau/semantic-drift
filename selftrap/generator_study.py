"""Model-written Nonogram generators as prospective personal failure hypotheses."""
import argparse
from pathlib import Path
from .nonogram import RULES,validate_case,actor_view
from .programs import run_generator
from .runner import request,parallel
from .runtime import ROOT,BudgetProvider,MODELS,usage
from .storage import digest,dumps,read_json,utc_now,write_json

OUT=ROOT/'runs/new-search/nonogram-generators-1'

def prepare():
    if (OUT/'manifest.json').exists():return read_json(OUT/'manifest.json')
    m={'at':utc_now(),'candidate':'real_logic_game_failure_generators','budget_before':usage(),'tokens':10000,'seconds':160,
        'construction':'One Python generate(seed) program/model, one validity-only repair. Development seed 0; freeze before hidden seeds 101,307. At most eight inspect calls/seed.',
        'baseline_seeds':[101,307],'confirmation_seeds':[503,709],
        'gate':'One SELF-targeted generator causes its predicted wrong cell value at BOTH fresh seeds, and the peer is correct at least once. No invalid-seed replacement.',
        'diagnosis_gate':'On two further seeds, the frozen diagnostic hint corrects target errors while a matched single-cell control hint does not. Both hints preserve the same unique solution and cannot reveal query.',
        'novelty':'The program is an executable, generative prediction of a specific logical mistake and a discriminating repair; creativity and generic puzzle hardness are not scored.',
        'source':{f:(ROOT/'selftrap'/f).read_text() for f in ['generator_study.py','nonogram.py','programs.py','runner.py']}}
    write_json(OUT/'manifest.json',m);return m

def code_tool():
    return {'type':'function','name':'submit_generator','description':'Submit pure Python source defining generate(seed).',
        'parameters':{'type':'object','additionalProperties':False,'required':['code'],
            'properties':{'code':{'type':'string','maxLength':18000}}}}

def invent():
    m=prepare();p=BudgetProvider()
    try:
        def job(model):
            folder=OUT/'authors'/model.id
            if (folder/'result.json').exists():return read_json(folder/'result.json')
            text='''Implement a PYTHON GENERATOR of real Nonogram puzzles that exposes a
SPECIFIC logical mistake in a fresh copy of YOUR model, preferably while a peer
answers correctly. We score prediction and diagnosis, NOT creativity or merely
making hard puzzles. Invent the bitmap construction, query selection, and the
mechanism. You must implement generate(seed), not manually describe puzzles.

Runtime: Python, pure computation. random is pre-supplied, so use
rng=random.Random(seed); DO NOT import anything. Basic loops, comprehensions,
lists/dicts, string join/count/index and standard numeric/list builtins work.
No files, network, eval, while loops, classes, or private attributes. Hard caps:
3 CPU seconds, 256MB memory, 8 inspect(bitmap) calls per seed.

An ordinary Nonogram bitmap is a list of 5 or 7 equal-length strings of 0/1
(1=BLACK, 0=WHITE). Clues are computed from maximal BLACK runs. The bitmap is
ALWAYS a winning witness, so you need not reason out a solution. Choose 7x7
unless a smaller puzzle better targets your claimed mechanism.
Helper inspect(bitmap) returns:
  unique: whether a COMPLETE exact search proved a unique solution;
  row_patterns / column_patterns: lists of 0/1 strings allowed per individual
  line (before combining lines). You can use these to seek ambiguity requiring
  genuine cross-line inference. It returns no model performance information.
Use inspect to avoid invalid puzzles. You may test at most eight bitmap
candidates per seed. All final puzzles must be unique; unknown/timeout fails.

Return exactly these fields from generate(seed):
 bitmap: solution strings;
 query: [row,column], zero-based;
 predicted_wrong: 0 or 1, OPPOSITE the true query bit;
 diagnostic_hint: a DIFFERENT cell whose true value should correct your
     specific misconception on this puzzle;
 control_hint: another distinct cell, same one-cell information budget, that
     should NOT correct the misconception;
 p_wrong: two probabilities of the EXACT wrong answer, rows [Qwen,Gemma];
 diagnosis: <=60 words explaining the logical error and why the diagnostic hint
     should help while the control hint should not.
Query and both hints must be distinct. Do not reveal hints in baseline.

Actors see clues, all individual-line candidate patterns, and the query; they
choose only the queried bit through a native submit_cell tool (no full-grid
output). Unique solution is stated. 10k tokens/160s, no external code execution.
Neither bitmap, generator, diagnosis, forecasts nor author identity is shown.
Future seeds are held out. Use seed to create a FAMILY, not a constant bitmap.
Call submit_generator with concise working source. No imports or prose outside
that function call. One validity-only repair is allowed; never actor feedback.
'''+dumps({'your_model':model.id,'models':[x.id for x in MODELS],'game_rules':RULES})
            error=None;r={}
            for attempt in range(2):
                prompt=text if error is None else text+'\nCompiler feedback on development seed 0: '+error+'\nReturn the complete corrected generator, keeping the intended target mechanism.'
                r=request(folder,p,model,prompt,f'code-{attempt}',m,[code_tool()]);r.update(model=model.id)
                if r['status']!='valid':break
                try:
                    code=r['object']['code'];generated=run_generator(code,0)
                    case=validate_case(generated['value']);r.update(code=code,development_case=case,inspect_calls=generated['inspect_calls']);error=None
                except (ValueError,TypeError,KeyError,SyntaxError) as exc:
                    error=str(exc);r.update(status='invalid_program',reason=error)
                write_json(folder/f'validation-{attempt}.json',{'at':utc_now(),'error':error})
                if error is None:break
            write_json(folder/'result.json',r);return r
        authors=parallel(MODELS,job,'Implement real Nonogram failure generators');write_json(OUT/'authors.json',authors)
        write_json(OUT/'code_seal.json',{'at':utc_now(),'authors_hash':digest(authors),'before_heldout_seed_generation':True})
        cases=[]
        for a in authors:
            if a['status']!='valid':continue
            for seed in m['baseline_seeds']:
                record={'author':a['model'],'seed':seed,'code_hash':digest(a['code'])}
                try:record.update(status='valid',case=validate_case(run_generator(a['code'],seed)['value']))
                except (ValueError,KeyError,TypeError) as exc:record.update(status='invalid',reason=str(exc))
                cases.append(record)
        write_json(OUT/'cases.json',cases);write_json(OUT/'case_seal.json',{'at':utc_now(),'cases_hash':digest(cases),'authors_hash':digest(authors)})
        print(dumps({'authors':[(a['model'],a['status'],a.get('reason')) for a in authors],'cases':[(c['author'],c['seed'],c['status']) for c in cases]}))
    finally:p.close()

def cell_tool():
    return {'type':'function','name':'submit_cell','description':'Submit the value of the queried Nonogram cell.',
        'parameters':{'type':'object','additionalProperties':False,'required':['value'],
            'properties':{'value':{'type':'integer','enum':[0,1]}}}}

def actor_job(j,p,m,phase):
    record,model,arm=j;case=record['case'];job_id=f"{record['author']}-{record['seed']}-{model.id}-{arm}";folder=OUT/phase/job_id
    if (folder/'result.json').exists():return read_json(folder/'result.json')
    prompt=RULES+'\n'+dumps(actor_view(case,arm))
    r=request(folder,p,model,prompt,'cell',m,[cell_tool()]);r.update(job_id=job_id,author=record['author'],seed=record['seed'],model=model.id,arm=arm)
    if r['status']=='valid':
        value=r['object'].get('value')
        if type(value) is not int or value not in {0,1}:r['status']='invalid'
        else:r.update(correct=value==case['truth'],predicted_error=value==case['predicted_wrong'])
    write_json(folder/'result.json',r);return r

def baseline():
    m=prepare();cases=read_json(OUT/'cases.json');seal=read_json(OUT/'case_seal.json')
    if seal['cases_hash']!=digest(cases):raise ValueError('Cases changed')
    p=BudgetProvider()
    try:
        rows=parallel([(r,model,'baseline') for r in cases if r['status']=='valid' for model in MODELS],lambda j:actor_job(j,p,m,'baseline'),'Predict own first logical mistake')
        write_json(OUT/'baseline.json',rows);qualified=[]
        for model in MODELS:
            slots=[c for c in cases if c['author']==model.id]
            own=[r for r in rows if r['author']==model.id and r['model']==model.id]
            peer=[r for r in rows if r['author']==model.id and r['model']!=model.id]
            if len(slots)==2 and all(c['status']=='valid' for c in slots) and len(own)==2 and all(r.get('predicted_error') for r in own) and any(r.get('correct') for r in peer):qualified.append(model.id)
        gate={'at':utc_now(),'passed':bool(qualified),'qualifying_generators':qualified}
        write_json(OUT/'baseline_gate.json',gate);print(dumps({'rows':rows,'gate':gate}))
    finally:p.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['invent','baseline']);globals()[p.parse_args().stage]()
