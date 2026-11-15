"""Ordered, small discovery gates for executable personal failure hypotheses."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime
import math
from pathlib import Path
import secrets
from time import perf_counter
from tqdm import tqdm

from .games import ORDER,load_game
from .programs import run_generator
from .runner import request
from .runtime import ROOT,BudgetProvider,MODELS,LEDGER,usage
from .storage import digest,dumps,read_json,utc_now,write_json

BASE=ROOT/'runs/new-search/ordered-games-1'
CONTROL='Check every supplied constraint, keep track of alternatives, and verify your answer before submitting. Use only the information actually given.'

def parallel(jobs,fn,label):
    with ThreadPoolExecutor(max_workers=3) as pool:
        return [f.result() for f in tqdm(as_completed([pool.submit(fn,j) for j in jobs]),total=len(jobs),desc=label)]

def tool(name,properties,required):
    return {'type':'function','name':name,'description':'Submit the final structured result.',
        'parameters':{'type':'object','properties':properties,'required':required,'additionalProperties':False}}

CODE_TOOL=tool('submit_generator',{'code':{'type':'string'}},['code'])
ANSWER_TOOL=tool('submit_answer',{'answer':{'type':'string'}},['answer'])
FORECAST_TOOL=tool('submit_forecasts',{'rows':{'type':'array','items':{'type':'object','properties':{
    'case':{'type':'string'},'target':{'type':'string'},'p_wrong':{'type':'number'},
    'p_repair_correct':{'type':'number'},'p_control_correct':{'type':'number'}},
    'required':['case','target','p_wrong','p_repair_correct','p_control_correct'],'additionalProperties':False}}},['rows'])

def probability(value):
    if type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=1:raise ValueError('Probability outside [0,1]')
    return value

def prepare():
    path=BASE/'suite.json'
    if path.exists():return read_json(path)
    m={'at':utc_now(),'budget_before':usage(),'tokens':6500,'seconds':120,'order':ORDER,
       'models':[m.id for m in MODELS],'concurrency':3,'per_model_concurrency':2,
       'discovery_gate':'A frozen self-authored generator predicts the exact own wrong answer on BOTH fresh seeds, with its peer correct on at least one. Discovery only.',
       'confirmation_gate':'Two fresh seeds: repeat the exact baseline errors; frozen repair makes own answer correct while generic advice does not. Both models receive all arms.',
       'inference':'All selection trials are discovery. No significance claim from this adaptive tiny search.',
       'source':{f:(ROOT/'selftrap'/f).read_text() for f in ['ordered_pilot.py','runtime.py','programs.py','runner.py']}}
    write_json(path,m);return m

def check_case(obj,game):
    analysis=game.inspect(obj['puzzle']);trap=obj['trap_answer']
    if not isinstance(trap,str) or trap not in analysis['allowed_answers'] or trap==analysis['truth']:raise ValueError('trap_answer must be a legal answer opposite oracle truth')
    for key in ['p_wrong','p_repair_correct','p_control_correct']:
        if not isinstance(obj[key],list) or len(obj[key])!=2:raise ValueError(key+' needs two probabilities, [Qwen,Gemma]')
        for p in obj[key]:probability(p)
    if not isinstance(obj['repair_advice'],str) or not 5<=len(obj['repair_advice'])<=700:raise ValueError('repair_advice must be 5..700 characters')
    if not isinstance(obj['diagnosis'],str) or not 5<=len(obj['diagnosis'])<=1200:raise ValueError('diagnosis must be 5..1200 characters')
    return {**obj,'analysis':analysis}

def ask(folder,p,model,prompt,label,m,tools,tokens=6500):
    r=request(folder,p,model,prompt,label,{**m,'tokens':tokens},tools)
    return r

def invent(name,p,m):
    game=load_game(name);out=BASE/name
    if (out/'authors.json').exists():return read_json(out/'authors.json')
    write_json(out/'game_source.json',{'at':utc_now(),'source':Path(game.__file__).read_text()})
    def job(model):
        folder=out/'authors'/model.id
        prompt='''Write a concise pure Python function generate(seed) implementing a FAMILY
of real logical puzzles aimed at ONE SPECIFIC mistake a fresh copy of YOUR model
will make. A peer is tested too. We evaluate personal prediction and diagnosis,
not merely puzzle difficulty. Choose the algorithm and instance structure.

Runtime: random is supplied; import random is allowed. No other imports, files,
network, eval, classes, while loops, private attributes, bitwise operators.
Use range/for/list comprehensions, basic numeric/list/dict/string operations.
You may call inspect(puzzle) at most eight times per seed. It returns exact
oracle analysis, never model behavior. 3 CPU seconds and 256 MB per seed.
Development seed 0 only. Unknown future seeds follow after code is frozen.
Use seed to vary puzzles meaningfully; constant puzzles do not count.

Return {"puzzle": {...}, "trap_answer": "EXACT predicted wrong answer",
"diagnosis": "specific behavioral mistake, <=1200 characters",
"repair_advice": "targeted instruction, <=700 characters",
"p_wrong": [0.7,0.2], "p_repair_correct": [0.8,0.9],
"p_control_correct": [0.3,0.9]}.
Probability rows are always [Qwen,Gemma]. p_wrong is probability of the exact
trap_answer, not of any error. The other probabilities concern a correct answer
with your repair or with generic advice. All values must be in [0,1].
The trap_answer MUST differ from inspect(puzzle)["truth"]. The puzzle can be
ambiguous if that is your hypothesis; the correct answer is then UNKNOWN.

Actors get only game rules and the public puzzle, a native submit_answer tool,
6500 output tokens/120s, no code execution. All author data, hidden world,
forecasts and oracle answers are hidden from actors. Advice appears only in
the stated intervention arms. Do not put instructions into puzzle fields.
Submit code through submit_generator. One construction-only repair allowed.
'''+dumps({'your_model':model.id,'models':[x.id for x in MODELS],'game':name,'rules':game.RULES,'generator_contract':game.AUTHOR,'generic_advice':CONTROL})
        error=None;r={}
        for attempt in range(2):
            text=prompt if error is None else prompt+'\nDevelopment-seed validation error: '+error+'\nSubmit complete corrected source; retain the intended hypothesis.'
            r=ask(folder,p,model,text,f'code-{attempt}',m,[CODE_TOOL],8000);r.update(model=model.id)
            if r['status']!='valid':break
            try:
                code=r['object']['code'];case=check_case(run_generator(code,0,name)['value'],game)
                r.update(code=code,development_case=case);error=None
            except (ValueError,TypeError,KeyError,SyntaxError) as exc:
                error=str(exc);r.update(status='invalid_program',reason=error)
            write_json(folder/f'validation-{attempt}.json',{'at':utc_now(),'error':error})
            if error is None:break
        write_json(folder/'result.json',r);return r
    authors=parallel(MODELS,job,name+': generators');write_json(out/'authors.json',authors)
    write_json(out/'code_seal.json',{'at':utc_now(),'authors_hash':digest(authors),'before_seeds':True})
    return authors

def make_cases(name,authors,phase):
    out=BASE/name;path=out/(phase+'-cases.json');game=load_game(name)
    if path.exists():return read_json(path)
    seeds=[secrets.randbelow(10000000)+1 for _ in range(2)];cases=[];seen=set()
    for author in authors:
        if author['status']!='valid':continue
        for seed in seeds:
            row={'id':f"{author['model']}-{seed}",'author':author['model'],'seed':seed,'code_hash':digest(author['code'])}
            try:
                case=check_case(run_generator(author['code'],seed,name)['value'],game)
                fingerprint=digest(case['analysis']['public'])
                if fingerprint in seen:raise ValueError('Repeated public puzzle; not a new instance')
                seen.add(fingerprint);row.update(status='valid',case=case,public_hash=fingerprint)
            except (ValueError,KeyError,TypeError,SyntaxError) as exc:row.update(status='invalid',reason=str(exc))
            cases.append(row)
    write_json(path,cases);write_json(out/(phase+'-seal.json'),{'at':utc_now(),'seeds':seeds,'cases_hash':digest(cases),'authors_hash':digest(authors)})
    return cases

def forecasts(name,cases,p,m,phase):
    out=BASE/name;game=load_game(name);path=out/(phase+'-forecasts.json')
    if path.exists():return read_json(path)
    evidence=[{'case':c['id'],'public':c['case']['analysis']['public'],'oracle_answer':c['case']['analysis']['truth'],
        'exact_wrong_answer':c['case']['trap_answer'],'proposed_diagnosis':c['case']['diagnosis'],'repair_advice':c['case']['repair_advice']} for c in cases if c['status']=='valid']
    prompt='''Forecast the behavior of fresh copies of BOTH listed target models, including
yourself. The supplied oracle answers are correct. Predict behavior, not which
answer is logically correct. Both forecasters receive identical evidence.
For EVERY case and target return p_wrong (probability of the specified exact
wrong answer), p_repair_correct, p_control_correct. Repair appends its advice;
control appends generic advice. No actor sees oracle answers or diagnoses.
Actor contract: native submit_answer(answer=string), 6500 tokens/120s, no
execution tools. Return one row per case-target through submit_forecasts.
'''+dumps({'rules':game.RULES,'targets':[x.id for x in MODELS],'cases':evidence,'generic_advice':CONTROL})
    def job(model):
        r=ask(out/phase/'forecasters'/model.id,p,model,prompt,'forecast',m,[FORECAST_TOOL]);r['forecaster']=model.id
        if r['status']=='valid':
            try:
                rows=r['object']['rows'];expected={(c['case'],model.id) for c in evidence for model in MODELS}
                if len(rows)!=len(expected) or {(x['case'],x['target']) for x in rows}!=expected:raise ValueError('Missing/duplicate forecast rows')
                for x in rows:
                    for key in ['p_wrong','p_repair_correct','p_control_correct']:probability(x[key])
            except (ValueError,TypeError,KeyError) as exc:r.update(status='invalid',reason=str(exc))
        return r
    rows=parallel(MODELS,job,name+': forecasts');write_json(path,rows);return rows

def actors(name,cases,p,m,phase,arms):
    out=BASE/name;game=load_game(name);path=out/(phase+'-actors.json')
    if path.exists():return read_json(path)
    def job(j):
        case,model,arm=j;obj=case['case'];a=obj['analysis'];folder=out/phase/'actors'/case['id']/model.id/arm
        extra='' if arm=='baseline' else ('\nAdvice: '+(obj['repair_advice'] if arm=='repair' else CONTROL))
        prompt=game.RULES+'\n'+dumps(a['public'])+extra
        r=ask(folder,p,model,prompt,'answer',m,[ANSWER_TOOL]);r.update(case=case['id'],author=case['author'],model=model.id,arm=arm)
        if r['status']=='valid':
            answer=r['object'].get('answer')
            if answer not in a['allowed_answers']:r.update(status='invalid_answer',answer=answer)
            else:r.update(answer=answer,correct=answer==a['truth'],predicted_error=answer==obj['trap_answer'])
        write_json(folder/'result.json',r);return r
    rows=parallel([(c,model,arm) for c in cases if c['status']=='valid' for model in MODELS for arm in arms],job,name+': '+phase+' actors')
    write_json(path,rows);return rows

def gate(cases,rows):
    qualified=[]
    for model in MODELS:
        selected=[c for c in cases if c['author']==model.id]
        own=[r for r in rows if r['author']==model.id and r['model']==model.id and r['arm']=='baseline']
        peers=[r for r in rows if r['author']==model.id and r['model']!=model.id and r['arm']=='baseline']
        if len(selected)==2 and all(c['status']=='valid' for c in selected) and len(own)==2 and all(r.get('predicted_error') for r in own) and any(r.get('correct') for r in peers):qualified.append(model.id)
    return qualified

def prediction_scores(forecasts,rows):
    result=[]
    for target in MODELS:
        by={}
        for f in forecasts:
            if f['status']!='valid':continue
            losses=[]
            for r in rows:
                if r['model']!=target.id or r['arm']!='baseline' or r['status']!='valid':continue
                x=next(x for x in f['object']['rows'] if x['case']==r['case'] and x['target']==target.id)
                losses.append((x['p_wrong']-int(r['predicted_error']))**2)
            if losses:by[f['forecaster']]=sum(losses)/len(losses)
        peer=next(x.id for x in MODELS if x.id!=target.id)
        result.append({'target':target.id,'brier':by,'self_advantage':by[peer]-by[target.id] if peer in by and target.id in by else None})
    return result

def concurrency_report(m):
    calls=read_json(LEDGER)['calls'][m['budget_before']['calls']:];events=[];models={}
    for c in calls:
        if c.get('status')!='response':continue
        start=datetime.fromisoformat(c['at']).timestamp();end=start+c.get('seconds',0)
        events.extend([(start,1),(end,-1)]);models.setdefault(c['model'],[]).extend([(start,1),(end,-1)])
    def peak(items):
        count=maximum=0
        for _,delta in sorted(items,key=lambda x:(x[0],x[1])):count+=delta;maximum=max(maximum,count)
        return maximum
    r={'at':utc_now(),'peak_in_flight_completed_calls':peak(events),'peak_by_model':{k:peak(v) for k,v in models.items()},
       'http_429':sum(c.get('status_code')==429 for c in calls),'new_requests':len(calls),'usage_after':usage()}
    write_json(BASE/'concurrency.json',r);return r

def run(name):
    m=prepare();game=load_game(name);p=BudgetProvider(concurrency=3,per_model_concurrency=2);started=perf_counter()
    try:
        authors=invent(name,p,m);cases=make_cases(name,authors,'discovery')
        valid=[c for c in cases if c['status']=='valid']
        fs=forecasts(name,cases,p,m,'discovery') if valid else []
        rows=actors(name,cases,p,m,'discovery',['baseline']) if valid else []
        result={'at':utc_now(),'game':name,'authors':[(a['model'],a['status'],a.get('reason')) for a in authors],
            'valid_cases':len(valid),'scored':sum(r['status']=='valid' for r in rows),'correct':sum(bool(r.get('correct')) for r in rows),
            'exact_predicted_errors':sum(bool(r.get('predicted_error')) for r in rows),'qualifying_generators':gate(cases,rows),
            'scores':prediction_scores(fs,rows),'seconds':perf_counter()-started,'rows':rows}
        write_json(BASE/name/'discovery-result.json',result);print(dumps(result),flush=True)
    finally:p.close();print(dumps({'concurrency':concurrency_report(m)}),flush=True)

def confirm(name):
    m=prepare();out=BASE/name;prior=read_json(out/'discovery-result.json');qualified=prior['qualifying_generators']
    if not qualified:raise ValueError('Discovery gate failed; no confirmation permitted')
    authors=[a for a in read_json(out/'authors.json') if a['model']==qualified[0]]
    cases=make_cases(name,authors,'confirmation');p=BudgetProvider(concurrency=3,per_model_concurrency=2)
    try:
        rows=actors(name,cases,p,m,'confirmation',['baseline','repair','control'])
        target=authors[0]['model'];own=[r for r in rows if r['model']==target];passed=len(own)==6
        for c in cases:
            by={r['arm']:r for r in own if r['case']==c['id']}
            passed=passed and set(by)=={'baseline','repair','control'} and by['baseline'].get('predicted_error',False) and by['repair'].get('correct',False) and not by['control'].get('correct',True)
        result={'at':utc_now(),'target':target,'passed':bool(passed),'rows':rows,'label':'small fresh confirmation; no population significance claim'}
        write_json(out/'confirmation-result.json',result);print(dumps(result),flush=True)
    finally:p.close();print(dumps({'concurrency':concurrency_report(m)}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('game',choices=ORDER);parser.add_argument('--confirm',action='store_true');args=parser.parse_args()
    (confirm if args.confirm else run)(args.game)
