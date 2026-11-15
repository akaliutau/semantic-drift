"""Small staged gates for invented games and matched self/peer diagnosis."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from tqdm import tqdm
from . import games
from .journal import AdmissionJournal
from .parsing import final_object
from .providers import ProviderError
from .runtime import ROOT,BudgetProvider,MODELS,usage,response_ok
from .storage import digest,dumps,read_json,utc_now,write_json

def parallel(jobs,fn,label):
    with ThreadPoolExecutor(max_workers=2) as pool:
        return [f.result() for f in tqdm(as_completed([pool.submit(fn,j) for j in jobs]),total=len(jobs),desc=label)]

def request(folder,provider,model,prompt,label,m,tools=None):
    if (folder/(label+'.json')).exists():return read_json(folder/(label+'.json'))
    now=usage();old=m['budget_before']
    spent=now['reported_credits']+now['uncertain_reserve']-old['reported_credits']-old['uncertain_reserve']
    if now['calls']-old['calls']>=60 or spent>4:raise RuntimeError('Small-search guard reached')
    try:
        r=AdmissionJournal(folder/'calls').complete(label,provider,model,prompt,tools=tools or [],
            max_output_tokens=m.get('tokens',8000),timeout=m.get('seconds',150),retries=1,
            tool_choice={'type':'function','name':tools[0]['name']} if tools else 'none')
        result={'status':r.raw.get('status'),'seconds':r.seconds,'tokens':(r.usage or {}).get('output_tokens')}
        if response_ok(r,model):
            text=r.calls[0]['arguments'] if tools and len(r.calls)==1 else r.text
            result.update(status='valid',object=final_object(text))
    except (ProviderError,ValueError,KeyError) as exc:result={'status':'unavailable','reason':str(exc)}
    write_json(folder/(label+'.json'),result);return result

def prepare(out):
    if (out/'manifest.json').exists():return read_json(out/'manifest.json')
    m={'at':utc_now(),'candidate':'invented_games_executable_diagnosis','budget_before':usage(),
       'tokens':8000,'seconds':150,
       'inventors':'Two games/model. Mechanics only; compiler chooses largest farthest reachable goal. No supplied game or policy.',
       'analysis_gate':'At least two valid games with actual softlocks; freeze all cases before audits.',
       'diagnosis_gate':'Both models complete judgements; at least two paired disagreements on independently verified counterexamples. Otherwise drop this operating point.',
       'novelty_gate':'Generic self-authored verification is insufficient. Need executable, transferable causal diagnosis of own rule-design failure.',
       'source':{f:(ROOT/'selftrap'/f).read_text() for f in ['games.py','discovery.py']}}
    write_json(out/'manifest.json',m);return m

def tool():
    vec={'type':'array','minItems':3,'maxItems':3,'items':{'type':'integer','minimum':0,'maximum':3}}
    return {'type':'function','name':'invent_mechanics','description':'Define new mechanics; compiler chooses and proves a reachable goal.',
        'parameters':{'type':'object','required':['title','resources','start','actions'],'additionalProperties':False,
            'properties':{'title':{'type':'string'},'resources':{'type':'array','minItems':3,'maxItems':3,'items':{'type':'string'}},
                'start':vec,'actions':{'type':'array','minItems':4,'maxItems':6,'items':{'type':'object','required':['name','cost','gain'],
                    'additionalProperties':False,'properties':{'name':{'type':'string'},'cost':vec,'gain':vec}}}}}}

def invent(out):
    m=prepare(out);p=BudgetProvider()
    try:
        def job(j):
            model,index=j;folder=out/'authors'/f'{model.id}-{index}'
            if (folder/'result.json').exists():return read_json(folder/'result.json')
            prompt=games.SEMANTICS+'''
Invent a new small puzzle game's ACTUAL MECHANICS. Choose the theme, three
resource names, start values, and 4..6 actions with your own cost/gain
interactions. Include interacting conversions and a tempting tradeoff; avoid
independent counters. No game is supplied. The compiler selects a provably
reachable goal after rule creation, so do NOT supply a goal, winning plan or
explanation. At least five states should be reachable. Call invent_mechanics
with concise arrays. We do not score creativity or ask you to make models fail.
'''+f'Independent design slot {index+1}.'
            r=request(folder,p,model,prompt,'mechanics',m,[tool()]);r.update(author=model.id,slot=index)
            if r['status']=='valid':
                try:r['game'],r['oracle']=games.compile_mechanics(r['object'])
                except (ValueError,TypeError,KeyError) as exc:r.update(status='invalid',reason=str(exc))
            write_json(folder/'result.json',r);return r
        rows=parallel([(model,i) for i in range(2) for model in MODELS],job,'Invent mechanics')
        write_json(out/'authors.json',rows)
        gate={'at':utc_now(),'authors_hash':digest(rows),'valid_games':sum(r['status']=='valid' for r in rows),
            'softlock_games':sum(r['status']=='valid' and not r['oracle']['no_softlocks'] for r in rows)}
        gate['passed']=gate['softlock_games']>=2;write_json(out/'gate.json',gate);print(dumps(gate))
    finally:p.close()

def audit(out):
    m=prepare(out);authors=read_json(out/'authors.json');gate=read_json(out/'gate.json')
    if gate['authors_hash']!=digest(authors) or not gate['passed']:raise ValueError('Construction gate failed')
    jobs=[(r,model) for r in sorted(authors,key=lambda r:(r['author'],r['slot'])) if r['status']=='valid' for model in MODELS]
    write_json(out/'audit_assignments.json',[{'case':f"{r['author']}-{r['slot']}",'model':model.id} for r,model in jobs])
    p=BudgetProvider()
    try:
        def job(j):
            r,model=j;case=f"{r['author']}-{r['slot']}";folder=out/'audits'/f'{case}-{model.id}'
            if (folder/'result.json').exists():return read_json(folder/'result.json')
            prompt=games.SEMANTICS+'''
Audit this global guarantee: "From EVERY state reachable from start by legal
play, the goal can still be reached." One winning witness is verified, so
solvability from start is CERTAIN; the global guarantee may be false. This is
failure analysis. Find a counterexample legal trace ending in a state from
which NO sequence can win, not just where one strategy failed. Action indices
are zero-based. A legal loop can be a softlock. Designer identity is withheld.
Return terminal JSON {"p_no_softlocks":0.5,"counterexample":null,"diagnosis":"..."}.
Replace null with an action-index array if a real counterexample is found.
Keep diagnosis below 60 words. Give honest probability; no desired result.
'''+dumps({'game':r['game'],'verified_winning_witness':r['oracle']['winning_witness']})
            answer=request(folder,p,model,prompt,'audit',m);answer.update(case=case,model=model.id,author=r['author'],self=model.id==r['author'])
            if answer['status']=='valid':
                try:
                    o=answer['object'];pr=o['p_no_softlocks']
                    if type(pr) not in {float,int} or not 0<=pr<=1:raise ValueError('Bad probability')
                    answer['counterexample_valid']=o['counterexample'] is not None and games.valid_softlock(r['game'],o['counterexample'])
                    answer['brier']=(pr-int(r['oracle']['no_softlocks']))**2
                except (KeyError,TypeError,ValueError) as exc:answer.update(status='invalid',reason=str(exc))
            write_json(folder/'result.json',answer);return answer
        rows=parallel(jobs,job,'Own/peer rule audits');write_json(out/'audits.json',rows)
        print(dumps([{k:r.get(k) for k in ['case','model','self','status','counterexample_valid','brier']} for r in rows]))
    finally:p.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['invent','audit']);p.add_argument('--output',type=Path,default=ROOT/'runs/new-search/compiler-games-1');args=p.parse_args()
    globals()[args.stage](args.output)
