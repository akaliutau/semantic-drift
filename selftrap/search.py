"""Small, frozen-stage gates for novel self-diagnosis candidates."""

import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm
from .journal import AdmissionJournal
from .parsing import final_object
from .providers import ProviderError
from .runtime import ROOT,BudgetProvider,MODELS,usage,response_ok
from .storage import digest,dumps,read_json,utc_now,write_json
from . import worlds


def collect(jobs,fn,description):
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending=[pool.submit(fn,j) for j in jobs]
        return [f.result() for f in tqdm(as_completed(pending),total=len(jobs),desc=description)]


def manifest(out):
    p=out/'manifest.json'
    if p.exists():return read_json(p)
    data={'at':utc_now(),'candidate':'authored_rule_blindspots','models':[asdict(m) for m in MODELS],
        'budget_before':usage(),'max_new_calls':60,'credit_guard':5,
        'sample':'Two independently invented games per model; four total; no outcome-based replacement.',
        'gate':'At least two valid solvable games with model-claimed no-softlock p>=.7 but actual softlocks; then test matched own/peer counterexamples. Otherwise drop this operating point.',
        'tokens':6000,'seconds':100,
        'novelty_question':'Can an author identify counterexamples to its own global rule guarantee better than a peer, and can their respective repairs actually restore it without making goals unreachable?',
        'controls':'Oracle exhaustive (<=1296 states); solution existence; authored rules vs claims; same evidence for own/peer; no evaluation of invention quality.',
        'source':{f:(ROOT/'selftrap'/f).read_text() for f in ['worlds.py','search.py']}}
    write_json(p,data);return data


def request(out,provider,model,prompt,label,m):
    before=m['budget_before'];now=usage()
    if now['calls']-before['calls']>=m['max_new_calls']:raise RuntimeError('Candidate call guard reached')
    spent=now['reported_credits']+now['uncertain_reserve']-before['reported_credits']-before['uncertain_reserve']
    if spent+1>m['credit_guard']:raise RuntimeError('Candidate credit guard reached')
    try:
        r=AdmissionJournal(out/'calls').complete(label,provider,model,prompt,tools=[],
            max_output_tokens=m['tokens'],timeout=m['seconds'],tool_choice='none',retries=1)
        if not response_ok(r,model):return {'status':r.raw.get('status','unknown'),'response_model':r.raw.get('model')}
        return {'status':'completed','object':final_object(r.text),'tokens':(r.usage or {}).get('output_tokens'),'seconds':r.seconds}
    except (ProviderError,ValueError) as exc:return {'status':'unavailable','reason':str(exc)}


def invent(out):
    m=manifest(out);provider=BudgetProvider()
    jobs=[(model,i) for i in range(2) for model in MODELS]
    try:
        def job(j):
            model,i=j;folder=out/'authors'/f'{model.id}-{i}';saved=folder/'result.json'
            if saved.exists():return read_json(saved)
            theme=['interlocking resources, tools and one-way conversions','switches, transport, locks and consumable energy'][i]
            prompt=worlds.RULES+'\nDesign theme: '+theme+'. Invent your own actual mechanics. Do not deliberately insert a softlock. Return {"game": YOUR_GAME_OBJECT}.'
            r=request(folder,provider,model,prompt,'invent',m);r.update(author=model.id,slot=i)
            if r['status']=='completed':
                try:
                    game=r['object']['game'];r['oracle']=worlds.analyse(game);r['game']=game
                    r['eligible']=r['oracle']['goal_reachable'] and r['oracle']['witness_valid'] and len(r['oracle']['shortest_win'])>=4
                except (ValueError,KeyError,TypeError) as exc:r.update(status='invalid',reason=str(exc),eligible=False)
            write_json(saved,r);return r
        results=collect(jobs,job,'Invented-rule feasibility');write_json(out/'authors.json',results)
        eligible=[r for r in results if r.get('eligible')]
        promising=[r for r in eligible if not r['oracle']['no_softlocks'] and r['game']['p_no_softlocks']>=.7]
        seal={'at':utc_now(),'authors_hash':digest(results),'eligible':len(eligible),'high_confidence_false_claims':len(promising),'gate_passed':len(promising)>=2}
        write_json(out/'invention_gate.json',seal);print(dumps(seal));return results
    finally:provider.close()


def audit(out):
    m=manifest(out);authors=read_json(out/'authors.json');gate=read_json(out/'invention_gate.json')
    if digest(authors)!=gate['authors_hash']:raise ValueError('Frozen authors changed')
    if not gate['gate_passed']:raise ValueError('Candidate failed its feasibility gate; do not run more calls')
    provider=BudgetProvider()
    jobs=[(r,model) for r in authors if r.get('eligible') for model in MODELS]
    try:
        def job(j):
            r,model=j;case=f"{r['author']}-{r['slot']}";folder=out/'audits'/f'{case}-{model.id}'
            if (folder/'result.json').exists():return read_json(folder/'result.json')
            prompt='''Audit this finite game for softlocks. You have exactly the same rules
as its designer but no oracle output. A fresh auditor can be the designer's
same model family or another; provenance is not shown. Find a legally reachable
non-goal state from which the goal is IMPOSSIBLE even with unlimited legal
moves. A stalled strategy is insufficient. Return concise terminal JSON with
p_no_softlocks, counterexample (action-name array, or null if none), and diagnosis
(at most 60 words explaining the mechanism). You may challenge any claim; the
oracle, not confidence or prose, determines whether the counterexample is real.
Use action guards and simultaneous deltas exactly as given. No tools available.
'''+dumps({'semantics':worlds.RULES.split('Return a game object')[0],
                'game':worlds.public(r['game'])})
            answer=request(folder,provider,model,prompt,'audit',m);answer.update(case=case,model=model.id,author=r['author'],self=model.id==r['author'])
            if answer['status']=='completed':
                try:
                    obj=answer['object'];trace=obj['counterexample']
                    answer['valid_counterexample']=isinstance(trace,list) and worlds.counterexample_valid(r['game'],trace)
                    p=obj['p_no_softlocks']
                    if type(p) not in {int,float} or not 0<=p<=1:raise ValueError('Invalid probability')
                    answer['brier']=(p-int(r['oracle']['no_softlocks']))**2
                except (ValueError,KeyError,TypeError) as exc:answer.update(status='invalid',reason=str(exc))
            write_json(folder/'result.json',answer);return answer
        rows=collect(jobs,job,'Own/peer rule diagnosis');write_json(out/'audits.json',rows)
        print(dumps([{k:r.get(k) for k in ['case','model','self','status','valid_counterexample','brier']} for r in rows]))
    finally:provider.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['invent','audit']);p.add_argument('--output',type=Path,default=ROOT/'runs/new-search/rule-blindspots-1');args=p.parse_args()
    globals()[args.stage](args.output)
