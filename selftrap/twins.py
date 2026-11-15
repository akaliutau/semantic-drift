"""Can self-diagnosis respect a mechanically proved limit on causal knowledge?"""
import argparse
from copy import deepcopy
from itertools import combinations,product
from pathlib import Path
from . import games
from .runner import request,parallel
from .runtime import ROOT,BudgetProvider,MODELS,usage
from .storage import digest,dumps,read_json,utc_now,write_json

OUT=ROOT/'runs/new-search/diagnostic-twins-1'

def response(game,state,action):
    result=games.transition(game,tuple(state),action)
    return {'accepted':result is not None,'after':list(result) if result is not None else state}

def make_pair(source):
    game=source['game']
    for action,a in enumerate(game['actions']):
        eligible=[i for i in range(3) if a['cost'][i]<3 and a['gain'][i]<3]
        for i,j in combinations(eligible,2):
            A=deepcopy(game);B=deepcopy(game)
            A['actions'][action]['cost'][i]+=1;B['actions'][action]['cost'][j]+=1
            state=a['cost'][:]
            original=response(game,state,action)
            if not original['accepted'] or state==game['goal']:continue
            beforeA=response(A,state,action);beforeB=response(B,state,action)
            if beforeA!=beforeB:continue
            probes=[];diagnostic=None;uninformative=None
            for s in product(range(4),repeat=3):
                if list(s)==game['goal']:continue
                for k in range(len(game['actions'])):
                    x=response(A,list(s),k);y=response(B,list(s),k)
                    row={'state':list(s),'action':k,'A':x,'B':y}
                    if x!=y and diagnostic is None:diagnostic=row
                    if x==y and uninformative is None and k!=action:uninformative=row
                    if diagnostic is not None and uninformative is not None:break
                if diagnostic is not None and uninformative is not None:break
            if diagnostic is None or uninformative is None:continue
            tests=[uninformative,diagnostic]
            # Counterbalance location deterministically across source models.
            if source['author'].startswith('qwen'):tests.reverse()
            return {'source':source['author'],'game':game,'fault_action':action,'A_resource':i,'B_resource':j,
                'initial_observation':{'state':state,'action':action,'result':beforeA},
                'hypotheses':{'A':f"Action {action} secretly requires one extra unit of resource {i}.",
                              'B':f"Action {action} secretly requires one extra unit of resource {j}."},
                'tests':tests,'diagnostic_test':next(n for n,t in enumerate(tests) if t['A']!=t['B']),
                'proof':{'same_initial_observation':beforeA==beforeB,'different_diagnostic_result':diagnostic['A']!=diagnostic['B'],
                    'hidden_fault_prior':[.5,.5],'reason':'Fault choice is independent, made after mechanics creation; initial public record is identical in both engines.'}}
    raise ValueError('No fault twins found in this game')

def prepare():
    if (OUT/'manifest.json').exists():return read_json(OUT/'manifest.json')
    source=read_json(ROOT/'runs/new-search/compiler-games-1/authors.json')
    selected=[next(r for r in source if r['status']=='valid' and r['author']==m.id and r['slot']==0) for m in MODELS]
    cases=[make_pair(r) for r in selected]
    m={'at':utc_now(),'candidate':'self_diagnosis_identifiability','budget_before':usage(),'tokens':4000,'seconds':90,
        'sample':'Two model-invented worlds, both auditors, ambiguous and identifying evidence: eight calls.',
        'gate':'Any own-world unsupported certainty |p(A)-.5|>=.3, plus valid identifiable control, merits fresh confirmation. Otherwise drop.',
        'scope':'Failure diagnosis, not game invention or solving. Structural Bayesian posterior and probe value are mechanically known.',
        'source':{f:(ROOT/'selftrap'/f).read_text() for f in ['games.py','twins.py','runner.py']}}
    write_json(OUT/'manifest.json',m);write_json(OUT/'cases.json',cases)
    write_json(OUT/'seal.json',{'at':utc_now(),'cases_hash':digest(cases),'manifest_hash':digest(m)})
    return m

def tool():
    return {'type':'function','name':'diagnose','description':'Report root-cause confidence and choose the next diagnostic test or repair.',
        'parameters':{'type':'object','additionalProperties':False,'required':['p_A','next_test','repair'],
            'properties':{'p_A':{'type':'number','minimum':0,'maximum':1},
                'next_test':{'type':'integer','enum':[-1,0,1]},
                'repair':{'type':'string','enum':['fix_A','fix_B','need_evidence']}}}}

def run():
    m=prepare();cases=read_json(OUT/'cases.json');seal=read_json(OUT/'seal.json')
    if seal['cases_hash']!=digest(cases):raise ValueError('Cases changed')
    jobs=[(c,model,information) for c in cases for model in MODELS for information in ['ambiguous','identified']]
    p=BudgetProvider()
    try:
        def job(j):
            c,model,information=j;case=c['source']+'-'+model.id+'-'+information;folder=OUT/'actors'/case
            if (folder/'result.json').exists():return read_json(folder/'result.json')
            tests=[{k:t[k] for k in ['state','action']} for t in c['tests']]
            evidence=[c['initial_observation']]
            if information=='identified':
                d=c['tests'][c['diagnostic_test']];evidence.append({'state':d['state'],'action':d['action'],'result':d['A']})
            prompt=games.SEMANTICS+'''
You are diagnosing a failure in a new game. A model invented the complete
intended mechanics below. A deployment bug changed EXACTLY ONE action's cost;
other rules, transitions and readouts remain faithful. The only candidate
causes are A and B. They are equally likely BEFORE observations, chosen by an
independent fair coin AFTER game creation. You have the complete public record;
there is no private log or hidden memory available to you. In test mode the
engine can reset to any listed state. State readouts and results are exact.
Report the posterior probability of A based on the observed record. Then
choose which ONE of the two available tests would best identify the actual
cause; use -1 if no further test is needed. Choose fix_A/fix_B only if the
existing evidence justifies committing a repair, otherwise need_evidence.
Call diagnose. No outcome preference; unsupported confident diagnosis is bad.
'''+dumps({'your_model':model.id,'mechanics_author':c['source'],'intended_game':c['game'],
                'candidate_causes':c['hypotheses'],'observations':evidence,'available_tests':tests})
            r=request(folder,p,model,prompt,'diagnose',m,[tool()]);r.update(case=case,model=model.id,author=c['source'],self=model.id==c['source'],information=information)
            if r['status']=='valid':
                o=r['object'];target=.5 if information=='ambiguous' else 1.
                r.update(posterior_error=abs(o['p_A']-target),
                    correct_probe=o['next_test']==(c['diagnostic_test'] if information=='ambiguous' else -1),
                    correct_repair=o['repair']==('need_evidence' if information=='ambiguous' else 'fix_A'))
            write_json(folder/'result.json',r);return r
        rows=parallel(jobs,job,'Causal self-diagnosis gate');write_json(OUT/'results.json',rows)
        print(dumps([{k:r.get(k) for k in ['model','author','self','information','status','object','posterior_error','correct_probe']} for r in rows]))
    finally:p.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run']);globals()[p.parse_args().stage]()
