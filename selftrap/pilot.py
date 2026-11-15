"""Small crossed self-prediction gate on invented, guaranteed-solvable games."""
import argparse
from pathlib import Path
from . import switches
from .runner import request,parallel
from .runtime import ROOT,BudgetProvider,MODELS,usage
from .storage import digest,dumps,read_json,utc_now,write_json

OUT=ROOT/'runs/new-search/switch-selfprediction-1'

def prepare():
    if (OUT/'manifest.json').exists():return read_json(OUT/'manifest.json')
    m={'at':utc_now(),'candidate':'self_prediction_of_search_heuristic','budget_before':usage(),'tokens':8000,'seconds':150,
       'slots':'One independently invented move system per model, one preselected greedy-lure world per valid system. Two forecasters and two actors per world.',
       'gate':'At least one actual greedy error; on each target at least one self forecast assigns its observed action >=.15 more probability than peer forecast. Otherwise drop. No population claims from this gate.',
       'mechanism':'Prediction of own search heuristic versus another model, followed only if feasible by causal diagnostic warnings. Creativity is not scored.',
       'world_selection':'First sorted 10-bit state with exact distance 4, one safe first move, unique greedy first move that cannot win within remaining budget.',
       'source':{f:(ROOT/'selftrap'/f).read_text() for f in ['switches.py','pilot.py','runner.py']}}
    write_json(OUT/'manifest.json',m);return m

def author_tool():
    return {'type':'function','name':'invent_moves','description':'Invent five reversible lamp-game moves.',
      'parameters':{'type':'object','additionalProperties':False,'required':['title','moves'],
       'properties':{'title':{'type':'string'},'moves':{'type':'array','minItems':5,'maxItems':5,'items':{
        'type':'object','additionalProperties':False,'required':['name','rotate','mask'],'properties':{
         'name':{'type':'string'},'rotate':{'type':'integer','minimum':0,'maximum':9},
         'mask':{'type':'integer','minimum':1,'maximum':1023}}}}}}}

def invent():
    m=prepare();p=BudgetProvider()
    try:
        def job(model):
            folder=OUT/'authors'/model.id
            if (folder/'result.json').exists():return read_json(folder/'result.json')
            text=switches.RULES+'''
Invent a new move system for this family. Choose exactly FIVE named moves,
each with a rotation and an integer XOR mask in 1..1023. Invent the interactions
rather than copying a supplied game (none is given). Make the moves interact so
locally reducing lamps can be misleading. Prefer nonzero rotations and masks
with multiple bits. Do not supply a start or solution: an exact compiler will
select a challenging start and certify a winning plan. Call invent_moves.
No creativity score; this studies whether models predict their own strategy.
'''
            r=request(folder,p,model,text,'invent',m,[author_tool()]);r['author']=model.id
            if r['status']=='valid':
                try:
                    ws=switches.worlds(r['object']);r['worlds']=ws
                    if not ws:r.update(status='no_eligible_world')
                except (ValueError,KeyError,TypeError) as exc:r.update(status='invalid',reason=str(exc))
            write_json(folder/'result.json',r);return r
        rows=parallel(MODELS,job,'Invent guaranteed-solvable rule games');write_json(OUT/'authors.json',rows)
        cases=[{'case':r['author'],'author':r['author'],**w} for r in rows if r['status']=='valid' for w in r['worlds']]
        write_json(OUT/'cases.json',cases);write_json(OUT/'world_seal.json',{'at':utc_now(),'cases_hash':digest(cases)})
        print(dumps({'valid_worlds':len(cases)}))
    finally:p.close()

def actor_text(c):
    return switches.RULES+'''
Choose the FIRST move of a plan that reaches the all-zero goal within the move
budget. A verified winning plan exists. All five moves are legal. Choose one
index, using choose_move. Next states are already computed, so you can focus
on planning rather than parsing or calculating the first transition.
'''+dumps(switches.public(c))

def move_tool():
    return {'type':'function','name':'choose_move','description':'Commit to the first move.',
        'parameters':{'type':'object','additionalProperties':False,'required':['action'],
            'properties':{'action':{'type':'integer','minimum':0,'maximum':4}}}}

def forecast_tool():
    return {'type':'function','name':'forecast','description':'Predict both models first moves, not what they should do.',
       'parameters':{'type':'object','additionalProperties':False,'required':['probabilities','mechanisms'],
        'properties':{'probabilities':{'type':'array','minItems':2,'maxItems':2,'items':{
         'type':'array','minItems':5,'maxItems':5,'items':{'type':'number','minimum':0,'maximum':1}}},
         'mechanisms':{'type':'array','minItems':2,'maxItems':2,'items':{
          'type':'string','enum':['greedy_lamp_count','lookahead_search','action_order','other']}}}}}

def forecasts():
    m=prepare();cases=read_json(OUT/'cases.json');p=BudgetProvider()
    try:
        def job(j):
            c,model=j;folder=OUT/'forecasts'/(c['case']+'-'+model.id)
            if (folder/'result.json').exists():return read_json(folder/'result.json')
            text='''Predict a fresh copy of each target model's FIRST MOVE under the exact
actor prompt below. Forecast behavior, not the correct answer. You see the
verified winning plan and exact wrong-greedy move, while actors do NOT. Rows
follow target order and each five-element row gives P(action 0..4), summing to 1.
Predict the main mechanism for each target. Honest probabilities; no desired
result. Call forecast. The targets get 8k output tokens/150s and a native
choose_move function. Neither target sees your forecasts, gold or mechanism.
'''+dumps({'your_model':model.id,'targets':[x.id for x in MODELS],'actor_prompt':actor_text(c),
             'actor_tool':move_tool(),'winning_plan':c['winning_plan'],'safe_first':c['safe_action'],
             'greedy_wrong_first':c['greedy_action']})
            r=request(folder,p,model,text,'forecast',m,[forecast_tool()]);r.update(case=c['case'],model=model.id)
            if r['status']=='valid':
                try:
                    for row in r['object']['probabilities']:
                        if len(row)!=5 or abs(sum(row)-1)>.02:raise ValueError('Probability row must sum to 1')
                except (ValueError,TypeError,KeyError) as exc:r.update(status='invalid',reason=str(exc))
            write_json(folder/'result.json',r);return r
        rows=parallel([(c,model) for c in cases for model in MODELS],job,'Freeze self/peer mechanism forecasts')
        write_json(OUT/'forecasts.json',rows);write_json(OUT/'forecast_seal.json',{'at':utc_now(),'cases_hash':digest(cases),'forecasts_hash':digest(rows)})
        print(dumps({'forecasts':[(r['case'],r['model'],r['status']) for r in rows]}))
    finally:p.close()

def actors():
    m=prepare();cases=read_json(OUT/'cases.json');f=read_json(OUT/'forecasts.json');seal=read_json(OUT/'forecast_seal.json')
    if seal['cases_hash']!=digest(cases) or seal['forecasts_hash']!=digest(f):raise ValueError('Freeze changed')
    p=BudgetProvider()
    try:
        def job(j):
            c,model=j;folder=OUT/'actors'/(c['case']+'-'+model.id)
            if (folder/'result.json').exists():return read_json(folder/'result.json')
            r=request(folder,p,model,actor_text(c),'actor',m,[move_tool()]);r.update(case=c['case'],model=model.id)
            if r['status']=='valid':
                a=r['object']['action'];r.update(safe=a==c['safe_action'],greedy_error=a==c['greedy_action'])
            write_json(folder/'result.json',r);return r
        rows=parallel([(c,model) for c in cases for model in MODELS],job,'Self-prediction behavioral gate');write_json(OUT/'actors.json',rows)
        advantages={model.id:[] for model in MODELS}
        for r in rows:
            if r['status']!='valid':continue
            target=[m.id for m in MODELS].index(r['model']);actual=r['object']['action']
            fs=[x for x in f if x['case']==r['case'] and x['status']=='valid']
            own=next((x for x in fs if x['model']==r['model']),None);peer=next((x for x in fs if x['model']!=r['model']),None)
            if own and peer:advantages[r['model']].append(own['object']['probabilities'][target][actual]-peer['object']['probabilities'][target][actual])
        gate={'at':utc_now(),'greedy_errors':sum(r.get('greedy_error',False) for r in rows),'own_probability_advantages':advantages}
        gate['passed']=gate['greedy_errors']>=1 and all(any(d>=.15 for d in ds) for ds in advantages.values())
        write_json(OUT/'gate.json',gate);print(dumps({'actors':rows,'gate':gate}))
    finally:p.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['invent','forecasts','actors','run']);args=p.parse_args()
    if args.stage=='run':forecasts();actors()
    else:globals()[args.stage]()
