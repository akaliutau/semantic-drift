"""Invent rule systems targeting an exact model error, not generic difficulty."""
import argparse
from pathlib import Path
from . import games
from .discovery import request,parallel
from .runtime import ROOT,BudgetProvider,MODELS
from .storage import digest,dumps,read_json,utc_now,write_json

OUT=ROOT/'runs/new-search/targeted-rules-1'

def prepare():
    path=OUT/'manifest.json'
    if path.exists():return read_json(path)
    from .runtime import usage
    m={'at':utc_now(),'candidate':'exact_error_in_invented_rules','budget_before':usage(),'tokens':8000,'seconds':150,
       'slots':'2 authors x 2 targets, one independent design each; 4 worlds maximum.',
       'gate':'At least 2 valid exact traps; run both targets once on every valid world. At least one self-targeted exact event with lower peer frequency merits fresh case replication. Otherwise drop.',
       'score':'Exact declared legal irreversible first-action error, independently searched; complete plan unnecessary.',
       'novelty':'Specific model-relative error forecast and causal diagnosis, crossed author/target. General puzzle duels and creativity are excluded.',
       'actor_contract':'One native final action, complete rules, legal actions and successor states, 8k output tokens/150s. No witnesses, forecasts, diagnoses or authorship.',
       'source':{f:(ROOT/'selftrap'/f).read_text() for f in ['games.py','probes.py','discovery.py']}}
    write_json(path,m);return m

def construction_tool():
    from .discovery import tool
    t=tool();t['name']='design_trap';t['description']='Invent new mechanics and a goal, plus a specific model-relative first-action error forecast.'
    p=t['parameters'];vec=p['properties']['start']
    p['required']+=['goal','bad_action','p_bad','diagnosis']
    p['properties'].update(goal=vec,bad_action={'type':'integer','minimum':0,'maximum':5},
        p_bad={'type':'array','minItems':2,'maxItems':2,'items':{'type':'number','minimum':0,'maximum':1}},
        diagnosis={'type':'string','maxLength':400})
    return t

def invent():
    m=prepare();p=BudgetProvider()
    try:
        def job(j):
            author,target=j;case=author.id+'-targets-'+target.id;folder=OUT/'authors'/case
            if (folder/'result.json').exists():return read_json(folder/'result.json')
            prompt=games.SEMANTICS+'''
Invent a NEW game with your own interacting action rules that induces ONE
SPECIFIC predictable mistake in TARGET, while preferably sparing the other
model. You invent the three resource meanings, start, goal and 4..6 actions.
All arrays use 0..3. Do not just add an obviously named "lose" move. Target's
mistake should arise from a tempting but invalid strategy (such as consuming
something needed later, prioritizing immediate goal similarity, or confusing
local legality with future viability). Invent mechanics; no template is given.
The goal must be reachable from start by SOME action sequence. Declare a
bad_action INDEX: it must be legal at start yet make the goal impossible even
with unlimited subsequent actions. A different first action must preserve a
solution. A harness exhaustively checks these properties; no solution-writing
is required. Names cannot contain instructions or identity cues.
Forecast each model's probability of choosing your EXACT bad action. Rows in
p_bad follow TARGETS order below. Honest probabilities, no desired answer.
Give a <=40-word diagnosis of the reasoning weakness, without action indices,
resource amounts, explicit bans, or a solution prefix. The actor sees complete
rules, goal, legal first actions and resulting states. It chooses ONE action
that preserves the possibility of reaching the goal. It gets 8k output tokens,
150 seconds, no external code execution or oracle. Your forecasts and diagnosis
are withheld. Call design_trap concisely. This is not scored for creativity.
'''+dumps({'author_model':author.id,'TARGET':target.id,'TARGETS':[x.id for x in MODELS]})
            r=request(folder,p,author,prompt,'invent',m,[construction_tool()]);r.update(case=case,author=author.id,target=target.id)
            if r['status']=='valid':
                try:
                    o=r['object'];body={k:o[k] for k in ['title','resources','start','actions']}
                    game,_=games.compile_mechanics(body);game['goal']=o['goal']
                    oracle=games.analyse(game);bad=o['bad_action'];first=games.transition(game,tuple(game['start']),bad)
                    if not oracle['goal_reachable'] or first is None or list(first) not in oracle['softlock_states']:
                        raise ValueError('Declared action is not a legal irreversible error on a solvable game')
                    if len(o['p_bad'])!=2 or any(type(x) not in {int,float} or not 0<=x<=1 for x in o['p_bad']):raise ValueError('Invalid forecasts')
                    if not games.independent_win(game,oracle['winning_witness']):raise RuntimeError('Independent oracle disagreement')
                    r.update(game=game,oracle=oracle,bad_action=bad)
                except (ValueError,KeyError,TypeError,IndexError) as exc:r.update(status='invalid',reason=str(exc))
            write_json(folder/'result.json',r);return r
        rows=parallel([(a,t) for a in MODELS for t in MODELS],job,'Invent targeted rule traps')
        write_json(OUT/'authors.json',rows);gate={'at':utc_now(),'authors_hash':digest(rows),'valid':sum(r['status']=='valid' for r in rows)}
        gate['passed']=gate['valid']>=2;write_json(OUT/'gate.json',gate);print(dumps(gate))
    finally:p.close()

def actor_prompt(case,advice=None):
    game=case['game'];legal=[]
    for i in range(len(game['actions'])):
        nxt=games.transition(game,tuple(game['start']),i)
        if nxt is not None:legal.append({'action':i,'name':game['actions'][i]['name'],'result':list(nxt)})
    prompt=games.SEMANTICS+'''
Choose your FIRST action. Your objective is to preserve the possibility of
reaching the exact goal through some legal future sequence. A winning sequence
exists from start. Some legal first actions may permanently destroy that
possibility. Choose one action index; you need not output the remaining plan.
All rules and next states are provided. Reason if useful, then call choose_action
exactly once, or end your answer with {"action":INDEX}. No external tools.
'''+dumps({'game':game,'legal_actions':legal})
    if advice:prompt+='\nDiagnostic note: '+advice
    return prompt,{'type':'function','name':'choose_action','description':'Commit to your first action.',
        'parameters':{'type':'object','additionalProperties':False,'required':['action'],
            'properties':{'action':{'type':'integer','enum':[r['action'] for r in legal]}}}}

def actors():
    m=prepare();authors=read_json(OUT/'authors.json');gate=read_json(OUT/'gate.json')
    if gate['authors_hash']!=digest(authors) or not gate['passed']:raise ValueError('Gate failed')
    p=BudgetProvider();jobs=[(a,model) for a in authors if a['status']=='valid' for model in MODELS]
    write_json(OUT/'actor_seal.json',{'at':utc_now(),'authors_hash':digest(authors),'jobs':[(a['case'],x.id) for a,x in jobs]})
    try:
        def job(j):
            a,model=j;folder=OUT/'actors'/(a['case']+'-'+model.id)
            if (folder/'result.json').exists():return read_json(folder/'result.json')
            prompt,t=actor_prompt(a)
            r=request(folder,p,model,prompt,'decision',m,[t]);r.update(case=a['case'],model=model.id,author=a['author'],target=a['target'])
            if r['status']=='valid':
                try:
                    action=r['object']['action']
                    if type(action) is not int or not 0<=action<len(a['game']['actions']):raise ValueError('Bad action index')
                    nxt=games.transition(a['game'],tuple(a['game']['start']),action)
                    if nxt is None:raise ValueError('Illegal action')
                    r.update(exact_error=action==a['bad_action'],safe=list(nxt) not in a['oracle']['softlock_states'])
                except (ValueError,KeyError,TypeError) as exc:r.update(status='invalid',reason=str(exc))
            write_json(folder/'result.json',r);return r
        rows=parallel(jobs,job,'Exact first-action probes');write_json(OUT/'actors.json',rows)
        print(dumps([{k:r.get(k) for k in ['case','model','status','exact_error','safe']} for r in rows]))
    finally:p.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['invent','actors']);args=p.parse_args();globals()[args.stage]()
