"""A small prospective tool-choice diagnostic on previously failed puzzles."""
from pathlib import Path
from . import ordered_pilot as study
from .games import load_game
from .runtime import BudgetProvider,MODELS
from .storage import digest,dumps,read_json,write_json,utc_now

def run():
    out=study.BASE/'harness-probe';m={**study.prepare(),'max_calls':68}
    cases=[c for c in read_json(study.BASE/'mastermind/screen-cases.json') if c['kind']=='ambiguous']
    old=[r for r in read_json(study.BASE/'mastermind/screen-actors.json') if r['case'] in {c['id'] for c in cases}]
    if not (out/'design.json').exists():
        write_json(out/'design.json',{'at':utc_now(),'cases_hash':digest(cases),'selected_before_new_arms':True,
            'selection':'Two prior Mastermind ambiguity cases with instant incorrect Gemma native calls. These are diagnostic repeats, not independent confirmation.',
            'new_calls_max':8,'suite_call_guard_amended_from':60,'suite_call_guard_amended_to':68,
            'credit_guard_unchanged':4,'original_190_credit_ledger_unchanged':True,
            'arms':'Only tool_choice changes: forced submit_answer vs auto. Same prompts, parser, 12000-token budget and 180s. Two fresh forced Gemma repeats; auto trials on both models.',
            'source':Path(__file__).read_text(),'runner_source':(Path(__file__).parent/'runner.py').read_text()})
    props={'case':{'type':'string'},'target':{'type':'string'},'arm':{'type':'string','enum':['forced','auto']},
           'p_correct':{'type':'number'},'p_complete':{'type':'number'}}
    forecast_tool=study.tool('submit_harness_forecasts',{'rows':{'type':'array','items':{'type':'object','properties':props,'required':list(props),'additionalProperties':False}}},['rows'])
    evidence=[{'case':c['id'],'public':c['case']['analysis']['public'],'oracle_answer':c['case']['analysis']['truth']} for c in cases]
    prompt='''Predict BOTH target models on each case, including yourself. These exact
cases have prior observations below. We will make new calls changing only the
Responses API tool_choice setting: forced submit_answer vs auto. Prompts,
parsing, model IDs, 12000 output tokens and 180-second limit remain the same.
The task asks for UNKNOWN when multiple codes satisfy the clues. Accepted
output is native submit_answer, an explicit final submit_answer(answer=...) text
line, or a final JSON object with answer. No code execution tools exist.
For EVERY case x target x arm give p_correct and p_complete. The prior wrong
Gemma responses contained a native function call and just 20 output tokens
(2 reported reasoning tokens). Its other correct trials included a message
before the function call. This is an association; DO NOT assume causation.
Both forecasters get the same evidence. Forecast before new trials run.
'''+dumps({'rules':load_game('mastermind').RULES,'targets':[x.id for x in MODELS],'cases':evidence,'prior_trials':old})
    p=BudgetProvider(concurrency=3,per_model_concurrency=2)
    try:
        if (out/'forecasts.json').exists():forecasts=read_json(out/'forecasts.json')
        else:
            def forecast(model):
                r=study.ask(out/'forecasters'/model.id,p,model,prompt,'forecast',m,[forecast_tool]);r['forecaster']=model.id
                if r['status']=='valid':
                    try:
                        rows=r['object']['rows'];expected={(c['id'],t.id,a) for c in cases for t in MODELS for a in ['forced','auto']}
                        if len(rows)!=len(expected) or {(r['case'],r['target'],r['arm']) for r in rows}!=expected:raise ValueError('Missing forecast cells')
                        for r0 in rows:
                            study.probability(r0['p_correct']);study.probability(r0['p_complete'])
                    except (ValueError,TypeError,KeyError) as exc:r.update(status='invalid',reason=str(exc))
                return r
            forecasts=study.parallel(MODELS,forecast,'Forecast tool-choice effects');write_json(out/'forecasts.json',forecasts)
        def arm(mode):
            config={**m,'tool_choice':'auto' if mode=='auto' else {'type':'function','name':'submit_answer'}}
            return {'arm':mode,'rows':study.actors('mastermind',cases,p,config,'harness-'+mode,['baseline'],models=MODELS if mode=='auto' else (MODELS[1],))}
        results=study.parallel(['forced','auto'],arm,'Matched harness diagnostic')
        write_json(out/'result.json',{'at':utc_now(),'forecasts':forecasts,'arms':results,'label':'Exploratory paired diagnostic; previously selected puzzles; no population inference.'})
        print(dumps(results),flush=True)
    finally:p.close();print(dumps(study.concurrency_report(m)),flush=True)

if __name__=='__main__':run()
