"""Learn falsifiable generators of exact personal errors, then test unseen n."""
import argparse
from pathlib import Path
from .runner import request,parallel
from .runtime import ROOT,BudgetProvider,MODELS,usage
from .storage import digest,dumps,read_json,utc_now,write_json

OUT=ROOT/'runs/new-search/executable-selfdiagnosis-1'

def compile_law(o,n):
    if set(o)!={'prefix','atom','suffix','character','scale','offset','p_exact','mechanism'}:raise ValueError('Wrong law fields')
    for key in ['prefix','atom','suffix']:
        t=o[key]
        if not isinstance(t,str) or not t.isascii() or any(c not in 'abcdefghijklmnopqrstuvwxyz' for c in t):raise ValueError('ASCII lowercase only')
    text=o['prefix']+o['atom']*n+o['suffix'];char=o['character']
    if len(char)!=1 or char not in 'abcdefghijklmnopqrstuvwxyz' or not 20<=len(text)<=80:raise ValueError('Bad target or length')
    if type(o['scale']) is not int or type(o['offset']) is not int:raise ValueError('Integer coefficients required')
    wrong=o['scale']*n+o['offset'];true=sum(c==char for c in text)
    if not 0<=wrong<=80 or wrong==true:raise ValueError('Law must predict a wrong count, not the answer')
    return {'n':n,'text':text,'character':char,'true':true,'predicted':wrong}

def prepare():
    if (OUT/'manifest.json').exists():return read_json(OUT/'manifest.json')
    source=ROOT/'runs/new-search/string-selftraps-1';authors=read_json(source/'authors.json');actors=read_json(source/'actors.json')
    observations=[]
    for a in authors:
        if a['status']!='valid':continue
        observations.append({'puzzle':a['puzzle'],'actual_counts':{r['model']:r['object'].get('count') for r in actors if r['case']==a['case'] and r['status']=='valid'}})
    m={'at':utc_now(),'candidate':'executable_personal_error_laws','budget_before':usage(),'tokens':8000,'seconds':150,
        'observations':observations,'heldout_n':[3,5],
        'slots':'Both authors target both models, one generator each; two unseen parameter values, both target actors. Maximum 16 actor calls.',
        'gate':'At least one self-authored law predicts exact wrong outputs on BOTH new n, with peer correct on at least one. Otherwise drop. Original observations never count as held-out evidence.',
        'novelty':'Constructive executable laws of personal error, not generic adversarial instances or verbal self-critique. Needs fresh n and new pattern family replication.',
        'source':{f:(ROOT/'selftrap'/f).read_text() for f in ['laws.py','runner.py']}}
    write_json(OUT/'manifest.json',m);return m

def tool():
    return {'type':'function','name':'propose_error_law','description':'Propose a parametric input generator and exact wrong-output law.',
      'parameters':{'type':'object','additionalProperties':False,'required':['prefix','atom','suffix','character','scale','offset','p_exact','mechanism'],
       'properties':{'prefix':{'type':'string'},'atom':{'type':'string'},'suffix':{'type':'string'},'character':{'type':'string'},
        'scale':{'type':'integer','minimum':-20,'maximum':20},'offset':{'type':'integer','minimum':-80,'maximum':80},
        'p_exact':{'type':'array','minItems':2,'maxItems':2,'items':{'type':'number','minimum':0,'maximum':1}},
        'mechanism':{'type':'string','maxLength':600}}}}

def invent():
    m=prepare();p=BudgetProvider()
    try:
        def job(j):
            author,target=j;case=author.id+'-targets-'+target.id;folder=OUT/'authors'/case
            if (folder/'result.json').exists():return read_json(folder/'result.json')
            text='''You are an experimental scientist investigating model-specific counting
errors. You see a small completed experiment with BOTH models' actual outputs.
Discover a mechanism, then encode a FALSIFIABLE law of TARGET's future error.
Create a NEW parametric family: string(n)=prefix + atom repeated n times + suffix.
Use lowercase ASCII letters only. For EVERY n in 3..7 the string must have
20..80 letters. Choose one target character and a wrong-output formula
predicted_count(n)=scale*n+offset, with integer scale/offset. It must differ from
the true count for every n in 3..7. You need not supply the true answer.
Aim to predict TARGET's EXACT wrong number across fresh n while the peer is
correct. The target sees only the expanded literal string and count request;
no n, recipe, formula, feedback or mechanism. 6k tokens/120s, native count tool.
Propose a fresh atom, not a copy of a prior full string. Explain the proposed
mechanism in <=60 words and give P(exact law match) for [Qwen,Gemma]. We will
freeze the generator before choosing/evaluating the held-out n. Do not promise
certainty; the old data might not support a transferable law. Call propose_error_law.
'''+dumps({'your_model':author.id,'TARGET':target.id,'observations':m['observations']})
            r=request(folder,p,author,text,'law',m,[tool()]);r.update(case=case,author=author.id,target=target.id)
            if r['status']=='valid':
                try:
                    for n in range(3,8):compile_law(r['object'],n)
                except (ValueError,TypeError,KeyError) as exc:r.update(status='invalid',reason=str(exc))
            write_json(folder/'result.json',r);return r
        rows=parallel([(a,t) for a in MODELS for t in MODELS],job,'Discover executable error laws')
        write_json(OUT/'authors.json',rows);write_json(OUT/'law_seal.json',{'at':utc_now(),'authors_hash':digest(rows)})
        print(dumps({'valid_laws':sum(r['status']=='valid' for r in rows)}))
    finally:p.close()

def actors():
    m=prepare();authors=read_json(OUT/'authors.json');seal=read_json(OUT/'law_seal.json')
    if digest(authors)!=seal['authors_hash']:raise ValueError('Law freeze changed')
    p=BudgetProvider();am={**m,'tokens':6000,'seconds':120}
    try:
        def job(j):
            a,n,model=j;folder=OUT/'actors'/f"{a['case']}-{n}-{model.id}"
            if (folder/'result.json').exists():return read_json(folder/'result.json')
            puzzle=compile_law(a['object'],n)
            t={'type':'function','name':'submit_count','description':'Submit the exact occurrence count.',
               'parameters':{'type':'object','additionalProperties':False,'required':['count'],'properties':{'count':{'type':'integer','minimum':0,'maximum':80}}}}
            text='Count the exact number of occurrences of the requested character in the literal ASCII string. Every occurrence counts. Call submit_count.\n'+dumps({'string':puzzle['text'],'character':puzzle['character']})
            r=request(folder,p,model,text,'count',am,[t]);r.update(case=a['case'],n=n,model=model.id,author=a['author'],target=a['target'],puzzle=puzzle)
            if r['status']=='valid':
                val=r['object'].get('count');r.update(exact_error=val==puzzle['predicted'],correct=val==puzzle['true'])
            write_json(folder/'result.json',r);return r
        rows=parallel([(a,n,model) for a in authors if a['status']=='valid' for n in m['heldout_n'] for model in MODELS],job,'Test laws on unseen parameter values')
        write_json(OUT/'actors.json',rows);qualified=[]
        for a in authors:
            if a['status']!='valid' or a['author']!=a['target']:continue
            own=[r for r in rows if r['case']==a['case'] and r['model']==a['target']]
            peer=[r for r in rows if r['case']==a['case'] and r['model']!=a['target']]
            if len(own)==2 and all(r.get('exact_error') for r in own) and any(r.get('correct') for r in peer):qualified.append(a['case'])
        gate={'at':utc_now(),'passed':bool(qualified),'qualifying_laws':qualified}
        write_json(OUT/'gate.json',gate);print(dumps({'results':[{k:r.get(k) for k in ['case','model','n','status','object','correct','exact_error']} for r in rows],'gate':gate}))
    finally:p.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['invent','actors']);globals()[p.parse_args().stage]()
