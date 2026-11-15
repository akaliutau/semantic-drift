"""A fresh test of experimentally learned personal error laws and proposed cures."""
import argparse
from pathlib import Path
from .runner import request,parallel
from .runtime import ROOT,BudgetProvider,MODELS,usage
from .storage import digest,dumps,read_json,utc_now,write_json

OUT=ROOT/'runs/new-search/active-selfdiagnosis-1'

def puzzle(recipe,n):
    text=recipe['prefix']+recipe['atom']*n+recipe['suffix'];char=recipe['character']
    true=text.count(char)
    if sum(b==ord(char) for b in text.encode('ascii'))!=true:raise RuntimeError('Independent count disagreement')
    return text,true

def render(recipe,n,repair):
    text,true=puzzle(recipe,n)
    if repair=='atom_boundaries':text=' '.join([recipe['prefix']]+[recipe['atom']]*n+[recipe['suffix']])
    elif repair=='characters':text=' '.join(text)
    elif repair=='reverse':text=text[::-1]
    elif repair!='none':raise ValueError('Unknown repair')
    if text.count(recipe['character'])!=true:raise RuntimeError('Repair changed target count')
    return text,true

def predict(law,n):
    if law['mode']=='unresolved':return None
    if law['mode']=='piecewise' and n>=law['threshold']:return law['scale_high']*n+law['offset_high']
    return law['scale']*n+law['offset']

def prepare():
    if (OUT/'manifest.json').exists():return read_json(OUT/'manifest.json')
    source=ROOT/'runs/new-search/executable-selfdiagnosis-1';authors=read_json(source/'authors.json');actors=read_json(source/'actors.json')
    selected=next(a for a in authors if a['case']=='qwen3.5-397b-targets-gemma-4-31b-it')
    observations=[{k:r[k] for k in ['n','model','object','puzzle','status']} for r in actors if r['case']==selected['case']]
    m={'at':utc_now(),'candidate':'experimentally_learned_personal_error_law','budget_before':usage(),'tokens':6000,'seconds':120,
        'recipe':selected['object'],'observations':observations,'heldout_n':[4,6],
        'selection':'Selected after observing Gemma=true-1 twice and different Qwen errors. Discovery does not count as confirmation.',
        'writers':'Both models see identical observations and recipe, independently predict both models and propose one repair for themselves.',
        'gate':'Gemma SELF-written raw law predicts exact incorrect counts at BOTH new n, Qwen raw is correct at least once, and Gemma self-proposed repair yields correct counts at BOTH n. Otherwise drop; no replacement.',
        'interpretation':'Experimentally acquired model-specific diagnosis, not evidence of privileged innate introspection. Peer forecasts are retained and compared.',
        'source':{f:(ROOT/'selftrap'/f).read_text() for f in ['active_diagnosis.py','runner.py']}}
    write_json(OUT/'manifest.json',m);return m

def law_tool():
    integer={'type':'integer','minimum':-100,'maximum':100}
    return {'type':'function','name':'write_diagnosis','description':'Write testable raw-answer laws and one proposed personal repair.',
      'parameters':{'type':'object','additionalProperties':False,'required':['laws','self_repair','p_repair_correct','mechanism'],
       'properties':{'laws':{'type':'array','minItems':2,'maxItems':2,'items':{
        'type':'object','additionalProperties':False,'required':['mode','scale','offset','threshold','scale_high','offset_high','p_exact'],
        'properties':{'mode':{'type':'string','enum':['linear','piecewise','unresolved']},'scale':integer,'offset':integer,
           'threshold':{'type':'integer','minimum':0,'maximum':20},'scale_high':integer,'offset_high':integer,
           'p_exact':{'type':'number','minimum':0,'maximum':1}}}},
        'self_repair':{'type':'string','enum':['atom_boundaries','characters','reverse','none']},
        'p_repair_correct':{'type':'number','minimum':0,'maximum':1},'mechanism':{'type':'string','maxLength':600}}}}

def diagnoses():
    m=prepare();p=BudgetProvider()
    try:
        def job(model):
            folder=OUT/'diagnoses'/model.id
            if (folder/'result.json').exists():return read_json(folder/'result.json')
            text='''Study the measured behavior of both models on one parametric string family.
Do not invent a different family. Learn an EXECUTABLE theory of their actual
counts, not of the correct answer. string(n)=prefix+atom*n+suffix, concatenated
without spaces. You see only n=3 and n=5. Predict fresh n using either a linear
law scale*n+offset, a piecewise law switching to scale_high*n+offset_high at
n>=threshold, or unresolved if the evidence is insufficient. Use integer
coefficients. The two law rows follow [Qwen,Gemma]. Ignore high coefficients
for linear mode; use 0 placeholders for unresolved. Report honest p_exact.
Then propose one representation change to repair YOUR OWN future counts:
atom_boundaries (spaces between prefix, each atom and suffix), characters
(spaces between every character), reverse, or none. All preserve the target
letter count. Give P(your repair yields a correct count) and a short mechanism.
The fresh actors receive only the expanded literal string and original count
request, no n, recipe, theory or feedback. Same 6k-token/120s native count tool.
No desired outcome. A fit to two points is not by itself a causal explanation.
Call write_diagnosis.
'''+dumps({'your_model':model.id,'models':[x.id for x in MODELS],'recipe':m['recipe'],'observations':m['observations']})
            r=request(folder,p,model,text,'diagnosis',m,[law_tool()]);r['model']=model.id
            write_json(folder/'result.json',r);return r
        rows=parallel(MODELS,job,'Discover a personal law and cure');write_json(OUT/'diagnoses.json',rows)
        write_json(OUT/'diagnosis_seal.json',{'at':utc_now(),'diagnoses_hash':digest(rows),'manifest_hash':digest(m)})
        print(dumps(rows))
    finally:p.close()

def actors():
    m=prepare();diagnoses=read_json(OUT/'diagnoses.json');seal=read_json(OUT/'diagnosis_seal.json')
    if seal['diagnoses_hash']!=digest(diagnoses) or seal['manifest_hash']!=digest(m):raise ValueError('Freeze changed')
    by_model={d['model']:d for d in diagnoses};p=BudgetProvider()
    try:
        def job(j):
            model,n,arm=j;d=by_model[model.id];repair=d['object']['self_repair'] if arm=='self_repair' else 'none'
            text,true=render(m['recipe'],n,repair);folder=OUT/'actors'/f'{model.id}-{n}-{arm}'
            if (folder/'result.json').exists():return read_json(folder/'result.json')
            t={'type':'function','name':'submit_count','description':'Submit the exact occurrence count.',
               'parameters':{'type':'object','additionalProperties':False,'required':['count'],'properties':{'count':{'type':'integer','minimum':0,'maximum':80}}}}
            prompt='Count the exact number of occurrences of the requested character in the literal ASCII string. Every occurrence counts. Call submit_count.\n'+dumps({'string':text,'character':m['recipe']['character']})
            r=request(folder,p,model,prompt,'count',m,[t]);r.update(model=model.id,n=n,arm=arm,repair=repair,true=true)
            if r['status']=='valid':r.update(correct=r['object']['count']==true)
            write_json(folder/'result.json',r);return r
        jobs=[(model,n,arm) for n in m['heldout_n'] for model in MODELS if by_model[model.id]['status']=='valid' for arm in ['raw','self_repair']]
        rows=parallel(jobs,job,'Fresh law-and-repair confirmation');write_json(OUT/'actors.json',rows)
        match={}
        for d in diagnoses:
            if d['status']!='valid':continue
            for ix,target in enumerate(MODELS):
                rs=[r for r in rows if r['model']==target.id and r['arm']=='raw']
                match[d['model']+'->'+target.id]=[r['status']=='valid' and predict(d['object']['laws'][ix],r['n'])==r['object']['count'] for r in sorted(rs,key=lambda r:r['n'])]
        g=next(x for x in MODELS if x.id.startswith('gemma'));q=next(x for x in MODELS if x.id.startswith('qwen'))
        gr=[r for r in rows if r['model']==g.id and r['arm']=='raw'];gf=[r for r in rows if r['model']==g.id and r['arm']=='self_repair'];qr=[r for r in rows if r['model']==q.id and r['arm']=='raw']
        passed=(match.get(g.id+'->'+g.id)==[True,True] and len(gr)==2 and all(not r.get('correct',True) for r in gr)
                and any(r.get('correct') for r in qr) and len(gf)==2 and all(r.get('correct') for r in gf))
        gate={'at':utc_now(),'passed':passed,'law_matches':match}
        write_json(OUT/'gate.json',gate);print(dumps({'results':rows,'gate':gate}))
    finally:p.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['diagnoses','actors']);globals()[p.parse_args().stage]()
