"""Do models invent exact, model-specific counting illusions?"""
import argparse
from pathlib import Path
from .runner import request,parallel
from .runtime import ROOT,BudgetProvider,MODELS,usage
from .storage import digest,dumps,read_json,utc_now,write_json

OUT=ROOT/'runs/new-search/string-selftraps-1'

def compile_string(o):
    if not isinstance(o,dict) or set(o)!={'parts','character','predicted_wrong','p_wrong','diagnosis'}:raise ValueError('Wrong construction fields')
    if not isinstance(o['parts'],list) or not 2<=len(o['parts'])<=6:raise ValueError('2..6 parts required')
    text=''
    for part in o['parts']:
        if not isinstance(part,dict) or set(part)!={'text','repeat'} or not isinstance(part['text'],str) or not part['text'].isalpha() or not part['text'].isascii() or part['text']!=part['text'].lower():raise ValueError('Lowercase ASCII letters only')
        if type(part['repeat']) is not int or not 1<=part['repeat']<=5:raise ValueError('Repeat 1..5')
        text+=part['text']*part['repeat']
    if not 20<=len(text)<=80 or 'strawberry' in text:raise ValueError('20..80 fresh letters; exclude strawberry')
    char=o['character']
    if not isinstance(char,str) or len(char)!=1 or char not in 'abcdefghijklmnopqrstuvwxyz':raise ValueError('One lowercase ASCII target')
    true=text.count(char)
    if type(o['predicted_wrong']) is not int or not 0<=o['predicted_wrong']<=80 or o['predicted_wrong']==true:raise ValueError('Declared error must differ from true count')
    if len(o['p_wrong'])!=2 or any(type(p) not in {float,int} or not 0<=p<=1 for p in o['p_wrong']):raise ValueError('Two probabilities required')
    # Independent count by integer-byte equality, without str.count().
    if sum(b==ord(char) for b in text.encode('ascii'))!=true:raise RuntimeError('Count disagreement')
    return {'text':text,'character':char,'true':true,'wrong':o['predicted_wrong']}

def prepare():
    if (OUT/'manifest.json').exists():return read_json(OUT/'manifest.json')
    m={'at':utc_now(),'candidate':'self_specific_counting_illusion','budget_before':usage(),'tokens':6000,'seconds':120,
       'design':'Two authors x two targets, one construction each. String programs eliminate answer-construction errors. Both models count each valid string once in fresh contexts.',
       'gate':'At least one SELF-targeted author predicts the EXACT wrong numeric answer produced by its own actor, while peer actor is correct. Otherwise drop without expanding.',
       'novelty':'Exact personalized error anticipation, not generic character-counting difficulty. Needs fresh repetitions, new strings and nontrivial mechanism validation before any novelty claim.',
       'source':{f:(ROOT/'selftrap'/f).read_text() for f in ['stringtraps.py','runner.py']}}
    write_json(OUT/'manifest.json',m);return m

def inventor_tool():
    return {'type':'function','name':'invent_trap','description':'Construct a counting challenge with an exact model-relative error forecast.',
      'parameters':{'type':'object','required':['parts','character','predicted_wrong','p_wrong','diagnosis'],'additionalProperties':False,
       'properties':{'parts':{'type':'array','minItems':2,'maxItems':6,'items':{'type':'object','required':['text','repeat'],'additionalProperties':False,
        'properties':{'text':{'type':'string'},'repeat':{'type':'integer','minimum':1,'maximum':5}}}},
        'character':{'type':'string','minLength':1,'maxLength':1},'predicted_wrong':{'type':'integer','minimum':0,'maximum':80},
        'p_wrong':{'type':'array','minItems':2,'maxItems':2,'items':{'type':'number','minimum':0,'maximum':1}},'diagnosis':{'type':'string','maxLength':400}}}}

def invent():
    m=prepare();p=BudgetProvider()
    try:
        def job(j):
            author,target=j;case=author.id+'-targets-'+target.id;folder=OUT/'authors'/case
            if (folder/'result.json').exists():return read_json(folder/'result.json')
            prompt='''Invent a new character-counting challenge targeted at a SPECIFIC model's
predictable mistake. You write 2..6 parts, each lowercase ASCII text repeated
1..5 times. The compiler concatenates them exactly, without spaces. Total length
20..80 letters. Invent a fresh pattern; do not use strawberry or a quoted
instruction. Choose one letter to count. Predict an EXACT WRONG integer TARGET
will answer, while preferably the peer model answers correctly. You need not
supply the true count; the compiler calculates it independently.
Forecast P(exact predicted wrong integer) for [Qwen,Gemma], then give a short
mechanism diagnosis. Honest probabilities. A fresh actor sees ONLY the final
string and count request (no program, target identity, predictions or diagnosis).
It gets 6k output tokens/120s and a native count submission. No code execution.
This is personalized failure forecasting, not a creativity score. Call invent_trap.
'''+dumps({'author':author.id,'TARGET':target.id,'models':[x.id for x in MODELS]})
            r=request(folder,p,author,prompt,'invent',m,[inventor_tool()]);r.update(case=case,author=author.id,target=target.id)
            if r['status']=='valid':
                try:r['puzzle']=compile_string(r['object'])
                except (ValueError,KeyError,TypeError) as exc:r.update(status='invalid',reason=str(exc))
            write_json(folder/'result.json',r);return r
        rows=parallel([(a,t) for a in MODELS for t in MODELS],job,'Invent exact personal counting traps')
        write_json(OUT/'authors.json',rows);write_json(OUT/'seal.json',{'at':utc_now(),'authors_hash':digest(rows)})
        print(dumps({'valid':sum(r['status']=='valid' for r in rows),'slots':len(rows)}))
    finally:p.close()

def actors():
    m=prepare();authors=read_json(OUT/'authors.json');seal=read_json(OUT/'seal.json')
    if digest(authors)!=seal['authors_hash']:raise ValueError('Author freeze changed')
    p=BudgetProvider()
    try:
        def job(j):
            a,model=j;folder=OUT/'actors'/(a['case']+'-'+model.id)
            if (folder/'result.json').exists():return read_json(folder/'result.json')
            tool={'type':'function','name':'submit_count','description':'Submit the exact number of occurrences.',
                'parameters':{'type':'object','required':['count'],'additionalProperties':False,'properties':{'count':{'type':'integer','minimum':0,'maximum':80}}}}
            text='Count the exact number of occurrences of the requested character in the literal ASCII string. Every occurrence counts. Call submit_count.\n'+dumps({'string':a['puzzle']['text'],'character':a['puzzle']['character']})
            r=request(folder,p,model,text,'count',m,[tool]);r.update(case=a['case'],model=model.id,author=a['author'],target=a['target'])
            if r['status']=='valid':
                value=r['object'].get('count');r.update(correct=value==a['puzzle']['true'],exact_error=value==a['puzzle']['wrong'])
            write_json(folder/'result.json',r);return r
        rows=parallel([(a,model) for a in authors if a['status']=='valid' for model in MODELS],job,'Personal counting-error gate');write_json(OUT/'actors.json',rows)
        eligible=[]
        for a in authors:
            if a['status']!='valid' or a['author']!=a['target']:continue
            own=next(r for r in rows if r['case']==a['case'] and r['model']==a['author'])
            peer=next(r for r in rows if r['case']==a['case'] and r['model']!=a['author'])
            if own.get('exact_error') and peer.get('correct'):eligible.append(a['case'])
        gate={'at':utc_now(),'passed':bool(eligible),'qualifying_cases':eligible}
        write_json(OUT/'gate.json',gate);print(dumps({'results':rows,'gate':gate}))
    finally:p.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['invent','actors']);globals()[p.parse_args().stage]()
