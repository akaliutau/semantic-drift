"""Auditable edit lattices and capped black-box scoring for executable masks.

Legal membership and protected tokens are mechanically checked. General English
semantic equivalence is NOT certified by a synonym dictionary or this module.
"""

import ast
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from .programs import validate_source


WORD_GROUPS=(('among','amongst'),('toward','towards'),('amid','amidst'),
    ('afterward','afterwards'),('color','colour'),('colors','colours'),
    ('behavior','behaviour'),('behaviors','behaviours'),('center','centre'),
    ('centers','centres'),('analyze','analyse'),('analyzed','analysed'),
    ('organize','organise'),('organized','organised'),('organization','organisation'),
    ('realize','realise'),('realized','realised'),('recognize','recognise'),
    ('recognized','recognised'),('gray','grey'),('labor','labour'),
    ('neighbor','neighbour'),('neighbors','neighbours'),('neighboring','neighbouring'))
GROUP={word:group for group in WORD_GROUPS for word in group}
PROTECTED=re.compile(r"\b(?:not|no|never|only|all|some|often|may|might|could|would|must|can|will|seems?|suggests?|proves?)\b|\d+(?:\.\d+)?",re.I)


def lattice(text):
    pattern=re.compile(r"\b(?:"+'|'.join(sorted(GROUP,key=len,reverse=True))+r")\b|[’'“”\"]|(?<=[.!?]) +(?=[A-Z])",re.I)
    sites=[];last=0;pieces=[]
    for match in pattern.finditer(text):
        old=match.group();low=old.lower()
        if low in GROUP:
            if old!=low and low not in ('among','amongst','toward','towards','amid','amidst','afterward','afterwards'):
                continue  # Do not respell a potential proper name such as Gray.
            options=list(GROUP[low])
            if old[:1].isupper():options=[s.capitalize() for s in options]
        elif old in ("'",'’'):options=["'",'’']
        elif old in ('“','”','"'):
            # Pair direction is preserved; straight quotes acquire a direction
            # from their occurrence parity, not from an attacker-supplied word.
            position=len([s for s in sites if s['kind']=='quote'])
            options=['"',old if old!='"' else ('“' if position%2==0 else '”')]
        else:options=[old,'\n','\n\n']
        options=list(dict.fromkeys([old]+options))
        if len(options)<2:continue
        pieces.append(text[last:match.start()]);last=match.end()
        sites.append({'id':len(sites),'original':old,'options':options,
                      'kind':'quote' if old in ('“','”','"') else 'word' if low in GROUP else 'apostrophe' if old in ("'",'’') else 'whitespace'})
    pieces.append(text[last:])
    return {'original':text,'pieces':pieces,'sites':sites}


def render(problem,plan):
    if not isinstance(plan,list) or len(plan)!=len(problem['sites']):raise ValueError('One option index per site required')
    parts=[]
    for i,(site,option) in enumerate(zip(problem['sites'],plan)):
        if type(option) is not int or not 0<=option<len(site['options']):raise ValueError('Illegal database option')
        parts.extend((problem['pieces'][i],site['options'][option]))
    parts.append(problem['pieces'][-1]);text=''.join(parts)
    if PROTECTED.findall(text)!=PROTECTED.findall(problem['original']):raise ValueError('Protected token changed')
    return text


def validate_policy(code):
    if len(code)>6000:raise ValueError('Policy exceeds 6000 characters')
    tree=ast.parse(code)
    class ScopeCheck(ast.NodeTransformer):
        def visit_Nonlocal(self,node):
            if any(n.startswith('__') for n in node.names):raise ValueError('Private nonlocal excluded')
            return ast.copy_location(ast.Pass(),node)
    # Nonlocal only binds names inside generated nested functions. Validate
    # every other node with the established sandbox, then execute ORIGINAL code.
    validate_source(ast.unparse(ScopeCheck().visit(tree)))
    return code


WORKER=r'''
import json,sys,random,time,resource
resource.setrlimit(resource.RLIMIT_AS,(2048*1024*1024,2048*1024*1024))
resource.setrlimit(resource.RLIMIT_CPU,(30,30))
sys.path.insert(0,sys.argv[1])
from selftrap.programs import BUILTINS
from selftrap.swap_harness import validate_policy,render
import joblib
from threadpoolctl import threadpool_limits
threadpool_limits(limits=1)
req=json.load(sys.stdin);artifact=joblib.load(req['artifact'])
classes=list(artifact['classes']);target=classes.index(req['target']);all_results=[]
def raw(text):
 x=artifact['vectorizer'].transform([text]);v=artifact['classifier'].decision_function(x)[0]
 margin=float(v[target]-max(float(s) for i,s in enumerate(v) if i!=target))
 return margin,str(classes[int(v.argmax())])
for task in req['tasks']:
 problem=task['problem'];budget=task['budget'];rng=random.Random(task['seed']);random.seed(task['seed']);trace=[]
 def score(plan):
  if len(trace)>=budget:raise ValueError('Classifier query cap exceeded')
  margin,label=raw(render(problem,plan));trace.append({'plan':list(plan),'margin':margin,'label':label})
  return margin
 def db(i):return list(problem['sites'][i]['options'])
 def randint(n):return rng.randrange(n)
 def safe_import(name,*args,**kwargs):
  if name!='random':raise ValueError('Only random may be imported')
  return random
 builtins={k:getattr(__import__('builtins'),k) for k in BUILTINS};builtins['__import__']=safe_import
 g={'__builtins__':builtins,'score':score,'db':db,'randint':randint,'random':random}
 started=time.perf_counter()
 try:
  exec(compile(validate_policy(req['code']),'<edit-policy>','exec'),g)
  public={'sites':problem['sites'],'budget':budget}
  plan=g['generate'](public);text=render(problem,plan);margin,label=raw(text)
  base,base_label=raw(problem['original'])
  all_results.append({'id':task['id'],'ok':True,'plan':plan,'text':text,'margin':margin,'top1':label,
   'original_margin':base,'original_top1':base_label,'queries':len(trace),'seconds':time.perf_counter()-started,
   'edited_sites':sum(v!=0 for v in plan),'sites':len(plan),'trace':trace})
 except Exception as exc:all_results.append({'id':task['id'],'ok':False,'error':type(exc).__name__+': '+str(exc),'trace':trace})
print(json.dumps(all_results,allow_nan=False))
'''


def execute(code,tasks,artifact,target):
    validate_policy(code)
    root=Path(__file__).resolve().parent.parent
    request={'code':code,'tasks':tasks,'artifact':str(Path(artifact).resolve()),'target':target}
    try:
        p=subprocess.run([sys.executable,'-I','-c',WORKER,str(root)],input=json.dumps(request),
            capture_output=True,text=True,timeout=45,cwd='/tmp',
            env={'PATH':os.defpath,'PYTHONHASHSEED':'0','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
    except subprocess.TimeoutExpired:raise ValueError('Policy worker exceeded 45 seconds') from None
    if p.returncode!=0 or len(p.stdout)>2_000_000:raise ValueError('Worker failed resource/output limits: '+p.stderr[-300:])
    return json.loads(p.stdout)


RANDOM='''def generate(problem):
    best = [0 for s in problem['sites']]
    best_score = score(best)
    for j in range(problem['budget'] - 1):
        plan = [randint(len(s['options'])) for s in problem['sites']]
        value = score(plan)
        if value > best_score:
            best, best_score = plan, value
    return best
'''

GREEDY='''def generate(problem):
    best = [0 for s in problem['sites']]
    best_score = score(best)
    used = 1
    for j in range(problem['budget'] - 1):
        if not best:
            break
        i = j % len(best)
        option = (j // len(best) + 1) % len(problem['sites'][i]['options'])
        plan = best.copy()
        plan[i] = option
        value = score(plan)
        if value > best_score:
            best, best_score = plan, value
    return best
'''
