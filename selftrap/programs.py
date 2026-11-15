"""Execute pure Python puzzle generators with restricted syntax and hard limits."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys

ALLOWED=(ast.Module,ast.FunctionDef,ast.arguments,ast.arg,ast.Return,ast.Assign,ast.AugAssign,
    ast.For,ast.If,ast.IfExp,ast.Expr,ast.BoolOp,ast.BinOp,ast.UnaryOp,ast.Compare,ast.Subscript,
    ast.Slice,ast.List,ast.Dict,ast.Set,ast.Tuple,ast.ListComp,ast.SetComp,ast.DictComp,
    ast.GeneratorExp,ast.comprehension,ast.Call,ast.Name,ast.Load,ast.Store,ast.Constant,
    ast.keyword,ast.Attribute,ast.Lambda,ast.And,ast.Or,ast.Not,ast.Add,ast.Sub,ast.Mult,
    ast.Div,ast.FloorDiv,ast.Mod,ast.USub,ast.UAdd,ast.Eq,ast.NotEq,ast.Lt,ast.LtE,
    ast.Gt,ast.GtE,ast.In,ast.NotIn,ast.Is,ast.IsNot,ast.Break,ast.Continue,ast.Pass,ast.Import,ast.alias,ast.Try,ast.ExceptHandler,ast.Raise)
ATTRIBUTES={'Random','randrange','randint','choice','choices','sample','shuffle','random',
    'append','extend','copy','reverse','count','index','get','items','keys','values','join'}
BUILTINS=('range','len','min','max','sum','abs','int','str','bool','list','tuple','dict','set',
    'enumerate','zip','sorted','all','any','reversed','Exception','ValueError','RuntimeError')

def validate_source(code):
    if not isinstance(code,str) or len(code)>18000:raise ValueError('Generator code too long')
    tree=ast.parse(code)
    if not any(isinstance(n,ast.FunctionDef) and n.name=='generate' for n in tree.body):raise ValueError('Define generate(seed)')
    for n in ast.walk(tree):
        if not isinstance(n,ALLOWED):raise ValueError('Unsupported generator syntax: '+type(n).__name__)
        if isinstance(n,ast.Import) and any(a.name!='random' for a in n.names):raise ValueError('Only import random is allowed')
        if isinstance(n,ast.Name) and n.id.startswith('__'):raise ValueError('Private names excluded')
        if isinstance(n,ast.Attribute) and n.attr not in ATTRIBUTES:raise ValueError('Unsupported attribute: '+n.attr)
        if isinstance(n,ast.FunctionDef) and (n.decorator_list or n.name.startswith('__')):raise ValueError('No decorators/private functions')
    return code

WORKER=r'''
import json,random,resource,sys
resource.setrlimit(resource.RLIMIT_AS,(256*1024*1024,256*1024*1024))
resource.setrlimit(resource.RLIMIT_CPU,(3,3))
sys.path.insert(0,sys.argv[1])
from selftrap.nonogram import inspect_bitmap
from selftrap.programs import BUILTINS,validate_source
request=json.load(sys.stdin);calls=0
def inspect(bitmap):
 global calls
 calls+=1
 if calls>8:raise ValueError('At most eight inspect calls per seed')
 return inspect_bitmap(bitmap)
def safe_import(name,*args,**kwargs):
 if name!='random':raise ValueError('Only random may be imported')
 return random
builtins={k:__builtins__[k] if isinstance(__builtins__,dict) else getattr(__builtins__,k) for k in BUILTINS}
builtins['__import__']=safe_import
g={'__builtins__':builtins,'random':random,'inspect':inspect}
try:
 exec(compile(validate_source(request['code']),'<generator>','exec'),g)
 value=g['generate'](request['seed'])
 print(json.dumps({'ok':True,'value':value,'inspect_calls':calls},allow_nan=False))
except Exception as exc:print(json.dumps({'ok':False,'error':type(exc).__name__+': '+str(exc)}))
'''

def run_generator(code,seed):
    validate_source(code);root=Path(__file__).resolve().parent.parent
    try:
        process=subprocess.run([sys.executable,'-I','-c',WORKER,str(root)],
            input=json.dumps({'code':code,'seed':seed}),text=True,capture_output=True,
            timeout=5,cwd='/tmp',env={'PATH':os.defpath,'PYTHONHASHSEED':'0'})
    except subprocess.TimeoutExpired:raise ValueError('Generator exceeded five-second wall cap') from None
    if process.returncode!=0 or len(process.stdout)>50000:raise ValueError('Generator failed CPU/memory/output cap')
    result=json.loads(process.stdout)
    if not result['ok']:raise ValueError(result['error'])
    return result
