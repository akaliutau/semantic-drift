"""Small resumable API jobs; all share the original spending ledger."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from tqdm import tqdm
from .journal import AdmissionJournal
from .parsing import submission
from .providers import ProviderError
from .runtime import usage,response_ok
from .storage import read_json,write_json

def parallel(jobs,fn,label):
    with ThreadPoolExecutor(max_workers=2) as pool:
        return [f.result() for f in tqdm(as_completed([pool.submit(fn,j) for j in jobs]),total=len(jobs),desc=label)]

def request(folder,provider,model,prompt,label,m,tools=None):
    if (folder/(label+'.json')).exists():return read_json(folder/(label+'.json'))
    now=usage();old=m['budget_before']
    spent=now['reported_credits']+now['uncertain_reserve']-old['reported_credits']-old['uncertain_reserve']
    if now['calls']-old['calls']>=60 or spent>4:raise RuntimeError('Small-search guard reached')
    try:
        r=AdmissionJournal(folder/'calls').complete(label,provider,model,prompt,tools=tools or [],
            max_output_tokens=m.get('tokens',8000),timeout=m.get('seconds',150),retries=1,
            tool_choice=m.get('tool_choice',{'type':'function','name':tools[0]['name']} if tools else 'none'))
        result={'status':r.raw.get('status'),'seconds':r.seconds,'tokens':(r.usage or {}).get('output_tokens')}
        if response_ok(r,model):
            obj,representation=submission(r,tools or [])
            result.update(status='valid',object=obj,representation=representation)
    except (ProviderError,ValueError,KeyError,SyntaxError,TypeError) as exc:result={'status':'unavailable','reason':str(exc)}
    write_json(folder/(label+'.json'),result);return result

