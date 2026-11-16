"""Small JSON experiments using the original provider, throttle and budget.

No automatic retry of model outcomes. Designs and all request identities are
sealed; incomplete responses are technical outcomes, never reasoning scores.
"""

from dataclasses import asdict
from pathlib import Path
import threading
from time import perf_counter

from .journal import AdmissionJournal
from .parsing import submission
from .providers import ModelSpec
from .runtime import BudgetProvider, response_ok, usage
from .storage import digest, read_json, utc_now, write_json


MODELS = (ModelSpec('gemma-4-31b-it', input_per_million=.09, output_per_million=.34,
                    settings={'temperature':0}),
          ModelSpec('glm-5.3-flash', input_per_million=.15, output_per_million=.5,
                    settings={'temperature':0}))
TOKENS = {'gemma-4-31b-it':3000, 'glm-5.3-flash':8192}


def seal(directory, spec):
    path=Path(directory)/'design.json'
    if path.exists():
        design=read_json(path)
        if design['spec_hash']!=digest(spec):raise ValueError('Changed design; use a fresh directory')
        return design
    design={'at':utc_now(),'before_api':True,'spec_hash':digest(spec),'spec':spec,'budget_before':usage()}
    write_json(path,design)
    return design


class Calls:
    def __init__(self,directory,design,*,max_calls,credit_cap=1.0,provider=None):
        self.directory=Path(directory);self.design=design
        self.max_calls=max_calls;self.credit_cap=credit_cap
        self.count=len(list((self.directory/'receipts').rglob('*.request.json')))
        self.lock=threading.Lock();self.provider=provider or BudgetProvider(concurrency=2)

    def ask(self,model,label,prompt,*,tokens=None,tools=None):
        tokens=tokens or TOKENS[model.id]
        tools=tools or []
        identity_fields={'model':asdict(model),'prompt':prompt,'tokens':tokens}
        if tools:identity_fields['tools']=tools
        identity=digest(identity_fields)
        path=self.directory/'responses'/model.id/(label+'.json')
        if path.exists():
            row=read_json(path)
            if row['request_identity']!=identity:raise ValueError('Cached request changed')
            return row
        receipts=self.directory/'receipts'/model.id/label
        restoring=bool(list(receipts.glob('*.request.json')))
        with self.lock:
            if not restoring:
                now,old=usage(),self.design['budget_before']
                spent=now['reported_credits']+now['uncertain_reserve']-old['reported_credits']-old['uncertain_reserve']
                reserve=2*((len(prompt.encode())+4096)*model.input_per_million+tokens*model.output_per_million)/1e6
                if self.count>=self.max_calls or spent+reserve>self.credit_cap:
                    raise RuntimeError('Small-pilot request or credit cap reached')
                self.count+=1
        row={'request_identity':identity,'model':model.id,'label':label,'started_at':utc_now()}
        started=perf_counter()
        try:
            r=AdmissionJournal(receipts).complete(label,self.provider,model,
                [{'role':'user','content':prompt}],tools=tools,
                tool_choice={'type':'function','name':tools[0]['name']} if tools else 'none',
                max_output_tokens=tokens,timeout=120,retries=0)
            row.update(api_status=r.raw.get('status'),served_model=r.raw.get('model'),seconds=r.seconds,usage=r.usage)
            if not response_ok(r,model):raise ValueError('Incomplete response or wrong model')
            obj,encoding=submission(r,tools)
            row.update(status='completed',object=obj,encoding=encoding)
        except Exception as exc:
            row.update(status='unavailable',error=f'{type(exc).__name__}: {exc}')
        row.update(wall_seconds=perf_counter()-started,finished_at=utc_now())
        write_json(path,row)
        return row

    def close(self):self.provider.close()


def accounting(directory):
    responses=[read_json(p) for p in Path(directory).rglob('*.response.json')]
    requests=list(Path(directory).rglob('*.request.json'))
    return {'request_attempts':len(requests),'responses':len(responses),
        'input_tokens':sum((r['usage'] or {}).get('input_tokens',0) for r in responses),
        'output_tokens':sum((r['usage'] or {}).get('output_tokens',0) for r in responses),
        'receipt_estimated_credits':sum(r['estimated_cost'] or 0 for r in responses),
        'api_seconds':sum(r['seconds'] for r in responses),
        'unknown_outcome_requests':len(requests)-len(responses)}
