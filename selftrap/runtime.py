"""Reusable provider composition, shared throttling, and the original credit cap."""

from pathlib import Path
import threading
from time import perf_counter

from .providers import AdaptionProvider,ModelSpec,ProviderError
from .streaming import StreamingProvider
from .storage import dumps,read_json
from .throttle import SharedBudget,SharedThrottle

ROOT=Path(__file__).resolve().parent.parent
LEDGER=ROOT/'runs/microresearch/runs/study/budget.json'

MODELS=(ModelSpec('qwen3.5-397b',input_per_million=.55,output_per_million=3.5),
        ModelSpec('gemma-4-31b-it',input_per_million=.09,output_per_million=.34))
RESPONSE_IDS={'qwen3.5-397b':'qwen3.5-397b-a17b-nvfp4','gemma-4-31b-it':'gemma-4-31b-it',
              'glm-5.2':'glm-5.2'}


def usage(path=LEDGER):
    data=read_json(path);calls=data['calls']
    return {'calls':len(calls),'reported_credits':sum(r.get('charged') or 0 for r in calls),
        'uncertain_reserve':sum(r['reserved'] for r in calls if r.get('charged') is None),
        'input_tokens':sum((r.get('usage') or {}).get('input_tokens',0) for r in calls),
        'output_tokens':sum((r.get('usage') or {}).get('output_tokens',0) for r in calls),
        'in_flight':sum(r['status']=='in_flight' for r in calls),
        'hard_ceiling':data['hard_ceiling']}


class BudgetProvider:
    def __init__(self,concurrency=3,*,per_model_concurrency=1,client=None,ledger_path=LEDGER,throttle=None):
        if concurrency < 1 or per_model_concurrency < 1:raise ValueError('Concurrency must be positive')
        self.budget=SharedBudget(ledger_path);self.path=self.budget.path
        directory=self.path.parent/'shared-admission'
        config=read_json(directory/'state.json')['config'] if (directory/'state.json').exists() else {}
        self.throttle=throttle or SharedThrottle(directory,**config)
        self.client=client or StreamingProvider(AdaptionProvider(ROOT/'.env'))
        self.slots=threading.Semaphore(concurrency)
        self.model_locks={};self.lock=threading.Lock();self.per_model_concurrency=per_model_concurrency

    def complete(self,model,messages,*,tools,max_output_tokens,**kwargs):
        with self.lock:lock=self.model_locks.setdefault(model.id,threading.Semaphore(self.per_model_concurrency))
        with lock,self.slots:
            lease=self.throttle.acquire(model.id,len(dumps([messages,tools]).encode())+max_output_tokens)
            index=None;started=perf_counter()
            try:
                index=self.budget.reserve(model,messages,tools,max_output_tokens)
                try:
                    response=self.client.complete(model,messages,tools=tools,max_output_tokens=max_output_tokens,**kwargs)
                except Exception as exc:
                    self.budget.settle(index,status='error',error=str(exc),seconds=perf_counter()-started,
                        status_code=getattr(exc,'status_code',None),retry_after_seconds=getattr(exc,'retry_after_seconds',None))
                    self.throttle.release(lease,model.id,error=exc);lease=None
                    raise
                native=response.usage or {};charge=native.get('cost')
                if type(charge) not in {int,float} or charge<0:charge=response.estimated_cost
                self.budget.settle(index,status='response',charged=charge,usage=native,seconds=response.seconds)
                error=ProviderError('Failed terminal API response') if response.raw.get('status') in {'failed','cancelled'} else None
                self.throttle.release(lease,model.id,usage=native,error=error);lease=None
                return response
            finally:
                if lease is not None:self.throttle.release(lease,model.id,not_sent=index is None)

    def close(self):self.client.close()


def response_ok(response,model):
    return response.raw.get('status')=='completed' and response.raw.get('model')==RESPONSE_IDS[model.id]
