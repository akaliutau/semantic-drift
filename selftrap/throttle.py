"""Cross-process pacing, concurrency slots and a persistent failure circuit."""

from contextlib import contextmanager
from datetime import timezone
from email.utils import parsedate_to_datetime
import fcntl
import math
from pathlib import Path
import random
import time
from uuid import uuid4

from selftrap.providers import ProviderError
from selftrap.storage import read_json,write_json,utc_now


class CollectionStopped(ProviderError):
    def __init__(self,message):
        super().__init__(message,False)
        self.billing='not_sent'


@contextmanager
def file_lock(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        try:yield
        finally:fcntl.flock(stream,fcntl.LOCK_UN)


def retry_after_seconds(value,now=None):
    """RFC Retry-After supports seconds and HTTP dates; never log raw headers."""
    if value is None:return None
    try:
        seconds=float(value)
        return max(0.,seconds) if math.isfinite(seconds) else None
    except (ValueError,TypeError):pass
    try:
        stamp=parsedate_to_datetime(value)
        if stamp.tzinfo is None:stamp=stamp.replace(tzinfo=timezone.utc)
        return max(0.,stamp.timestamp()-(time.time() if now is None else now))
    except (ValueError,TypeError,OverflowError):return None


class SharedThrottle:
    def __init__(self,directory,*,interval=6.,rpm=10,tpm=96000,concurrency=3,
                 failure_limit=2,clock=time.time,sleep=time.sleep):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.path=self.directory/'state.json';self.lock=self.directory/'state.lock'
        self.config={'interval':interval,'rpm':rpm,'tpm':tpm,'concurrency':concurrency,
                     'failure_limit':failure_limit}
        self.clock=clock;self.sleep=sleep
        with file_lock(self.lock):
            if self.path.exists():
                if read_json(self.path)['config']!=self.config:raise ValueError('Shared throttle configuration differs')
            else:write_json(self.path,{'config':self.config,'interval':interval,'starts':[],
                'cooldown_until':0,'stopped':False,'reason':None,'global_failures':0,
                'model_failures':{},'events':[]})

    def acquire(self,model,tokens,*,probe=False):
        if tokens>self.config['tpm']:raise ValueError('One request exceeds the configured token admission window')
        while True:
            with file_lock(self.lock):
                state=read_json(self.path);now=self.clock()
                if state['stopped'] and not probe:raise CollectionStopped(state['reason'])
                starts=[s for s in state['starts'] if s['at']>now-60]
                wait=max(0.,state['cooldown_until']-now)
                if starts:wait=max(wait,starts[-1]['at']+state['interval']-now)
                if len(starts)>=self.config['rpm']:wait=max(wait,starts[-self.config['rpm']]['at']+60-now)
                total=sum(s['tokens'] for s in starts)
                for previous in starts:
                    if total+tokens<=self.config['tpm']:break
                    total-=previous['tokens'];wait=max(wait,previous['at']+60-now)
                if wait<=0:
                    for slot in range(self.config['concurrency']):
                        handle=(self.directory/f'slot-{slot}.lock').open('a')
                        try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
                        except BlockingIOError:handle.close();continue
                        request_id=uuid4().hex
                        state['starts']=starts+[{'id':request_id,'at':now,'tokens':tokens,'model':model}]
                        write_json(self.path,state)
                        return request_id,handle
                    wait=.2
            # Do not hold the state lock during a wait: other processes must
            # be able to publish a cooldown or open the circuit immediately.
            self.sleep(min(max(wait,.01),1.))

    def release(self,lease,model,*,usage=None,error=None,not_sent=False):
        request_id,handle=lease
        try:
            with file_lock(self.lock):
                state=read_json(self.path);now=self.clock()
                if not_sent:
                    # A local budget rejection is neither a server success
                    # nor an infrastructure failure.
                    state['starts']=[s for s in state['starts'] if s['id']!=request_id]
                elif error is None:
                    state['global_failures']=0;state['model_failures'][model]=0
                    if usage:
                        actual=(usage.get('input_tokens') or 0)+(usage.get('output_tokens') or 0)
                        for s in state['starts']:
                            if s['id']==request_id:s['tokens']=actual
                else:
                    code=getattr(error,'status_code',None)
                    state['global_failures']+=1
                    state['model_failures'][model]=state['model_failures'].get(model,0)+1
                    if code==429:
                        streak=state['model_failures'][model]
                        fallback=min(300.,65.*2**(streak-1))+random.uniform(0,3)
                        delay=max(fallback,getattr(error,'retry_after_seconds',0) or 0)
                        state['cooldown_until']=max(state['cooldown_until'],now+delay)
                        state['interval']=min(60.,state['interval']*2)
                    else:state['cooldown_until']=max(state['cooldown_until'],now+10)
                    fatal=code in {400,401,402,403,404,422} or getattr(error,'quota_exhausted',False)
                    repeated=max(state['global_failures'],state['model_failures'][model])>=self.config['failure_limit']
                    if fatal or repeated:
                        state['stopped']=True;state['reason']=f'Collection stopped after infrastructure failures (HTTP {code})'
                    state['events'].append({'at':utc_now(),'model':model,'status_code':code,
                        'retry_after_seconds':getattr(error,'retry_after_seconds',None),
                        'cooldown_until':state['cooldown_until'],'stopped':state['stopped']})
                write_json(self.path,state)
        finally:
            fcntl.flock(handle,fcntl.LOCK_UN);handle.close()

    def reset(self,reason,*,min_interval=None):
        if not reason.strip():raise ValueError('A diagnostic reason is required')
        with file_lock(self.lock):
            state=read_json(self.path)
            state.update(stopped=False,reason=None,global_failures=0,model_failures={})
            if min_interval is not None:
                if not math.isfinite(min_interval) or min_interval<0:raise ValueError('Invalid start interval')
                state['interval']=max(state['interval'],min_interval)
            # Resetting never bypasses the server's Retry-After deadline or
            # restores a faster rate. It only permits a diagnosed resumption.
            state['events'].append({'at':utc_now(),'reset_reason':reason,'interval':state['interval']})
            write_json(self.path,state)


class SharedBudget:
    def __init__(self,path):
        self.path=Path(path);self.lock=self.path.with_suffix('.lock')
        if not self.path.exists():raise FileNotFoundError('Reuse the original study ledger; do not create another allowance')

    def reserve(self,model,messages,tools,max_output_tokens):
        from selftrap.storage import dumps,digest
        amount=2*((len(dumps([messages,tools]).encode())+4096)*model.input_per_million
                  +max_output_tokens*model.output_per_million)/1e6
        with file_lock(self.lock):
            ledger=read_json(self.path)
            held=sum(c['charged'] if c['charged'] is not None else c['reserved'] for c in ledger['calls'])
            if held+amount>min(160,ledger['operational_ceiling'],ledger['hard_ceiling']):
                raise CollectionStopped('Original study credit ceiling reached')
            entry={'index':len(ledger['calls']),'at':utc_now(),'model':model.id,'reserved':amount,
                'charged':None,'status':'in_flight','request_hash':digest([messages,tools,max_output_tokens])}
            ledger['calls'].append(entry);write_json(self.path,ledger)
            return entry['index']

    def settle(self,index,**fields):
        with file_lock(self.lock):
            ledger=read_json(self.path);ledger['calls'][index].update(fields)
            write_json(self.path,ledger)
