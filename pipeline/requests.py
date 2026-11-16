"""Content-addressed calls, bounded retries, concurrent throttling and budget."""

from pathlib import Path
import random
import shutil
import threading
import time
from uuid import uuid4

from .artifacts import digest,file_lock,read_json,utc_now,write_json
from .parsing import submission
from .providers import AdaptionProvider,ModelSpec,ProviderError
from .streaming import StreamingProvider


def request_spec(model,prompt,tools,temperature=0):
    return {'model':model['id'],'provider':'adaption','temperature':temperature,
            'messages':[{'role':'user','content':prompt}], 'tools':tools,
            'tool_choice':{'type':'function','name':tools[0]['name']},
            'max_output_tokens':model['max_output_tokens']}


class NotAdmitted(RuntimeError):pass


class Throttle:
    def __init__(self,config,deadline,stop):
        self.config=config;self.deadline=deadline;self.stop=stop
        self.cv=threading.Condition();self.starts=[];self.next_start=0.;self.cooldown=0.
        self.interval=60/config['requests_per_minute'];self.base_interval=self.interval
        self.successes=0

    def acquire(self,tokens):
        if tokens>self.config['tokens_per_minute']:raise NotAdmitted('Request exceeds token admission window')
        with self.cv:
            while True:
                now=time.monotonic()
                if self.stop.is_set():raise NotAdmitted('Run stopped')
                if now>=self.deadline:raise NotAdmitted('Run deadline reached')
                self.starts=[s for s in self.starts if s['at']>now-60]
                wait=max(0.,self.next_start-now,self.cooldown-now)
                if len(self.starts)>=self.config['requests_per_minute']:
                    wait=max(wait,self.starts[-self.config['requests_per_minute']]['at']+60-now)
                total=sum(s['tokens'] for s in self.starts)
                for s in self.starts:
                    if total+tokens<=self.config['tokens_per_minute']:break
                    total-=s['tokens'];wait=max(wait,s['at']+60-now)
                if wait<=0:
                    lease=uuid4().hex;self.starts.append({'id':lease,'at':now,'tokens':tokens})
                    self.next_start=now+self.interval
                    return lease
                self.cv.wait(min(wait,1.,max(.01,self.deadline-now)))

    def finish(self,lease,usage=None,retry_after=None):
        with self.cv:
            if usage:
                for s in self.starts:
                    if s['id']==lease:s['tokens']=(usage.get('input_tokens') or 0)+(usage.get('output_tokens') or 0)
            if retry_after is not None:
                self.cooldown=max(self.cooldown,time.monotonic()+retry_after)
                self.interval=min(60.,self.interval*2);self.successes=0
            else:
                self.successes+=1
                if self.successes>=10:
                    self.interval=max(self.base_interval,self.interval*.9);self.successes=0
            self.cv.notify_all()


class Budget:
    def __init__(self,path,config,run_id):
        self.path=Path(path);self.config=config;self.run_id=run_id
        with file_lock(self.path.with_suffix('.lock')):
            if not self.path.exists():
                baseline=config.get('budget_baseline',{})
                write_json(self.path,{'ceiling':config['total_cost_ceiling'],
                                     'carry_reported':baseline.get('reported',0.),
                                     'carry_unknown':baseline.get('uncertain',0.),'entries':[]})
    def reserve(self,amount):
        with file_lock(self.path.with_suffix('.lock')):
            data=read_json(self.path)
            value=lambda r:r['reserve'] if r['charge'] is None else r['charge']
            used=data['carry_reported']+data['carry_unknown']+sum(value(r) for r in data['entries'])
            current=sum(value(r) for r in data['entries'] if r['run']==self.run_id)
            if used+amount>min(data['ceiling'],self.config['total_cost_ceiling']):
                raise NotAdmitted('Cumulative spending ceiling reached')
            if current+amount>self.config['max_new_cost']:raise NotAdmitted('Run spending ceiling reached')
            i=len(data['entries']);data['entries'].append({'run':self.run_id,'reserve':amount,'charge':None})
            write_json(self.path,data);return i
    def settle(self,index,charge):
        with file_lock(self.path.with_suffix('.lock')):
            data=read_json(self.path);data['entries'][index]['charge']=charge;write_json(self.path,data)


class RequestRunner:
    def __init__(self,output,cache,config,deadline,run_id,provider=None):
        self.output=Path(output);self.cache=Path(cache);self.config=config;self.deadline=deadline
        self.run_id=run_id
        self.stop=threading.Event();self.throttle=Throttle(config,deadline,self.stop)
        self.provider=provider
        self._provider_lock=threading.Lock();self.lock=threading.Lock();self.model_locks={}
        self.budget=Budget(self.cache/'budget.json',config,run_id)
        self.new_attempts=0;self.cache_hits=0;self.disabled=set();self.failures={}

    def _client(self):
        with self._provider_lock:
            if self.provider is None:self.provider=StreamingProvider(AdaptionProvider())
            return self.provider

    def ask(self,model,prompt,tools):
        req=request_spec(model,prompt,tools,self.config['temperature']);key=digest(req)
        path=self.cache/'requests'/(key+'.json')
        with file_lock(path.with_suffix('.lock')):
            bundled=Path(__file__).resolve().parent.parent/'examples/cache/requests'/(key+'.json')
            if not path.exists() and bundled.exists():
                path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(bundled,path)
            if path.exists():
                record=read_json(path)
                if record['request_sha256']!=key or digest(record['request'])!=key:
                    raise ValueError('Cached request hash mismatch')
                if record['status']=='in_flight':
                    record['status']='unknown_interrupted';record['error']='Interrupted request is not automatically replayed.'
                with self.lock:self.cache_hits+=1
                return self._save(record,True)
            with self.lock:
                sem=self.model_locks.setdefault(model['id'],threading.Semaphore(self.config['per_model_concurrency']))
            with sem:
                record={'request_sha256':key,'request':req,'created_at':utc_now(),'origin_run_id':self.run_id,
                        'status':'in_flight','attempts':[]}
                write_json(path,record)
                for attempt in range(1,self.config['max_attempts']+1):
                    with self.lock:disabled=model['id'] in self.disabled
                    if disabled:
                        record.update(status='not_started_model_disabled');break
                    # Conservative bytes-based token reservation; replace with actual usage after completion.
                    tokens=len(str(req).encode())+req['max_output_tokens']
                    try:lease=self.throttle.acquire(tokens)
                    except NotAdmitted as exc:
                        record.update(status='not_started',error=str(exc));break
                    estimate=2*((len(str(req).encode())+4096)*model.get('input_per_million',0)
                                +req['max_output_tokens']*model.get('output_per_million',0))/1e6
                    try:index=self.budget.reserve(max(.1,estimate))
                    except NotAdmitted as exc:
                        self.throttle.finish(lease);self.stop.set()
                        record.update(status='not_started',error=str(exc));break
                    a={'attempt':attempt,'started_at':utc_now(),'charge':None}
                    record['attempts'].append(a);write_json(path,record)
                    with self.lock:self.new_attempts+=1
                    start=time.monotonic();remaining=self.deadline-start
                    try:
                        spec=ModelSpec(model['id'],input_per_million=model.get('input_per_million',0),
                                       output_per_million=model.get('output_per_million',0),
                                       settings={'temperature':self.config['temperature']})
                        response=self._client().complete(spec,req['messages'],tools=tools,
                            max_output_tokens=req['max_output_tokens'],tool_choice=req['tool_choice'],
                            timeout=max(.1,min(self.config['request_timeout_seconds'],remaining)))
                        record['response']=response.raw
                        usage=response.usage or {};charge=usage.get('cost')
                        if type(charge) not in (int,float) or charge<0:charge=response.estimated_cost
                        a.update(seconds=time.monotonic()-start,charge=charge,usage=usage)
                        self.budget.settle(index,charge);self.throttle.finish(lease,usage)
                        expected=model.get('response_id',model['id'])
                        if response.raw.get('status')!='completed' or response.raw.get('model')!=expected:
                            record.update(status='unavailable',error='Incomplete terminal response or unexpected served model')
                            break
                        try:
                            obj,encoding=submission(response,tools)
                            record.update(status='completed',object=obj,encoding=encoding)
                        except (ValueError,SyntaxError,TypeError) as exc:
                            record.update(status='invalid_submission',error=str(exc))
                        break  # Never retry model answers, invalid JSON, or incomplete generations.
                    except ProviderError as exc:
                        a.update(seconds=time.monotonic()-start,http_status=exc.status_code,error=str(exc),
                                 retryable=exc.retryable,retry_after_seconds=getattr(exc,'retry_after_seconds',None))
                        rejected=exc.status_code in {400,401,402,403,404,422,429}
                        if rejected:a['charge']=0.;self.budget.settle(index,0.)
                        wait=(getattr(exc,'retry_after_seconds',None) or 65.) if exc.status_code==429 else min(30.,2**attempt)
                        self.throttle.finish(lease,retry_after=wait if exc.status_code==429 else None)
                        record.update(status='api_error',error=str(exc))
                        if exc.status_code in {401,402,403} or getattr(exc,'quota_exhausted',False):self.stop.set()
                        if exc.status_code in {400,404,422}:
                            with self.lock:self.disabled.add(model['id'])
                        if not exc.retryable or attempt==self.config['max_attempts'] or self.stop.is_set():break
                        if time.monotonic()+wait>=self.deadline:break
                        self.stop.wait(wait+random.uniform(0,.25))
                    except Exception as exc:
                        self.throttle.finish(lease)
                        a.update(seconds=time.monotonic()-start,error=type(exc).__name__)
                        record.update(status='local_error',error=type(exc).__name__);break
                    finally:write_json(path,record)
                record['finished_at']=utc_now();write_json(path,record)
                if not record['attempts'] and record['status'].startswith('not_started'):
                    path.unlink(missing_ok=True)
                with self.lock:
                    self.failures[model['id']]=0 if record['status']=='completed' else self.failures.get(model['id'],0)+1
                    if self.failures[model['id']]>=2:self.disabled.add(model['id'])
                return self._save(record,False)

    def _save(self,record,cached):
        write_json(self.output/'calls'/(record['request_sha256']+'.json'),record)
        return {**record,'cached':cached}

    def close(self):
        if self.provider is not None:self.provider.close()
