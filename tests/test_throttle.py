"""No live calls: verify shared admission and spending, not model behavior."""

from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timezone
from email.utils import format_datetime
from pathlib import Path
import time

import pytest

from selftrap.runtime import BudgetProvider
from selftrap.providers import ModelSpec,ProviderError
from selftrap.storage import read_json,write_json
from selftrap.throttle import CollectionStopped,SharedBudget,SharedThrottle,retry_after_seconds


def test_retry_after_formats():
    assert retry_after_seconds('90',100)==90
    assert retry_after_seconds(format_datetime(datetime.fromtimestamp(180,timezone.utc)),100)==80
    assert retry_after_seconds('garbage',100) is None
    assert retry_after_seconds('nan',100) is None


class Clock:
    def __init__(self):self.now=1000.
    def time(self):return self.now
    def sleep(self,n):self.now+=n


def test_shared_cooldown_and_circuit(tmp_path):
    clock=Clock();one=SharedThrottle(tmp_path,clock=clock.time,sleep=clock.sleep)
    lease=one.acquire('model',100)
    error=ProviderError('429',True,429);error.retry_after_seconds=140
    one.release(lease,'model',error=error)
    two=SharedThrottle(tmp_path,clock=clock.time,sleep=clock.sleep)
    lease=two.acquire('model',100)
    assert clock.now>=1140
    two.release(lease,'model',error=error)
    with pytest.raises(CollectionStopped):one.acquire('other',100)
    until=read_json(one.path)['cooldown_until']
    one.reset('Excluded service diagnostic succeeded')
    assert read_json(one.path)['cooldown_until']==until


def test_token_and_request_windows_are_shared(tmp_path):
    clock=Clock()
    opts=dict(interval=0,rpm=2,tpm=10,clock=clock.time,sleep=clock.sleep)
    one=SharedThrottle(tmp_path,**opts);two=SharedThrottle(tmp_path,**opts)
    lease=one.acquire('m',6)
    one.release(lease,'m',usage={'input_tokens':1,'output_tokens':1})
    lease=two.acquire('n',8);two.release(lease,'n')
    assert clock.now==1000
    lease=one.acquire('m',1)
    assert clock.now>=1060
    one.release(lease,'m')


def test_local_rejection_does_not_clear_failure_streak(tmp_path):
    clock=Clock();limiter=SharedThrottle(tmp_path,clock=clock.time,sleep=clock.sleep)
    lease=limiter.acquire('m',1)
    limiter.release(lease,'m',error=ProviderError('429',True,429))
    lease=limiter.acquire('m',1);limiter.release(lease,'m',not_sent=True)
    assert read_json(limiter.path)['model_failures']['m']==1
    assert not any(s['id']==lease[0] for s in read_json(limiter.path)['starts'])


def paced_process(directory):
    limiter=SharedThrottle(directory,interval=.04,rpm=1000,tpm=100000,concurrency=3)
    lease=limiter.acquire('model',10);started=time.time()
    time.sleep(.01);limiter.release(lease,'model',usage={'input_tokens':1,'output_tokens':1})
    return started


def test_independent_processes_share_rate_window(tmp_path):
    with ProcessPoolExecutor(max_workers=3) as pool:
        times=sorted(pool.map(paced_process,[str(tmp_path)]*6))
    assert times[-1]-times[0]>=.18
    assert len(read_json(tmp_path/'state.json')['starts'])==6


def reserve_process(args):
    path,n=args;budget=SharedBudget(path)
    model=ModelSpec('model',input_per_million=1,output_per_million=1)
    i=budget.reserve(model,'x',[],100);budget.settle(i,status='response',charged=.001,usage={})
    return i


def test_budget_is_not_reset_or_lost_across_processes(tmp_path):
    path=tmp_path/'budget.json'
    write_json(path,{'hard_ceiling':190,'operational_ceiling':160,'calls':[
        {'index':0,'charged':None,'reserved':27,'status':'interrupted_unknown'}]})
    with ProcessPoolExecutor(max_workers=3) as pool:indices=list(pool.map(reserve_process,[(str(path),i) for i in range(9)]))
    assert sorted(indices)==list(range(1,10))
    ledger=read_json(path)
    assert len(ledger['calls'])==10 and ledger['calls'][0]['reserved']==27
    ledger['calls'][0]['reserved']=160;write_json(path,ledger)
    with pytest.raises(CollectionStopped):SharedBudget(path).reserve(ModelSpec('m',output_per_million=1),'x',[],1)


def test_blocked_request_never_reaches_client_or_reserves_money(tmp_path):
    path=tmp_path/'budget.json';write_json(path,{'hard_ceiling':190,'operational_ceiling':160,'calls':[]})
    clock=Clock();limiter=SharedThrottle(tmp_path/'admission',failure_limit=1,clock=clock.time,sleep=clock.sleep)
    lease=limiter.acquire('m',1);limiter.release(lease,'m',error=ProviderError('quota',False,402))
    class Client:
        def complete(self,*args,**kwargs):raise AssertionError('Network should not be reached')
    provider=BudgetProvider(client=Client(),ledger_path=path,throttle=limiter)
    with pytest.raises(CollectionStopped):provider.complete(ModelSpec('m'),'x',tools=[],max_output_tokens=1,timeout=5)
    assert not read_json(path)['calls']
