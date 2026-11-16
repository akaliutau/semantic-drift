import time

from pipeline.artifacts import digest,read_json,write_json
from pipeline.protocol import JUDGE_TOOL
from pipeline.providers import ModelResponse,ProviderError
from pipeline.requests import RequestRunner,request_spec


def config():
    return {'requests_per_minute':60000,'tokens_per_minute':10000000,'per_model_concurrency':2,
            'max_attempts':3,'request_timeout_seconds':1,'max_new_cost':5,'total_cost_ceiling':190,'temperature':0}


MODEL={'id':'test-model','max_output_tokens':100,'input_per_million':.1,'output_per_million':.1}


class RetryProvider:
    def __init__(self,failures):self.failures=failures;self.calls=0
    def complete(self,model,*args,**kwargs):
        self.calls+=1
        if self.calls<=self.failures:
            error=ProviderError('HTTP 429',True,429);error.retry_after_seconds=.001;raise error
        return ModelResponse('not JSON',[],{'status':'completed','model':model.id},.001,
                             {'input_tokens':2,'output_tokens':3,'cost':.0001},.0001,'credits')
    def close(self):pass


def test_three_total_attempts_and_no_retry_of_invalid_model_output(tmp_path):
    provider=RetryProvider(2)
    runner=RequestRunner(tmp_path/'out',tmp_path/'cache',config(),time.monotonic()+10,'run',provider)
    r=runner.ask(MODEL,'question',JUDGE_TOOL)
    assert provider.calls==3 and len(r['attempts'])==3
    assert r['status']=='invalid_submission'
    runner.ask(MODEL,'question',JUDGE_TOOL)
    assert provider.calls==3 and runner.cache_hits==1
    runner.close()


def test_retry_cap_and_interrupted_cache_are_preserved(tmp_path):
    provider=RetryProvider(100)
    runner=RequestRunner(tmp_path/'out',tmp_path/'cache',config(),time.monotonic()+10,'run',provider)
    r=runner.ask(MODEL,'fail',JUDGE_TOOL)
    assert provider.calls==3 and r['status']=='api_error'
    req=request_spec(MODEL,'interrupted',JUDGE_TOOL);key=digest(req)
    write_json(tmp_path/'cache'/'requests'/(key+'.json'),
               {'request_sha256':key,'request':req,'status':'in_flight','attempts':[]})
    r=runner.ask(MODEL,'interrupted',JUDGE_TOOL)
    assert r['status']=='unknown_interrupted' and provider.calls==3
    runner.close()
