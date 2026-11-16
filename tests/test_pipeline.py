import json
import threading
import time

from pipeline.artifacts import digest,read_json
from pipeline.protocol import load_cases
from pipeline.providers import ModelResponse
from pipeline.study import analyse,load_config,run


class FakeProvider:
    def __init__(self):self.calls=0;self.active=0;self.peak=0;self.lock=threading.Lock()
    def complete(self,model,messages,*,tools,**kwargs):
        with self.lock:
            self.calls+=1;self.active+=1;self.peak=max(self.peak,self.active)
        try:
            time.sleep(.01);prompt=messages[0]['content'];name=tools[0]['name']
            if name=='submit_rewrite':
                text=prompt.split('PROTECTED SENTENCE:\n')[1].split('\nREPORT:')[0]
                obj={'text':text}
            else:
                text=prompt.split('CANDIDATES:\n')[1].split('\nAUTOMATIC WORD DIFFS')[0].split('\nSubmit your final')[0]
                obj={r['id']:{'admissible':r['text'].startswith('## Report'),'reason':'fixture'} for r in json.loads(text)}
            output=[{'type':'function_call','name':name,'arguments':json.dumps(obj)}]
            usage={'input_tokens':10,'output_tokens':20,'cost':.0001}
            return ModelResponse('',output,{'status':'completed','model':model.id,'output':output,'usage':usage},
                                 .01,usage,.0001,'credits')
        finally:
            with self.lock:self.active-=1
    def close(self):pass


class MalformedAuthor(FakeProvider):
    def complete(self,model,messages,*,tools,**kwargs):
        response=super().complete(model,messages,tools=tools,**kwargs)
        if tools[0]['name']=='submit_rewrite':
            response.output[0]['arguments']='{"text":null}'
        return response


def test_full_pipeline_can_be_reanalysed_without_api_and_preserves_sources(tmp_path):
    c=load_config('config.json');c['judges']=c['judges'][:2]
    c['author']={**c['author'],'id':'fixture-author'}
    c.update(workers=3,requests_per_minute=60000,tokens_per_minute=10000000,
             max_minutes=5,analysis_reserve_seconds=1)
    cases=load_cases('data/cases.jsonl',2,c['seed']);provider=FakeProvider();out=tmp_path/'run'
    summary=run(out,tmp_path/'cache',c,cases,'data/cases.jsonl',provider)
    assert provider.calls==18 and 1<provider.peak<=3
    assert summary['metrics']['api_attempts']==18
    assert all(v['completed']==2 for counts in summary['counts'].values() for v in counts.values())
    original=digest(summary)
    assert digest(analyse(out))==original and provider.calls==18
    m=read_json(out/'manifest.json');assert len(m['source']['sha256'])==64
    assert (out/'observations.csv').exists() and (out/'share/cases.jsonl').exists()
    assert (out/'source/pipeline/study.py').exists()


def test_bad_generation_saves_an_analysable_partial_result_without_retry(tmp_path):
    c=load_config('config.json');c['judges']=c['judges'][:2]
    c['author']={**c['author'],'id':'fixture-author'}
    c.update(workers=3,requests_per_minute=60000,tokens_per_minute=10000000,
             max_minutes=5,analysis_reserve_seconds=1)
    provider=MalformedAuthor();out=tmp_path/'partial'
    s=run(out,tmp_path/'cache',c,load_cases('data/cases.jsonl',1,c['seed']),'data/cases.jsonl',provider)
    assert provider.calls==3 and s['construction_valid']==0
    assert read_json(out/'manifest.json')['status']=='complete_with_missing'
    assert all(v['rewritten']['missing']==1 and v['bare']['completed']==1 for v in s['counts'].values())
    assert (out/'summary.json').exists() and (out/'observations.csv').exists()
