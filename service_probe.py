"""Two excluded transport probes after an interrupted HTTP504 burst."""

from microstudy import DATA,MODELS,BudgetProvider,cost
from selftrap.storage import Journal,read_json,write_json
from stream_control import StreamingProvider


def main():
    directory=DATA/'service_probe_after_visibility_pause'
    provider=BudgetProvider(concurrency=1);provider.client=StreamingProvider(provider.client)
    model=next(m for m in MODELS if m.id=='gemma-4-31b-it')
    old=read_json(DATA/'r4_discovery/actor_manifest.json')
    job=next(j for j in old if j['model']==model.id and j['arm']=='baseline')
    actor_input=read_json(DATA/'r4_discovery/actors'/job['id']/'input.json')['prompt']
    results=[]
    try:
        for label,text,tokens in [('tiny','Return only JSON {"ok":true}.',64),
                                   ('old_actor',actor_input,8000)]:
            try:
                response=Journal(directory/label).complete('probe',provider,model,text,tools=[],
                    max_output_tokens=tokens,timeout=180,retries=0)
                result={'label':label,'status':response.raw.get('status'),'seconds':response.seconds,
                        'output_tokens':response.usage.get('output_tokens')}
            except Exception as exc:result={'label':label,'error':str(exc)}
            results.append(result);print(result,flush=True)
        write_json(directory/'results.json',{'excluded_transport_probes':results,'cost':cost()})
        print(cost())
    finally:provider.close()


if __name__=='__main__':main()
