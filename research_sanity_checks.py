"""Tiny synthetic checks of statistics; these are never research observations."""

from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile

from extended_analysis import analyse
from confirmatory_analysis import analyse as confirm
from admission_journal import AdmissionJournal
from selftrap.providers import ModelResponse,ModelSpec,ProviderError
from microstudy import DATA
from selftrap.storage import digest,write_json


def main():
    DATA.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='synthetic-check-',dir=DATA) as temporary:
        path=Path(temporary);models=['model-A','model-B'];puzzles=[{'id':f'p{i}'} for i in range(8)]
        write_json(path/'manifest.json',{'phase':'SYNTHETIC_UNIT_TEST','n':8,'models':models,'puzzles':puzzles,
                   'arms':['baseline','generic','self','peer','shuffled']})
        forecasts=[];actors=[]
        for i,p in enumerate(puzzles):
            actual=[bool(i%2),not bool(i%2)]
            for a in models:
                probs=[(.9 if y else .1) if a==t else (.1 if y else .9) for t,y in zip(models,actual)]
                forecasts.append({'status':'valid','author':a,'puzzle_id':p['id'],'p_safe':probs,
                                  'p_trap':[1-p for p in probs]})
            for t,y in zip(models,actual):
                for arm in ['baseline','generic','self','peer','shuffled']:
                    safe=y if arm=='baseline' else arm in {'self','peer'}
                    actors.append({'model':t,'puzzle_id':p['id'],'arm':arm,'safe':safe,'trap':not safe,
                                   'status':'safe' if safe else 'deadlock'})
        write_json(path/'forecasts.json',forecasts);write_json(path/'actors.json',actors)
        with redirect_stdout(io.StringIO()):report=analyse(path.name)
        assert abs(report['H1_raw_self_safe_Brier_advantage']['effect']-.8)<1e-9
        assert report['H1_constructive_selection_D']['effect']==2
        assert report['H1_margin_selection_D_secondary']['effect']==0
        assert report['contrasts']['self-vs-generic']['effect']==1
        assert report['contrasts']['self-vs-shuffled']['effect']==1
        assert report['contrasts']['self-vs-peer']['effect']==0
    # A different synthetic panel checks the sign, cluster count and fixed
    # reference of the primary test; it is never retained as research evidence.
    with tempfile.TemporaryDirectory(prefix='synthetic-primary-',dir=DATA) as temporary:
        path=Path(temporary);models=['model-A','model-B']
        puzzles=[{'id':f'q{i}'} for i in range(32)]
        reference={'p_safe':{t:.25 for t in models}}
        manifest={'n':32,'models':models,'puzzles':puzzles,'reference_hash':digest(reference)}
        fs=[{'author':t,'puzzle_id':p['id'],'status':'valid','p_safe':[.9,.9]}
            for p in puzzles for t in models]
        actors=[{'puzzle_id':p['id'],'model':t,'arm':'baseline','safe':bool(i%4==0)}
                for i,p in enumerate(puzzles) for t in models]
        protocol='SYNTHETIC ONLY'
        (path/'protocol.txt').write_text(protocol)
        for filename,value in [('manifest',manifest),('forecasts',fs),('actors',actors),
            ('frozen_reference',reference),('forecast_freeze',{'protocol_hash':digest(protocol),
              'manifest_hash':digest(manifest),'forecasts_hash':digest(fs)})]:
            write_json(path/(filename+'.json'),value)
        with redirect_stdout(io.StringIO()):report=confirm(path.name)
        assert report['complete_primary_puzzles']==32
        assert abs(report['effect']-.4225)<1e-9
        assert report['primary_positive']
        for row in actors:row['safe']=None
        write_json(path/'actors.json',actors)
        with redirect_stdout(io.StringIO()):missing=confirm(path.name)
        assert missing['complete_primary_puzzles']==0 and missing['paired_t_p_two_sided'] is None
    class Fake:
        def __init__(self,status):self.status=status;self.timeouts=[]
        def complete(self,*args,timeout,**kwargs):
            self.timeouts.append(timeout)
            if len(self.timeouts)==1:raise ProviderError('Synthetic admission/failure',False,self.status)
            return ModelResponse('',[],{'status':'incomplete'},0,{'input_tokens':0,'output_tokens':0},0,'credits')
    with tempfile.TemporaryDirectory(prefix='synthetic-admission-',dir=DATA) as temporary:
        model=ModelSpec('synthetic');provider=Fake(429)
        response=AdmissionJournal(Path(temporary)/'429').complete('check',provider,model,'dummy',
            tools=[],max_output_tokens=8,timeout=180,retries=2)
        assert response.raw['status']=='incomplete' and len(provider.timeouts)==2
        assert all(t>179.9 for t in provider.timeouts)  # full generation bound after rejection
        for status in [504,None]:
            provider=Fake(status)
            try:
                AdmissionJournal(Path(temporary)/str(status)).complete('check',provider,model,'dummy',
                    tools=[],max_output_tokens=8,timeout=180,retries=2)
            except ProviderError:pass
            else:raise AssertionError('Non-429 failure was unexpectedly retried')
            assert len(provider.timeouts)==1
    print('Synthetic estimator checks passed; temporary observations removed; no API used.')


if __name__=='__main__':main()
