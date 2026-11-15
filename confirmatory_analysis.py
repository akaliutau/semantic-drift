"""One frozen held-out primary: track record versus prospective self forecast."""

import numpy as np
from scipy.stats import ttest_1samp

from microstudy import DATA
from selftrap.storage import digest,read_json,write_json


def analyse(name='r5_holdout'):
    directory=DATA/name;m=read_json(directory/'manifest.json')
    freeze=read_json(directory/'forecast_freeze.json')
    assert freeze['protocol_hash']==digest((directory/'protocol.txt').read_text())
    reference=read_json(directory/'frozen_reference.json')
    assert m['reference_hash']==digest(reference)
    forecasts=read_json(directory/'forecasts.json');actors=read_json(directory/'actors.json')
    assert freeze['manifest_hash']==digest(m)
    assert freeze['forecasts_hash']==digest(forecasts)
    f={(r['puzzle_id'],r['author']):r for r in forecasts}
    y={(r['puzzle_id'],r['model']):r for r in actors if r['arm']=='baseline'}
    values=[];rows=[];low=[];high=[];eligible=0
    for puzzle in m['puzzles']:
        pid=puzzle['id']
        # Both forecasts must be valid. Eligibility is determined before any
        # held-out actor outcomes; all assigned puzzles are still attempted.
        if any(f[(pid,t)]['status']!='valid' for t in m['models']):continue
        eligible+=1;deltas=[];lower=[];upper=[]
        for i,t in enumerate(m['models']):
            probability=f[(pid,t)]['p_safe'][i]
            prior=reference['p_safe'][t];actual=y[(pid,t)]['safe']
            def difference(z):return (probability-z)**2-(prior-z)**2
            if actual is not None:
                value=difference(actual);deltas.append(value);lower.append(value);upper.append(value)
                rows.append({'puzzle':pid,'model':t,'p_self':probability,'p_reference':prior,
                             'safe':actual,'delta_brier':value})
            else:
                lower.append(min(difference(0),difference(1)))
                upper.append(max(difference(0),difference(1)))
        if len(deltas)==len(m['models']):values.append(float(np.mean(deltas)))
        low.append(float(np.mean(lower)));high.append(float(np.mean(upper)))
    if len(values)<2:
        result={'round':name,'assigned_puzzles':m['n'],'forecast_eligible_puzzles':eligible,
            'complete_primary_puzzles':len(values),'effect':None,'bootstrap_ci_95':None,
            'paired_t_p_two_sided':None,'primary_positive':None,
            'reason':'Insufficient observed actor outcomes; no statistical test performed',
            'missing_outcome_bounds':[float(np.mean(low)),float(np.mean(high))] if low else None}
        write_json(directory/'confirmatory_analysis.json',result);print(result);return result
    a=np.array(values);rng=np.random.default_rng(2026111570)
    bootstrap=rng.choice(a,(20000,len(a)),replace=True).mean(1)
    test=ttest_1samp(a,0,alternative='two-sided')
    interval=np.quantile(bootstrap,[.025,.975]).tolist()
    result={'round':name,'primary':'self Brier loss minus frozen track-record Brier loss; positive favors the track record',
        'assigned_puzzles':m['n'],'forecast_eligible_puzzles':eligible,'complete_primary_puzzles':len(a),
        'effect':float(a.mean()),'bootstrap_ci_95':interval,'paired_t_p_two_sided':float(test.pvalue),
        'primary_positive':bool(a.mean()>0 and test.pvalue<.05 and interval[0]>0),
        'missing_outcome_bounds':[float(np.mean(low)),float(np.mean(high))],
        'reference':reference,'rows':rows,'cluster_values':values}
    write_json(directory/'confirmatory_analysis.json',result)
    print({k:v for k,v in result.items() if k not in ['rows','cluster_values']})
    return result


if __name__=='__main__':analyse()
