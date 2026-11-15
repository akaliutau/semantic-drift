"""Derived secondary decision metric, with selections made without actor data."""

import argparse
from collections import Counter

import numpy as np

from binary_study import paired_summary
from microstudy import DATA
from selftrap.storage import digest,read_json,utc_now,write_json


def plan():
    directory=DATA/'r5_holdout';m=read_json(directory/'manifest.json')
    fs=read_json(directory/'forecasts.json');f={(x['puzzle_id'],x['author']):x for x in fs}
    prior=read_json(directory/'frozen_reference.json')['p_safe'];rows=[]
    for p in m['puzzles']:
        pid=p['id']
        if any(f[(pid,t)]['status']!='valid' for t in m['models']):continue
        own={t:f[(pid,t)]['p_safe'][i] for i,t in enumerate(m['models'])}
        rows.append({'puzzle':pid,'self_choice':max(m['models'],key=lambda t:(own[t],prior[t])),
                     'history_choice':max(m['models'],key=lambda t:prior[t])})
    result={'at':utc_now(),'phase':'EXPLORATORY SECONDARY; specified during collection, before inspecting R5 decisions',
        'actor_outcomes_read':False,'forecasts_hash':digest(fs),'manifest_hash':digest(m),
        'rule':'argmax own p_safe versus argmax frozen historical p_safe; own ties favor historical rate',
        'rows':rows}
    path=directory/'routing_plan.json'
    if path.exists():raise ValueError('Do not overwrite the selection plan')
    write_json(path,result)
    print('Recorded',len(rows),'policy comparisons using forecasts only; no actor data read')


def analyse():
    if not (DATA/'analysis_ready.json').exists():raise ValueError('Wait for the fixed primary run to finish')
    directory=DATA/'r5_holdout';plan=read_json(directory/'routing_plan.json')
    assert plan['forecasts_hash']==digest(read_json(directory/'forecasts.json'))
    assert plan['manifest_hash']==digest(read_json(directory/'manifest.json'))
    y={(r['puzzle_id'],r['model']):r['safe'] for r in read_json(directory/'actors.json')}
    pairs=[]
    for row in plan['rows']:
        a=y[(row['puzzle'],row['self_choice'])];b=y[(row['puzzle'],row['history_choice'])]
        if a is not None and b is not None:pairs.append((a,b))
    result={'phase':'EXPLORATORY DERIVED POLICY CHECK, NOT ANOTHER CONFIRMATORY PRIMARY',
        'n':len(pairs),'self_routing_safe_rate':float(np.mean([a for a,b in pairs])),
        'history_routing_safe_rate':float(np.mean([b for a,b in pairs])),
        'history_minus_self':paired_summary([float(b)-float(a) for a,b in pairs],np.random.default_rng(2026111590)),
        'self_choice_counts':dict(Counter(r['self_choice'] for r in plan['rows'])),
        'history_choice_counts':dict(Counter(r['history_choice'] for r in plan['rows']))}
    write_json(directory/'routing_analysis.json',result);print(result)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['plan','analyse'])
    plan() if parser.parse_args().stage=='plan' else analyse()
