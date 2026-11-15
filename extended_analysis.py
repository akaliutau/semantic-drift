"""Exploratory paired analyses and discovery-only calibration for a later holdout."""

import argparse
from collections import Counter,defaultdict
from pathlib import Path

import numpy as np

from binary_study import paired_summary
from microstudy import DATA,cost
from selftrap.storage import digest,read_json,write_json


def analyse(name,calibration_name=None):
    directory=DATA/name;m=read_json(directory/'manifest.json')
    fs=read_json(directory/'forecasts.json');actors=read_json(directory/'actors.json')
    models=m['models'];f={(x['puzzle_id'],x['author']):x for x in fs}
    y={(x['puzzle_id'],x['model'],x['arm']):x for x in actors}
    rng=np.random.default_rng(2026111550)
    common={t:[p['id'] for p in m['puzzles'] if y[(p['id'],t,'baseline')]['safe'] is not None
               and all(f[(p['id'],a)]['status']=='valid' for a in models)] for t in models}
    base={t:[y[(pid,t,'baseline')]['safe'] for pid in common[t]] for t in models}
    calibration={'source':name,'target_rate':{t:(sum(v)+1)/(len(v)+2) for t,v in base.items()},'forecast_mean':{}}
    for a in models:
        for i,t in enumerate(models):
            ps=[f[(pid,a)]['p_safe'][i] for pid in common[t]]
            calibration['forecast_mean'][a+'|'+t]=float(np.mean(ps)) if ps else None
    write_json(directory/'calibration.json',calibration)
    fixed=read_json(DATA/calibration_name/'calibration.json') if calibration_name else None
    raw=[];adjusted=[];prior_skill=[];cal_rows=[]
    for p in m['puzzles']:
        pid=p['id'];raw_one=[];adj_one=[];prior_one=[]
        for i,t in enumerate(models):
            actual=y[(pid,t,'baseline')]['safe']
            if actual is None or any(f[(pid,a)]['status']!='valid' for a in models):continue
            probs={a:f[(pid,a)]['p_safe'][i] for a in models}
            own_loss=(probs[t]-actual)**2
            peer_loss=np.mean([(v-actual)**2 for a,v in probs.items() if a!=t])
            raw_one.append(float(peer_loss-own_loss))
            corrected={}
            if fixed:
                target_rate=fixed['target_rate'][t]
                for a,v in probs.items():
                    corrected[a]=float(np.clip(v-fixed['forecast_mean'][a+'|'+t]+target_rate,.01,.99))
            else:
                remaining=[y[(qid,t,'baseline')]['safe'] for qid in common[t] if qid!=pid]
                target_rate=(sum(remaining)+1)/(len(remaining)+2)
                for a,v in probs.items():
                    other=[f[(qid,a)]['p_safe'][i] for qid in common[t] if qid!=pid]
                    if other:corrected[a]=float(np.clip(v-np.mean(other)+target_rate,.01,.99))
            if len(corrected)==len(models):
                adj_one.append(float(np.mean([(v-actual)**2 for a,v in corrected.items() if a!=t])
                                     -(corrected[t]-actual)**2))
            prior_one.append(float((target_rate-actual)**2-own_loss))
            cal_rows.append({'puzzle':pid,'target':t,'actual_safe':actual,'self_p':probs[t],
                             'peer_p':float(np.mean([v for a,v in probs.items() if a!=t]))})
        if len(raw_one)==len(models):raw.append(float(np.mean(raw_one)))
        if len(adj_one)==len(models):adjusted.append(float(np.mean(adj_one)))
        if len(prior_one)==len(models):prior_skill.append(float(np.mean(prior_one)))
    arms=m.get('arms',['baseline','generic','self','peer'])
    contrasts={}
    for a,b in [('self','generic'),('peer','generic'),('shuffled','generic'),('self','shuffled'),('self','peer'),('generic','baseline')]:
        if a not in arms or b not in arms:continue
        points=[];low=[];high=[];paired=0
        for p in m['puzzles']:
            ds=[];ls=[];hs=[]
            for t in models:
                one=y.get((p['id'],t,a));two=y.get((p['id'],t,b))
                # Missing intervention eligibility is fixed before actor outcomes.
                if one is None or two is None:continue
                x,z=one['safe'],two['safe']
                if x is not None and z is not None:
                    ds.append(float(x)-float(z));paired+=1
                ls.append((float(x) if x is not None else 0)-(float(z) if z is not None else 1))
                hs.append((float(x) if x is not None else 1)-(float(z) if z is not None else 0))
            if ds:points.append(float(np.mean(ds)))
            if ls:low.append(float(np.mean(ls)));high.append(float(np.mean(hs)))
        contrasts[a+'-vs-'+b]={**paired_summary(points,rng),'paired_model_puzzles':paired,
                  'missing_outcome_bounds':[float(np.mean(low)),float(np.mean(high))] if low else None,
                  'cluster_values':points}
    selections={};selection_rows=[]
    for criterion in ['target_only','target_minus_peers']:
        selection=[]
        for start in range(0,len(m['puzzles'])-3,4):
            block=m['puzzles'][start:start+4]
            if any(f[(p['id'],a)]['status']!='valid' for p in block for a in models):continue
            delta={}
            for a in models:
                for i,t in enumerate(models):
                    others=[j for j,o in enumerate(models) if o!=t]
                    def score(p):
                        ps=f[(p['id'],a)]['p_trap']
                        return ps[i] if criterion=='target_only' else ps[i]-np.mean([ps[j] for j in others])
                    chosen=max(block,key=score)
                    outcomes={o:y[(chosen['id'],o,'baseline')]['trap'] for o in models}
                    if any(v is None for v in outcomes.values()):continue
                    d=float(outcomes[t])-float(np.mean([v for o,v in outcomes.items() if o!=t]))
                    delta[(a,t)]=d
                    selection_rows.append({'criterion':criterion,'block':start//4,'author':a,'target':t,
                                           'puzzle':chosen['id'],'specificity':d})
            if len(delta)==len(models)**2:
                selection.append(float(np.mean([v for (a,t),v in delta.items() if a==t])
                                       -np.mean([v for (a,t),v in delta.items() if a!=t])))
        selections[criterion]=selection
    coverage={t:{arm:dict(Counter(r['status'] for r in actors if r['model']==t and r['arm']==arm)) for arm in arms} for t in models}
    # Failure classes chosen from the separate R2 feasibility data, not from
    # these actor comparisons: Qwen nondecision; Gemma committed bad move.
    class_for={t:('nondecision' if t=='qwen3.5-397b' else 'trap') for t in models}
    classified_selection=[]
    for start in range(0,len(m['puzzles'])-3,4):
        block=m['puzzles'][start:start+4]
        if any(f[(p['id'],a)]['status']!='valid' for p in block for a in models):continue
        delta={}
        for a in models:
            for i,t in enumerate(models):
                event=class_for[t]
                def predicted(p):
                    v=f[(p['id'],a)]
                    return v['p_trap'][i] if event=='trap' else 1-v['p_safe'][i]-v['p_trap'][i]
                chosen=max(block,key=predicted)
                actual={}
                for o in models:
                    result=y[(chosen['id'],o,'baseline')]
                    actual[o]=None if result['safe'] is None else (result['trap'] if event=='trap'
                                else not result['safe'] and not result['trap'])
                if any(v is None for v in actual.values()):continue
                delta[(a,t)]=float(actual[t])-float(np.mean([v for o,v in actual.items() if o!=t]))
        if len(delta)==len(models)**2:
            classified_selection.append(float(np.mean([v for (a,t),v in delta.items() if a==t])
                                       -np.mean([v for (a,t),v in delta.items() if a!=t])))
    # Additional epistemic description: forecasts are about submitting a safe
    # answer under the explicitly stated budget, not unbounded game competence.
    # These intervals are exploratory in discovery and are not new primaries.
    calibration_summary=[]
    for a in models:
        for i,t in enumerate(models):
            rows=[(f[(p['id'],a)]['p_safe'][i],y[(p['id'],t,'baseline')]['safe'])
                  for p in m['puzzles'] if f[(p['id'],a)]['status']=='valid'
                  and y[(p['id'],t,'baseline')]['safe'] is not None]
            if rows:
                calibration_summary.append({'author':a,'target':t,'n':len(rows),
                    'mean_p_safe':float(np.mean([p for p,z in rows])),
                    'observed_safe_rate':float(np.mean([z for p,z in rows])),
                    'brier':float(np.mean([(p-z)**2 for p,z in rows])),
                    'forecast_minus_observed':paired_summary([p-z for p,z in rows],rng)})
    resource=[]
    for arm in arms:
        for t in models:
            tokens=[];seconds=[]
            for path in (directory/'actors').glob(f'*-{t}-{arm}/calls/*.response.json'):
                r=read_json(path);tokens.append(r['usage']['output_tokens']);seconds.append(r['seconds'])
            if tokens:resource.append({'model':t,'arm':arm,'n':len(tokens),'mean_output_tokens':float(np.mean(tokens)),
                                      'median_output_tokens':float(np.median(tokens)),'mean_seconds':float(np.mean(seconds))})
    report={'round':name,'phase':m['phase'],'puzzles':m['n'],'models':models,
        'calibration_source':calibration_name or 'leave-one-puzzle-out exploratory',
        'H1_raw_self_safe_Brier_advantage':paired_summary(raw,rng),
        'H1_centered_self_safe_Brier_advantage':paired_summary(adjusted,rng),
        'self_forecast_vs_empirical_prior_Brier':paired_summary(prior_skill,rng),
        'H1_constructive_selection_D':paired_summary(selections['target_only'],rng),
        'H1_margin_selection_D_secondary':paired_summary(selections['target_minus_peers'],rng),
        'exploratory_failure_class_selection_D':paired_summary(classified_selection,rng),
        'preassigned_failure_class':class_for,
        'contrasts':contrasts,'coverage':coverage,'resource':resource,
        'forecast_status':dict(Counter((r['author']+'|'+r['status']) for r in fs)),
        'calibration_rows':cal_rows,'calibration_summary':calibration_summary,
        'selection_rows':selection_rows,'cost':cost(),
        'raw_cluster_values':raw,'centered_cluster_values':adjusted,
        'selection_cluster_values':selections['target_only']}
    write_json(directory/'extended_analysis.json',report)
    concise={k:v for k,v in report.items() if k not in ['calibration_rows','selection_rows','raw_cluster_values','centered_cluster_values','selection_cluster_values']}
    concise['contrasts']={k:{x:z for x,z in v.items() if x!='cluster_values'} for k,v in contrasts.items()}
    print(dumps_safe(concise))
    return report


def dumps_safe(value):
    from selftrap.storage import dumps
    return dumps(value)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--calibration')
    a=p.parse_args();analyse(a.name,a.calibration)
