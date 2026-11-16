"""Prepare, run, analyse and export a complete paired experiment."""

from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import csv
import importlib.metadata
import math
from pathlib import Path
import random
import shutil
import sys
import time
from uuid import uuid4

from tqdm import tqdm

from .artifacts import digest,dumps,file_sha256,read_json,source_snapshot,utc_now,write_json,write_jsonl
from .protocol import (AUTHOR_TOOL,JUDGE_TOOL,CONDITIONS,author_prompt,candidates,
                       construction_check,judge_prompt,load_cases,score)
from .requests import RequestRunner,request_spec
from .statistics import holm,paired_binary,wilson


def load_config(path,models=None):
    c=read_json(path)
    if type(c.get('temperature')) not in (int,float) or not math.isfinite(c['temperature']) or not 0<=c['temperature']<=2:
        raise ValueError('temperature must be between zero and two')
    for k in ('seed','workers','per_model_concurrency','requests_per_minute','tokens_per_minute',
              'max_attempts','request_timeout_seconds','max_minutes','analysis_reserve_seconds',
              'max_new_cost','total_cost_ceiling','share_cases'):
        if type(c.get(k)) not in (int,float) or not math.isfinite(c[k]) or c[k]<=0:
            raise ValueError(f'{k} must be a positive finite number')
    for k in ('workers','per_model_concurrency','requests_per_minute','tokens_per_minute','max_attempts','share_cases'):
        if type(c[k]) is not int:raise ValueError(f'{k} must be an integer')
    if c['max_attempts']>3 or c['max_minutes']>120:raise ValueError('At most 3 attempts and 120 minutes are allowed')
    if c['analysis_reserve_seconds']>=c['max_minutes']*60:raise ValueError('No time left for API calls')
    if not isinstance(c.get('judges'),list) or not c['judges']:raise ValueError('At least one judge required')
    if len({m['id'] for m in c['judges']})!=len(c['judges']):raise ValueError('Duplicate judge IDs')
    if models:
        ids=models.split(',');known={m['id'] for m in c['judges']}
        if len(set(ids))!=len(ids) or not set(ids)<=known:raise ValueError('Unknown or duplicate selected model')
        c['judges']=[m for m in c['judges'] if m['id'] in ids]
    for m in [c['author'],*c['judges']]:
        if not isinstance(m.get('id'),str) or not m['id']:raise ValueError('Model ID required')
        if type(m.get('max_output_tokens')) is not int or m['max_output_tokens']<1:raise ValueError('Invalid token budget')
        for k in ('input_per_million','output_per_million'):
            if type(m.get(k)) not in (int,float) or not math.isfinite(m[k]) or m[k]<0:raise ValueError('Invalid price estimate')
    return c


def plan(cases,config,cache):
    cached=0
    for case in cases:
        req=request_spec(config['author'],author_prompt(case),AUTHOR_TOOL,config['temperature'])
        p=Path(cache)/'requests'/(digest(req)+'.json')
        if p.exists() and read_json(p)['status']=='completed':cached+=1
    judge_calls=len(cases)*len(CONDITIONS)*len(config['judges'])
    new_calls=len(cases)-cached+judge_calls
    nominal=new_calls/config['requests_per_minute']
    planning=nominal*1.25+config['analysis_reserve_seconds']/60
    return {'topics':len(cases),'judges':len(config['judges']),'cached_rewrites':cached,
            'maximum_judge_calls':judge_calls,'calls_before_retries':new_calls,
            'rate_floor_minutes':nominal,'planning_minutes':planning,
            'max_minutes':config['max_minutes'],
            'note':'25% planning margin; retries and slow providers may exceed it. Actual deadline wins. '
                   'Existing judge caches can reduce calls. N counts source topics, not individual decisions.'}


def _batch(jobs,work,workers,title,stop=None):
    pool=ThreadPoolExecutor(max_workers=workers);futures=[];results=[]
    try:
        futures=[pool.submit(work,j) for j in jobs]
        for f in tqdm(as_completed(futures),total=len(jobs),desc=title,unit='call'):
            results.append(f.result())
    except BaseException:
        if stop is not None:stop.set()
        for f in futures:f.cancel()
        raise
    finally:pool.shutdown(wait=True,cancel_futures=True)
    return results


def run(output,cache,config,cases,dataset,provider=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    # Compute before copying so a mismatched resume never overwrites its source snapshot.
    root=Path(__file__).resolve().parent.parent
    paths=sorted((root/'pipeline').glob('*.py'))+[root/'pyproject.toml',root/'requirements.txt']
    hashes={str(p.relative_to(root)):file_sha256(p) for p in paths}
    signature=digest({'source':hashes,'config':config,'cases':cases})
    manifest_path=output/'manifest.json'
    if manifest_path.exists():
        manifest=read_json(manifest_path)
        if manifest['signature']!=signature:raise ValueError('Run inputs or source changed; choose a new output folder')
        if manifest['status']=='complete':return analyse(output)
    else:
        manifest={'protocol':'semantic-gate-audit-v1','run_id':uuid4().hex,'signature':signature,
                  'started_at':utc_now(),'status':'running','source':source_snapshot(output),
                  'dataset_sha256':file_sha256(dataset),'cases_sha256':digest(cases),'config':config,
                  'plan':plan(cases,config,cache),'sampling_unit':'source topic',
                  'primary_tests':['rewritten minus bare false acceptance','rewritten minus diff false acceptance'],
                  'statistics':'Two-sided exact paired McNemar; Holm over two contrasts x all configured judges; '
                               'source-topic paired bootstrap 95% intervals; Wilson descriptive intervals.',
                  'missing_rule':'No replacements. At least 95% complete planned pairs and N>=20 required '
                                 'for a confirmatory label; all unavailable and construction failures reported.',
                  'gold':'Single assistant reference annotations of planted commitment changes. '
                         'Automated construction guards are not complete semantic proofs or independent human review.',
                  'versions':{'python':sys.version.split()[0],**{p:importlib.metadata.version(p) for p in ('adaption','tqdm','python-dotenv')}}}
        write_json(manifest_path,manifest);write_jsonl(output/'cases.jsonl',cases)
    started=datetime.fromisoformat(manifest['started_at']).timestamp()
    remaining=max(0,started+config['max_minutes']*60-time.time())
    deadline=time.monotonic()+max(0,remaining-config['analysis_reserve_seconds'])
    runner=RequestRunner(output,cache,config,deadline,manifest['run_id'],provider=provider)
    try:
        def generate(case):
            path=output/'construction'/(case['id']+'.json')
            if path.exists():return read_json(path)
            record=runner.ask(config['author'],author_prompt(case),AUTHOR_TOOL)
            ok,reason=construction_check(case,record.get('object')) if record['status']=='completed' else (False,record.get('error',record['status']))
            row={'case_id':case['id'],'cluster':case['cluster'],'request_sha256':record['request_sha256'],
                 'eligible':ok,'reason':reason,'rewrite':record.get('object',{}).get('text')}
            write_json(path,row);return row
        built=_batch(cases,generate,config['workers'],'1/2 Construct rewrites',runner.stop)
        by_id={b['case_id']:b for b in built};jobs=[]
        for case in cases:
            for condition in CONDITIONS:
                rows=candidates(case,by_id[case['id']]['rewrite'],condition,config['seed'])
                for model in config['judges']:
                    jobs.append({'case_id':case['id'],'cluster':case['cluster'],'family':case['family'],
                                 'model':model['id'],'condition':condition,'candidates':rows,
                                 'prompt':judge_prompt(case,rows,condition),
                                 'construction_valid':by_id[case['id']]['eligible']})
        random.Random(config['seed']+1).shuffle(jobs)
        job_hash=digest(jobs);frozen=output/'jobs.json'
        if frozen.exists() and read_json(frozen)['sha256']!=job_hash:raise ValueError('Frozen jobs changed')
        write_json(frozen,{'sha256':job_hash,'jobs':jobs})  # Before any judge call.
        def judge(job):
            key=digest([job['case_id'],job['model'],job['condition']])
            path=output/'observations'/(key+'.json')
            if path.exists():return read_json(path)
            row={k:job[k] for k in ('case_id','cluster','family','model','condition')}
            if job['condition']!='bare' and not job['construction_valid']:
                row.update(status='construction_failed')
            else:
                model=next(m for m in config['judges'] if m['id']==job['model'])
                record=runner.ask(model,job['prompt'],JUDGE_TOOL)
                row.update(status=record['status'],request_sha256=record['request_sha256'])
                if record['status']=='completed':
                    try:row['judgments']=score(job['candidates'],record['object'])
                    except ValueError as exc:row.update(status='invalid_judgment',error=str(exc))
            write_json(path,row);return row
        _batch(jobs,judge,config['workers'],'2/2 Judge paired conditions',runner.stop)
        manifest.update(status='complete',finished_at=utc_now())
    except KeyboardInterrupt:
        runner.stop.set();manifest.update(status='interrupted',finished_at=utc_now())
    except BaseException:
        runner.stop.set();manifest.update(status='failed',finished_at=utc_now());raise
    finally:
        runner.close();write_json(manifest_path,manifest)
        analyse(output)
    return read_json(output/'summary.json') if (output/'summary.json').exists() else manifest


def analyse(output):
    output=Path(output);manifest=read_json(output/'manifest.json');config=manifest['config']
    cases=[read_json_line for line in (output/'cases.jsonl').read_text().splitlines()
           if line for read_json_line in [__import__('json').loads(line)]]
    n=len(cases);rows=[read_json(p) for p in (output/'observations').glob('*.json')]
    counts={};maps={};tests={}
    for model in config['judges']:
        mid=model['id'];counts[mid]={};maps[mid]={}
        for condition in CONDITIONS:
            rr=[r for r in rows if r['model']==mid and r['condition']==condition and 'judgments' in r]
            values={r['cluster']:next(j['admissible'] for j in r['judgments'] if j['kind']=='changed') for r in rr}
            if len(values)!=len(rr):raise ValueError('Duplicate source-topic observation')
            maps[mid][condition]=values;k=sum(values.values());done=len(values)
            counts[mid][condition]={'planned':n,'completed':done,'false_accepts':k,
                'rate':k/done if done else None,'wilson_95':wilson(k,done),
                'positive_accepted':sum(j['kind']=='positive' and j['admissible'] for r in rr for j in r['judgments']),
                'negative_accepted':sum(j['kind']=='negative' and j['admissible'] for r in rr for j in r['judgments']),
                'missing':n-done,'worst_case_rate_bounds':[k/n,(k+n-done)/n]}
        for name,other in [('rewording_effect','bare'),('diff_repair','diff')]:
            key=mid+':'+name;test=paired_binary(maps[mid]['rewritten'],maps[mid][other])
            test['confirmatory_eligible']=n>=20 and test['paired_n']>=math.ceil(.95*n)
            involved=[counts[mid][k] for k in ('rewritten',other)]
            test['controls_ok']=all(v['completed'] and v['positive_accepted']/v['completed']>=.90
                                   and v['negative_accepted']/v['completed']<=.05 for v in involved)
            test['confirmatory_eligible'] &= test['controls_ok']
            tests[key]=test
    adjusted=holm({k:v.get('exact_mcnemar_p',1.) for k,v in tests.items()})
    for k,v in tests.items():v['holm_p']=adjusted[k]
    call_records=[read_json(p) for p in (output/'calls').glob('*.json')]
    fresh=[r for r in call_records if r.get('origin_run_id')==manifest['run_id']]
    attempts=[a for r in fresh for a in r.get('attempts',[])]
    built=[read_json(p) for p in (output/'construction').glob('*.json')]
    metrics={'api_attempts':len(attempts),'cached_requests':len(call_records)-len(fresh),
        'input_tokens':sum((a.get('usage') or {}).get('input_tokens',0) or 0 for a in attempts),
        'output_tokens':sum((a.get('usage') or {}).get('output_tokens',0) or 0 for a in attempts),
        'reported_or_estimated_cost':sum(a.get('charge') or 0 for a in attempts),
        'unknown_cost_attempts':sum(a.get('charge') is None for a in attempts),
        'api_seconds_sum':sum(a.get('seconds',0) for a in attempts),
        'wall_seconds':(datetime.fromisoformat(manifest.get('finished_at',utc_now()))-
                        datetime.fromisoformat(manifest['started_at'])).total_seconds()}
    summary={'protocol':manifest['protocol'],'run_id':manifest['run_id'],'topics':n,
             'construction_valid':sum(b['eligible'] for b in built),'counts':counts,'paired_tests':tests,
             'metrics':metrics,'source_sha256':manifest['source']['sha256'],'gold':manifest['gold']}
    write_json(output/'summary.json',summary);write_jsonl(output/'observations.jsonl',sorted(rows,key=lambda r:(r['case_id'],r['model'],r['condition'])))
    with (output/'observations.csv').open('w',newline='') as f:
        cols=['case_id','cluster','family','model','condition','status','request_sha256','changed_accepted','positive_accepted','negative_accepted']
        writer=csv.DictWriter(f,fieldnames=cols);writer.writeheader()
        for r in sorted(rows,key=lambda r:(r['case_id'],r['model'],r['condition'])):
            flat={k:r.get(k) for k in cols[:7]}
            for j in r.get('judgments',[]):flat[j['kind']+'_accepted']=j['admissible']
            writer.writerow(flat)
    report=['# Semantic Gate Audit','',f'{n} source topics; {len(config["judges"])} judges. '
            f'{summary["construction_valid"]}/{n} rewrites passed the construction guard.','',
            '| Judge | Condition | Altered approved | Valid control approved | Denial approved | Missing |',
            '|---|---|---:|---:|---:|---:|']
    for mid,cc in counts.items():
        for condition,v in cc.items():
            report.append(f'| {mid} | {condition} | {v["false_accepts"]}/{v["completed"]} | '
                          f'{v["positive_accepted"]}/{v["completed"]} | {v["negative_accepted"]}/{v["completed"]} | {v["missing"]} |')
    report+=['','Paired contrasts compare the same source topics. Holm correction covers all configured judges '
             'and both primary contrasts. Full intervals, discordant pairs and missing-outcome bounds are in summary.json.','']
    for key,t in tests.items():
        report.append(f'- {key}: n={t["paired_n"]}, corrected p={t["holm_p"]:.6g}, '
                      f'eligible for confirmation={t["confirmatory_eligible"]}.')
    report+=['','Construction checks verify an exact unquoted assertion; they do not prove arbitrary prose semantics. '
             'Labels have single-assistant reference review. This is a curated challenge set, not a population estimate. '
             'A significant contrast alone does not establish novelty or deployment impact.','',
             f'Elapsed: {metrics["wall_seconds"]/60:.1f} min; new API attempts: {metrics["api_attempts"]}; '
             f'cached requests: {metrics["cached_requests"]}; cost recorded: {metrics["reported_or_estimated_cost"]:.6f}; '
             f'unknown-cost attempts: {metrics["unknown_cost_attempts"]}.']
    (output/'report.md').write_text('\n'.join(report)+'\n')
    export_subset(output,cases,rows,config['share_cases'])
    integrity={str(p.relative_to(output)):file_sha256(p) for p in sorted(output.rglob('*'))
               if p.is_file() and p.name not in ('integrity.json','.run.lock') and 'share' not in p.relative_to(output).parts}
    write_json(output/'integrity.json',integrity)
    return summary


def export_subset(output,cases,rows,count):
    destination=output/'share';destination.mkdir(exist_ok=True)
    selected=cases[:count];ids={c['id'] for c in selected};subset=[r for r in rows if r['case_id'] in ids]
    write_jsonl(destination/'cases.jsonl',selected);write_jsonl(destination/'observations.jsonl',subset)
    keys={r['request_sha256'] for r in subset if r.get('request_sha256')}
    for case in selected:
        p=output/'construction'/(case['id']+'.json')
        if p.exists():
            row=read_json(p);write_json(destination/'construction'/p.name,row);keys.add(row['request_sha256'])
    for key in sorted(keys):
        p=output/'calls'/(key+'.json')
        if p.exists():
            (destination/'calls').mkdir(exist_ok=True);shutil.copyfile(p,destination/'calls'/p.name)
    shutil.copyfile(output/'summary.json',destination/'full_run_summary.json')
    shutil.copyfile(output/'manifest.json',destination/'manifest.json')
    if (output/'source').exists():shutil.copytree(output/'source',destination/'source',dirs_exist_ok=True)
    (destination/'README.txt').write_text('Audit subset: first '+str(len(selected))+' frozen source topics. '
        'full_run_summary.json describes the WHOLE run, not this subset. The full results folder is required '
        'to recompute its statistics. Original prose: Name That LLM, Yao Yan, Martyna Plomecka, Ashley Oldacre, '
        'Kaggle 2026, CC BY-SA 4.0. No credentials are stored here.\n')
