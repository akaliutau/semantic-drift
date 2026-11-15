"""Finish the already frozen sequence; never choose steps from effect estimates."""

from pathlib import Path
import subprocess
import sys
import time

from microstudy import DATA,ROOT,cost
from selftrap.storage import read_json,write_json,utc_now


def run(*args,output=None):
    print('Stage:', ' '.join(args),flush=True)
    subprocess.run([sys.executable,*args],cwd=ROOT,check=True,stdout=output,
                   stderr=subprocess.STDOUT if output else None)


def main():
    # The separate R5 forecast command is already running. Wait until all its
    # results and billing entries are durable before creating another client.
    path=DATA/'r5_holdout/forecasts.json'
    print('Waiting for the fixed R5 forecast batch; no new API client yet.',flush=True)
    while not path.exists():time.sleep(5)
    forecasts=read_json(path)
    assert len(forecasts)==256
    ledger=read_json(DATA/'budget.json')
    assert not any(r['status']=='in_flight' for r in ledger['calls'])
    event=f"sequence-event-{len(list((DATA/'r5_holdout').glob('sequence-event-*.json')))}.json"
    write_json(DATA/'r5_holdout'/event,{'at':utc_now(),'cost':cost()})
    # These calls are sequential processes. The budget ledger is deliberately
    # never shared by two API-running processes at the same time.
    probe=DATA/'r5_compact_visibility_probe/forecasts.json'
    if not probe.exists():run('answer_visibility.py','collect','--compact','--probe')
    usable=sum(r['status']=='valid' for r in read_json(probe))>=3
    if usable:run('answer_visibility.py','collect','--compact')
    if not (DATA/'r5_holdout/forecast_freeze.json').exists():
        run('freeze_forecasts.py','r5_holdout')
    run('binary_study.py','actors','--round','r5_holdout')
    with (DATA/'followup_analysis.log').open('w') as log:
        run('confirmatory_analysis.py',output=log)
        run('binary_study.py','analyse','--round','r5_holdout',output=log)
        run('extended_analysis.py','r5_holdout','--calibration','r4_discovery',output=log)
        if usable:run('answer_visibility.py','analyse','--compact',output=log)
        run('audit_study.py','r5_holdout',output=log)
        run('research_outputs.py','r4_discovery','r5_holdout',output=log)
        run('export_evidence.py','r1_freeform','r2_workshop','r3_binary',
            'r4_stream_probe','r4_discovery','r5_holdout','r5_answer_visibility',
            'r5_compact_visibility_probe','r5_compact_visibility',
            'service_probe_after_visibility_pause',output=log)
    primary=read_json(DATA/'r5_holdout/confirmatory_analysis.json')
    print('Frozen primary:',{k:v for k,v in primary.items() if k not in ['rows','cluster_values','reference']},flush=True)
    if usable:print('Secondary:',{k:v for k,v in read_json(DATA/'r5_compact_visibility/analysis.json').items() if k!='rows'},flush=True)
    print('Final cost:',cost(),flush=True)
    write_json(DATA/'analysis_ready.json',{'at':utc_now(),'cost':cost(),'human_report_pending':True})


if __name__=='__main__':main()
