"""Commit-sized aggregate evidence; full API receipts stay in local runs/study."""

import argparse
import shutil

from microstudy import DATA,ROOT,cost
from selftrap.storage import digest,read_json,utc_now,write_json


FILES=['manifest.json','protocol.txt','authored.json','forecasts.json','actors.json','actors-baseline.json',
       'analysis.json','extended_analysis.json','forecast_freeze.json',
       'confirmatory_analysis.json','frozen_reference.json','before_api.json',
       'source_at_freeze.json',
       'exploratory_position_audit.json',
       'service_interruption.json','partial_forecasts_at_pause.json','results.json',
       'transport_recovery.json','transport_recovery_protocol.txt','transport_recovery_source.json',
       'routing_plan.json','routing_analysis.json',
       'precommitted_selections.json','warning_reviews.json','integrity_audit.json',
       'calibration.json','stopped_before_actors.json',
       'r1_format_audit.json','r2_format_audit_final.json',
       'r2_safe_forecast_audit.json','r2_random_baseline.json']


def export(names):
    destination=ROOT/'reports/evidence';destination.mkdir(parents=True,exist_ok=True)
    index={}
    for filename in ['feasibility_summary.json','r1_format_audit.json',
        'r2_format_audit_final.json','r2_safe_forecast_audit.json',
        'r2_random_baseline.json','budget.json','analysis_ready.json']:
        path=DATA/filename
        if path.exists():
            shutil.copyfile(path,destination/filename)
            index[filename]=digest(path.read_text())
    for name in names:
        source=DATA/name;target=destination/name;target.mkdir(exist_ok=True)
        for filename in FILES:
            path=source/filename
            if not path.exists():continue
            shutil.copyfile(path,target/filename)
            index[name+'/'+filename]=digest(path.read_text())
        archive=source/'rate_limit_attempt0'
        if archive.exists():
            saved=target/'rate_limit_attempt0';saved.mkdir(exist_ok=True)
            for filename in FILES+['analysis.log']:
                path=archive/filename
                if path.is_file():
                    shutil.copyfile(path,saved/filename)
                    index[name+'/rate_limit_attempt0/'+filename]=digest(path.read_text())
    write_json(destination/'index.json',{'at':utc_now(),'files':index,'budget':cost(),
        'raw_receipts':str(DATA),'note':'Full prompts, responses and errors remain in the local research worktree. Research branch only; do not merge.'})
    print('Exported',len(index),'aggregate evidence files; no API calls')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('names',nargs='+')
    export(parser.parse_args().names)
