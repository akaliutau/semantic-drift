"""Offline integrity checks; does not inspect treatment efficacy or call APIs."""

import argparse
from datetime import datetime

from binary_study import prompt
from microstudy import DATA
from selftrap.storage import digest, read_json, write_json


def audit(name):
    directory=DATA/name
    manifest=read_json(directory/'manifest.json')
    forecasts=read_json(directory/'forecasts.json')
    freeze=read_json(directory/'forecast_freeze.json')
    assert freeze['manifest_hash']==digest(manifest)
    assert freeze['forecasts_hash']==digest(forecasts)
    assert freeze['review_hash']==digest(read_json(directory/'warning_reviews.json'))
    jobs=read_json(directory/'actor_manifest.json')
    assert len({j['id'] for j in jobs})==len(jobs)
    board_ids=[p['id'] for p in manifest['puzzles']]
    assert len(set(board_ids))==len(board_ids)
    # Compare exact actor input to a reconstruction, so labels, warning, and
    # omission of the supplied gold witness can be checked without grading it.
    inspected=0
    for job in jobs:
        path=directory/'actors'/job['id']/'input.json'
        if not path.exists():continue
        p=job['puzzle'];seed=int(digest([p['id'],manifest['seed']])[:10],16)
        text,choices=prompt(p,seed,job['warning'])
        assert read_json(path)=={'prompt':text,'choices':choices}
        assert 'winning_plan' not in text and 'certified_deadlocking' not in text
        rejected=directory/'rate_limit_attempt0'/'actors'/job['id']/'input.json'
        if rejected.exists():assert read_json(rejected)==read_json(path)
        for request in (path.parent/'calls').glob('*.request.json'):
            assert datetime.fromisoformat(read_json(request)['created_at'])>=datetime.fromisoformat(freeze['at'])
        inspected+=1
    result={'round':name,'puzzles':len(board_ids),'fixed_actor_jobs':len(jobs),
            'actor_inputs_checked':inspected,'freeze_hashes_match':True,'actor_calls_after_freeze':True,
            'recovery_inputs_equal_rejected_inputs':(directory/'rate_limit_attempt0').exists(),
            'efficacy_outcomes_inspected':False}
    write_json(directory/'integrity_audit.json',result)
    print(result)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('name')
    audit(parser.parse_args().name)
