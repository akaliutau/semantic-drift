"""Seal prospective forecasts and source before any actor outcomes exist."""

import argparse
from pathlib import Path

from microstudy import DATA
from selftrap.storage import digest, read_json, utc_now, write_json


def freeze(name):
    directory=DATA/name
    if list((directory/'actors').glob('*/result.json')) or list((directory/'actors').glob('*/calls/*.request.json')):
        raise ValueError('Actor outcomes already exist; cannot claim a prospective freeze')
    manifest=read_json(directory/'manifest.json')
    forecasts=read_json(directory/'forecasts.json')
    reviews=read_json(directory/'warning_reviews.json')
    expected={(p['id'],m) for p in manifest['puzzles'] for m in manifest['models']}
    actual={(f['puzzle_id'],f['author']) for f in forecasts}
    if actual!=expected or len(forecasts)!=len(expected):
        raise ValueError('Forecast cohort is incomplete or duplicated')
    # Review records are retained separately. This script seals them; it does
    # not invent a human review or certify that a warning is causally correct.
    source={p.name:p.read_text() for p in Path(__file__).parent.glob('*.py')}
    record={'at':utc_now(),'actor_outcomes_seen':0,'manifest_hash':digest(manifest),
            'forecasts_hash':digest(forecasts),'review_hash':digest(reviews),
            'source_hash':digest(source),'forecast_count':len(forecasts)}
    lookup={(f['puzzle_id'],f['author']):f for f in forecasts}
    record['forecast_eligible_worlds']=sum(all(lookup[(p['id'],t)]['status']=='valid'
        for t in manifest['models']) for p in manifest['puzzles'])
    protocol=directory/'protocol.txt'
    if protocol.exists():record['protocol_hash']=digest(protocol.read_text())
    path=directory/'forecast_freeze.json'
    if path.exists():raise ValueError('A freeze already exists; do not overwrite it')
    write_json(directory/'source_at_freeze.json',source)
    write_json(path,record)
    print(record)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('name')
    freeze(parser.parse_args().name)
