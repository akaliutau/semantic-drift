"""python -m pipeline {plan,run,analyze}."""

import argparse
from pathlib import Path

from .artifacts import dumps,file_lock
from .protocol import load_cases
from .study import analyse,load_config,plan,run


def main():
    parser=argparse.ArgumentParser(description='Paired semantic-preservation audit; no more than two hours per run.')
    sub=parser.add_subparsers(dest='command',required=True)
    for command in ('plan','run'):
        p=sub.add_parser(command)
        p.add_argument('-n','--n',type=int,default=30,help='Independent source topics (1–40 in bundled dataset).')
        p.add_argument('--config',type=Path,default=Path('config.json'))
        p.add_argument('--dataset',type=Path,default=Path('data/cases.jsonl'))
        p.add_argument('--cache',type=Path,default=Path('cache'))
        p.add_argument('--models',help='Comma-separated judge IDs from config; default: all four.')
        p.add_argument('--minutes',type=float,help='Total run budget, at most 120 minutes.')
        if command=='run':p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('analyze');p.add_argument('output',type=Path)
    args=parser.parse_args()
    try:
        if args.command=='analyze':
            summary=analyse(args.output)
        else:
            config=load_config(args.config,args.models)
            if args.minutes is not None:
                if not 0<args.minutes<=120:raise ValueError('--minutes must be between 0 and 120')
                if args.minutes*60<=config['analysis_reserve_seconds']:raise ValueError('Budget is smaller than the analysis reserve')
                config['max_minutes']=args.minutes
            cases=load_cases(args.dataset,args.n,config['seed']);forecast=plan(cases,config,args.cache)
            print(dumps(forecast),flush=True)
            if args.command=='plan':return
            if forecast['planning_minutes']>config['max_minutes']:
                raise ValueError('Plan exceeds time budget; reduce N or selected models before starting')
            with file_lock(args.cache/'.pipeline.lock',blocking=False),file_lock(args.output/'.run.lock',blocking=False):
                summary=run(args.output,args.cache,config,cases,args.dataset)
        print(dumps({'topics':summary.get('topics'),'metrics':summary.get('metrics'),
                     'output':str(args.output.resolve())}))
    except (ValueError,FileNotFoundError,BlockingIOError) as exc:
        parser.exit(2,f'Error: {exc}\n')


if __name__=='__main__':main()
