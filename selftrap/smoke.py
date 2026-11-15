"""Exactly one journalled API request; never starts a research batch."""

import argparse
from pathlib import Path
from tqdm import tqdm
from .journal import AdmissionJournal
from .runtime import BudgetProvider,MODELS,response_ok,usage
from .storage import dumps


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--live',action='store_true');p.add_argument('--model',default=MODELS[0].id)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if not args.live:p.error('Add --live to make one API request')
    model=next(m for m in MODELS if m.id==args.model);provider=BudgetProvider()
    try:
        with tqdm(total=1,desc='API check') as bar:
            r=AdmissionJournal(args.output).complete('smoke',provider,model,
                'Return the JSON object {"ok":true}.',tools=[],max_output_tokens=1024,
                timeout=90,retries=0,tool_choice='none');bar.update()
        print(dumps({'valid_response':response_ok(r,model),'text':r.text,'seconds':r.seconds,'usage':r.usage,'budget':usage()}))
    finally:provider.close()


if __name__=='__main__':main()
