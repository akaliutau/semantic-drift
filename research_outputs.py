"""Export compact evidence tables and a standalone scientific figure."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from microstudy import DATA,ROOT
from selftrap.storage import read_json


def export(names):
    destination=ROOT/'reports';destination.mkdir(exist_ok=True)
    rows=[];cal=[]
    for name in names:
        result=read_json(DATA/name/'extended_analysis.json')
        for model,arms in result['coverage'].items():
            for arm,counts in arms.items():
                completed=sum(counts.values())-counts.get('service_failure',0)
                rows.append({'round':name,'model':model,'arm':arm,
                    'n':sum(counts.values()),'safe':counts.get('safe',0),
                    'deadlock':counts.get('deadlock',0),'exhausted':counts.get('exhausted',0),
                    'malformed':counts.get('malformed',0),
                    'service_failure':counts.get('service_failure',0),
                    'safe_rate_observed':counts.get('safe',0)/completed if completed else ''})
        cal.extend({'round':name,**r} for r in result['calibration_summary'] if r['author']==r['target'])
    with (destination/'actor_counts.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    forecast_rows=[]
    for r in cal:
        gap=r['forecast_minus_observed']
        forecast_rows.append({k:r[k] for k in ['round','target','n','mean_p_safe','observed_safe_rate','brier']}
            |{'gap':gap['effect'],'ci_low':gap['ci'][0],'ci_high':gap['ci'][1],'exploratory_p':gap['p']})
    with (destination/'forecast_calibration.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(forecast_rows[0]));writer.writeheader();writer.writerows(forecast_rows)
    fig,axes=plt.subplots(1,len(names),figsize=(5.2*len(names),3.8),squeeze=False)
    for ax,name in zip(axes.flat,names):
        result=read_json(DATA/name/'extended_analysis.json');subset=[]
        for target in result['models']:
            matched=[r for r in result['calibration_rows'] if r['target']==target]
            subset.append({'target':target,'n':len(matched),
                'mean_p_safe':sum(r['self_p'] for r in matched)/len(matched),
                'observed_safe_rate':sum(r['actual_safe'] for r in matched)/len(matched)})
        labels=[r['target'].replace('qwen3.5-397b','Qwen').replace('gemma-4-31b-it','Gemma')+f"\n(n={r['n']})" for r in subset]
        x=list(range(len(subset)))
        ax.bar([v-.18 for v in x],[r['mean_p_safe'] for r in subset],.36,label='Prospective P(safe)',color='#4477aa')
        ax.bar([v+.18 for v in x],[r['observed_safe_rate'] for r in subset],.36,label='Observed safe choice',color='#ee7733')
        ax.set_xticks(x,labels);ax.set_ylim(0,1);ax.set_ylabel('Probability / frequency')
        ax.set_title('Discovery' if name=='r4_discovery' else 'Fresh holdout');ax.spines[['top','right']].set_visible(False)
        ax.legend(fontsize=8,loc='upper center')
    fig.suptitle('Forecasted versus observed safe submissions, on matched worlds',fontsize=10)
    fig.tight_layout();fig.savefig(destination/'prospective_forecasts.png',dpi=180)
    fig.savefig(destination/'prospective_forecasts.pdf');plt.close(fig)
    print('Wrote actor_counts.csv, forecast_calibration.csv, prospective_forecasts.png/.pdf')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('names',nargs='+')
    export(parser.parse_args().names)
