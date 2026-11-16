"""Small, dependency-free statistics and measured run sizing.

Counts are source clusters, never individual rewrites or repeated judge calls.
Exact paired tests and multiplicity correction are kept separate from bootstrap
effect intervals. Missing outcomes must be reported by callers.
"""

from math import comb, sqrt
import random
from statistics import mean, median


def wilson(k, n, z=1.959963984540054):
    if not 0 <= k <= n:
        raise ValueError('Invalid binomial count')
    if n == 0:
        return None
    p = k/n
    d = 1+z*z/n
    center = (p+z*z/(2*n))/d
    half = z*sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [max(0.,center-half), min(1.,center+half)]


def paired_binary(a, b, *, seed=20261116, samples=10000):
    """a,b: mappings from distinct source-cluster ID to a binary outcome."""
    ids = sorted(a.keys() & b.keys())
    if not ids:
        return {'paired_n':0}
    if any(type(v) not in (bool,int) or v not in (0,1) for m in (a,b) for v in m.values()):
        raise ValueError('Paired exact test requires one binary observation per source cluster')
    gains = sum(a[i] and not b[i] for i in ids)
    losses = sum(b[i] and not a[i] for i in ids)
    d = gains+losses
    p = min(1.,2*sum(comb(d,j) for j in range(min(gains,losses)+1))/2**d) if d else 1.
    delta = [int(a[i])-int(b[i]) for i in ids]
    rng = random.Random(seed)
    boot = sorted(mean(rng.choices(delta,k=len(delta))) for _ in range(samples))
    return dict(paired_n=len(ids),a_only=gains,b_only=losses,
                a_rate=mean(a[i] for i in ids),b_rate=mean(b[i] for i in ids),
                risk_difference=mean(delta),bootstrap_95=[boot[int(.025*samples)],boot[int(.975*samples)-1]],
                exact_mcnemar_p=p,missing_a=len(b.keys()-a.keys()),missing_b=len(a.keys()-b.keys()))


def holm(pvalues):
    ordered=sorted(pvalues,key=pvalues.get);out={};previous=0.
    for i,name in enumerate(ordered):
        previous=max(previous,min(1.,(len(ordered)-i)*pvalues[name]))
        out[name]=previous
    return out


def percentile(values,p):
    if not values:return None
    values=sorted(values);x=(len(values)-1)*p
    lo=int(x);hi=min(lo+1,len(values)-1)
    return values[lo]+(values[hi]-values[lo])*(x-lo)


def forecast(summaries, calls, *, interval=6., analysis_minutes=30, safety=1.25):
    """Measured estimates, not a runtime guarantee or power guarantee."""
    useful=[s for s in summaries if s.get('wall_seconds') and
            len(s.get('results',[]))==s.get('accounting',{}).get('responses')]
    durations=[r['seconds'] for s in useful for r in s['results'] if r.get('seconds') is not None]
    attempts=sum(s['accounting']['responses'] for s in useful)
    seconds=sum(s['wall_seconds'] for s in useful)
    if not attempts or not durations:raise ValueError('Need a completed timing pilot')
    observed_per_call=seconds/attempts
    cost=sum(s['accounting']['receipt_estimated_credits'] for s in useful)/attempts
    per_call=max(interval,observed_per_call)
    return dict(calls=calls,measured_calls=attempts,measured_seconds=seconds,
                measured_calls_per_minute=60/observed_per_call,
                api_median_seconds=median(durations),api_p90_seconds=percentile(durations,.9),
                nominal_run_minutes=calls*per_call/60,
                planning_run_minutes=(calls*per_call*safety+percentile(durations,.9))/60,
                analysis_minutes=analysis_minutes,
                planning_total_minutes=(calls*per_call*safety+percentile(durations,.9))/60+analysis_minutes,
                estimated_credits=calls*cost,planning_credits=calls*cost*2,
                caveat='25% runtime and 2x cost margins are planning assumptions; upstream errors can exceed them. '
                       'Bootstrap/Wilson intervals do not establish novelty or sampling representativeness.')
