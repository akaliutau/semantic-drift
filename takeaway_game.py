"""Optional alternative substrate, offline only until explicitly scheduled.

Counterfactual take-away twins: identical pile and offered moves, changed rules,
opposite winning first moves. Recursive and bottom-up minimax must agree.
"""

from functools import lru_cache
import random

from selftrap.storage import digest,write_json


def solve(world):
    moves=world['moves_by_phase'];poison=set(world['poisoned_remainders'])
    misere=world['taking_last']=='loses';period=len(moves)
    @lru_cache(None)
    def win(n,phase):
        if n==0:return misere
        return any(outcome(n,phase,k) for k in moves[phase] if k<=n)
    def outcome(n,phase,k):
        remaining=n-k
        if remaining>0 and remaining in poison:return False
        return not win(remaining,(phase+1)%period)
    table=[]
    for n in range(world['stones']+1):
        for phase in range(period):
            table.append({'stones':n,'phase':phase,'winning':win(n,phase)})
    good=[k for k in moves[0] if k<=world['stones'] and outcome(world['stones'],0,k)]
    return good,table


def independent(world):
    # Different traversal, no recursion or solve() calls.
    schedule=world['moves_by_phase'];period=len(schedule)
    values=[[world['taking_last']=='loses']*period]
    for count in range(1,world['stones']+1):
        row=[]
        for phase in range(period):
            possible=[]
            for take in schedule[phase]:
                if take>count:continue
                remainder=count-take
                if remainder and remainder in world['poisoned_remainders']:possible.append(False)
                else:possible.append(not values[remainder][(phase+1)%period])
            row.append(any(possible))
        values.append(row)
    good=[]
    for k in schedule[0]:
        remaining=world['stones']-k
        if remaining<0 or remaining>0 and remaining in world['poisoned_remainders']:continue
        if not values[remaining][1%period]:good.append(k)
    return good


def generate_pairs(count=72,seed=2026111560):
    rng=random.Random(seed);pairs=[];seen=set();seen_worlds=set();families=['last_rule','phase_rule','poison_rule']
    attempts=0
    while len(pairs)<count:
        attempts+=1
        if attempts>100000:raise RuntimeError('Not enough distinct counterfactual pairs')
        # Reserve rare last-rule twins before other families can consume them.
        family=families[min(len(pairs)//((count+2)//3),2)]
        opening=sorted(rng.sample(range(1,8),2));n=rng.randint(7,40)
        base={'stones':n,'moves_by_phase':[opening],'taking_last':'wins','poisoned_remainders':[]}
        changed={**base}
        if family=='last_rule':changed['taking_last']='loses'
        elif family=='phase_rule':
            changed['moves_by_phase']=[opening,sorted(rng.sample(range(1,8),2))]
        else:changed['poisoned_remainders']=sorted(rng.sample(range(1,n),rng.choice([1,2,3])))
        initial,t0=solve(base);shifted,t1=solve(changed)
        if len(initial)!=1 or len(shifted)!=1 or initial==shifted:continue
        key=digest([base,changed])
        ids={digest(base),digest(changed)}
        if key in seen or ids&seen_worlds:continue
        assert independent(base)==initial and independent(changed)==shifted
        seen.add(key);seen_worlds.update(ids)
        cards=[]
        for kind,world,good,table in [('base',base,initial,t0),('shift',changed,shifted,t1)]:
            bad=next(k for k in opening if k not in good)
            cards.append({'id':digest(world),'kind':'takeaway','pair_id':key,'family':family,'condition':kind,
                          'world':world,'solution':[{'take':good[0]}],
                          'predicted_first_push':{'take':bad},'proof_table':table})
        pairs.append(cards)
    rng.shuffle(pairs)
    return [card for pair in pairs for card in pair]


def rules(world):
    return '''Two players alternate removing stones from one pile. You move now in
phase zero. The allowed removals depend on the phase, which advances cyclically
after EVERY player's turn. Each player follows the same schedule. Taking the
last stone has the stated wins/loses rule. Leaving a positive poisoned remainder
immediately loses. If no legal move exists, the current player loses. Your
opponent plays optimally. Terminal outcomes take precedence; the no-legal-move
rule applies only when stones remain. Choose a move that guarantees a win.
Return only JSON {"choice":"A"} or {"choice":"B"}. No tools are available.
Give your best decision even if uncertain; a complete proof is not required.'''


if __name__=='__main__':
    from pathlib import Path
    cards=generate_pairs()
    write_json('runs/study/takeaway_bank_offline_only.json',cards)
    print('Offline only:',len(cards),'verified cards; no API calls')
