"""Mechanically checked candidate bank for a possible adaptive workshop round."""

from pathlib import Path
import random
from time import perf_counter

from tqdm import tqdm

from selftrap.game import Board, State, DIRECTIONS, Level, canonical_id
from selftrap.generation import reverse_scramble
from selftrap.oracle import Oracle
from selftrap.storage import read_json, write_json


def static_viable(board):
    reached=set(board.goals)
    todo=list(reached)
    for r,c in todo:
        for dr,dc in DIRECTIONS.values():
            prev=(r-dr,c-dc)
            if board.floor(prev) and board.floor((r-2*dr,c-2*dc)) and prev not in reached:
                reached.add(prev); todo.append(prev)
    return reached


def small_level(rng):
    floor={(r,c) for r in range(1,5) for c in range(1,5) if rng.random()>.12}
    if len(floor)<10:return None
    boxes=frozenset(rng.sample(sorted(floor),2))
    board=Board(6,6,frozenset((r,c) for r in range(6) for c in range(6) if (r,c) not in floor),boxes)
    state=State(rng.choice(sorted(floor-boxes)),boxes)
    for _ in range(24):
        reached=board.reachable(state);pulls=[]
        for r,c in sorted(state.boxes):
            for dr,dc in DIRECTIONS.values():
                box,player=(r-dr,c-dc),(r-2*dr,c-2*dc)
                if box in reached and board.floor(player) and player not in state.boxes:
                    pulls.append(State(player,(state.boxes-{(r,c)})|{box}))
        if not pulls:break
        state=rng.choice(pulls)
    return Level(board,state) if not board.solved(state) else None


def make_bank(path, count=64, seed=2026111510, small=False):
    path=Path(path)
    records=read_json(path) if path.exists() else []
    seen={r['id'] for r in records}
    seed_index=max((r['seed_index'] for r in records), default=-1)+1
    started=perf_counter()
    with tqdm(total=count,initial=len(records),desc='Verified nonstatic-trap bank',unit='puzzle') as bar:
        while len(records)<count:
            level=small_level(random.Random(seed+seed_index)) if small else reverse_scramble(random.Random(seed+seed_index),steps=32)
            current_index=seed_index
            seed_index+=1
            if level is None:continue
            identity=canonical_id(level)
            if identity in seen:continue
            solved=Oracle(level).search(timeout=.5)
            low,high=(3,8) if small else (6,18)
            if solved.status!='SOLVED' or not low<=len(solved.plan)<=high:continue
            viable=static_viable(level.board)
            if not level.state.boxes<=viable:continue
            actions=level.board.legal_pushes(level.state)
            classified=[]
            for action in actions:
                after=level.board.push(level.state,action)
                result=Oracle(Level(level.board,after)).search(timeout=.3)
                classified.append({'action':action.to_dict(),'status':result.status,
                                   'nonstatic':after.boxes<=viable,'states':result.states})
            bad=[r for r in classified if r['status']=='UNSOLVABLE' and r['nonstatic']]
            good=[r for r in classified if r['status']=='SOLVED']
            if not bad or not good or any(r['status']=='UNKNOWN' for r in classified):continue
            # Prefer an apparently helpful deadlock: moving onto/closer to a goal.
            def temptation(record):
                p=record['action']['box']; dr,dc=DIRECTIONS[record['action']['direction']]
                dest=(p[0]+dr,p[1]+dc)
                distance=lambda q:min(abs(q[0]-g[0])+abs(q[1]-g[1]) for g in level.board.goals)
                return (dest in level.board.goals,distance(tuple(p))-distance(dest))
            bad.sort(key=temptation,reverse=True)
            record={'id':identity,'seed_index':current_index,'board':level.board.render(level.state),
                    'solution':[a.to_dict() for a in solved.plan],
                    'optimal_pushes':len(solved.plan),'predicted_first_push':bad[0]['action'],
                    'actions':classified,'temptation':list(temptation(bad[0]))}
            # Oracle witnesses always replay through the independent grid implementation.
            if not level.replay(solved.plan)['completed']:raise RuntimeError('Checker disagreement')
            records.append(record);seen.add(identity)
            write_json(path,records);bar.update()
    print({'puzzles':len(records),'seeds_examined':seed_index,'seconds':perf_counter()-started})


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('output');p.add_argument('--count',type=int,default=64)
    p.add_argument('--seed',type=int,default=2026111510)
    p.add_argument('--small',action='store_true')
    a=p.parse_args();make_bank(a.output,a.count,a.seed,a.small)
