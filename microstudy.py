"""Isolated exploratory code; never merge into the benchmark implementation."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import json
from pathlib import Path
import random
import threading
import time

from tqdm import tqdm

from selftrap.game import Level, Push
from selftrap.generation import ValidationConfig, author_slot
from selftrap.providers import AdaptionProvider, ModelSpec
from selftrap.solver import SolverConfig, run_episode
from selftrap.storage import digest, dumps, read_json, utc_now, usage_summary, write_json

ROOT = Path(__file__).resolve().parent
DATA = ROOT/"runs/study"
ENV_FILE = Path("/media/DATA/repos-research/apart1/.env")
MODELS = [ModelSpec("qwen3.5-397b", input_per_million=.55, output_per_million=3.5),
          ModelSpec("glm-5.2", input_per_million=1.4, output_per_million=4.4),
          ModelSpec("gemma-4-31b-it", input_per_million=.09, output_per_million=.34)]
RATE_LIMIT_RETRIES = 2  # Admission rejection only; never retry a model answer.
SOLVER = SolverConfig(max_output_tokens=4000, wall_seconds=90)


class BudgetStop(RuntimeError): pass


class BudgetProvider:
    """One persistent guard across ALL exploratory and confirmatory rounds."""
    def __init__(self,concurrency=2):
        self.client = AdaptionProvider(ENV_FILE)
        self.lock = threading.Lock()
        self.admission_lock = threading.Lock()
        self.next_request_at = time.monotonic()+65  # Clear the previous rate-limit window.
        self.semaphores = {m.id: threading.Semaphore(concurrency) for m in MODELS}
        self.path = DATA/"budget.json"
        self.ledger = read_json(self.path) if self.path.exists() else {
            "hard_ceiling":190, "operational_ceiling":160, "calls":[]}

    def complete(self, model, messages, *, tools, max_output_tokens, **kwargs):
        sem = self.semaphores.setdefault(model.id, threading.Semaphore(2))
        with sem:
            with self.admission_lock:
                delay=self.next_request_at-time.monotonic()
                while delay>0:
                    time.sleep(min(delay,1));delay=self.next_request_at-time.monotonic()
                self.next_request_at=time.monotonic()+3  # <=20 starts/minute across deployments.
            reservation = 2*((len(dumps([messages,tools]).encode())+4096)*model.input_per_million
                             +max_output_tokens*model.output_per_million)/1e6
            with self.lock:
                held = sum(c["charged"] if c["charged"] is not None else c["reserved"] for c in self.ledger["calls"])
                if held+reservation > self.ledger["operational_ceiling"]:
                    raise BudgetStop("Study-wide conservative credit limit reached")
                entry = {"index":len(self.ledger["calls"]),"at":utc_now(),"model":model.id,
                         "request_hash":digest([messages,tools,max_output_tokens]),
                         "reserved":reservation,"charged":None,"status":"in_flight"}
                self.ledger["calls"].append(entry)
                write_json(self.path,self.ledger)
            started=time.perf_counter()
            try:
                response=self.client.complete(model,messages,tools=tools,max_output_tokens=max_output_tokens,**kwargs)
            except Exception as exc:
                if getattr(exc,'status_code',None)==429:
                    with self.admission_lock:
                        self.next_request_at=max(self.next_request_at,time.monotonic()+65)
                with self.lock:
                    entry.update(status="error",error=str(exc),seconds=time.perf_counter()-started)
                    write_json(self.path,self.ledger)
                raise
            usage=response.usage or {}
            charge=usage.get("cost")
            if not isinstance(charge,(int,float)) or charge < 0: charge=response.estimated_cost
            with self.lock:
                entry.update(status="response",charged=charge,usage=usage,seconds=response.seconds)
                write_json(self.path,self.ledger)
            return response

    def close(self): self.client.close()


def cost():
    if not (DATA/"budget.json").exists(): return {"calls":0,"known_credits":0,"reserved_unknown":0}
    ledger=read_json(DATA/"budget.json")
    calls=ledger["calls"]
    return {"calls":len(calls),"known_credits":sum(c["charged"] or 0 for c in calls),
            "reserved_unknown":sum(c["reserved"] for c in calls if c["charged"] is None),
            "input_tokens":sum((c.get("usage") or {}).get("input_tokens",0) for c in calls),
            "output_tokens":sum((c.get("usage") or {}).get("output_tokens",0) for c in calls)}


def parallel(function, jobs, workers=12, desc="Jobs"):
    results=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(function,job):job for job in jobs}
        for future in tqdm(as_completed(futures),total=len(futures),desc=desc):
            results.append(future.result())
    return results


def feasibility():
    p=read_json(ROOT/"examples/tiny_puzzle.json")
    provider=BudgetProvider()
    try:
        def job(model):
            result=run_episode(provider,model,Level.from_rows(p["board"]),DATA/"feasibility"/model.id,
                bad_push=Push.from_dict(p["predicted_first_push"]),config=SOLVER,force_first_simulation=True)
            return {"model":model.id,"status":result["status"],"success":result["full_success"],
                    "tools":result["tool_calls"],"seconds":result["invocation_seconds"]}
        results=parallel(job,MODELS,workers=3,desc="Model feasibility")
        write_json(DATA/"feasibility_summary.json",results)
        print(dumps({"results":results,"cost":cost()}))
    finally: provider.close()


def author_round(name, slots_per_cell=1, min_pushes=6):
    directory=DATA/name
    cfg={"name":name,"kind":"freeform_sokoban","slots_per_cell":slots_per_cell,
         "roster":[asdict(m) for m in MODELS],"solver":asdict(SOLVER),
         "min_pushes":min_pushes,"seed":2026111501}
    path=directory/"manifest.json"
    if path.exists() and read_json(path)!=cfg: raise ValueError("Round changed; choose a new name")
    write_json(path,cfg)
    jobs=[]
    for a,author in enumerate(MODELS):
        for t,target in enumerate(MODELS+[None]):
            for n in range(slots_per_cell):
                jobs.append({"id":f"a{a}-t{t}-n{n}","author":author.id,
                             "target":target.id if target else None,"index":n})
    random.Random(cfg["seed"]).shuffle(jobs)
    write_json(directory/"assignments.json",jobs)
    provider=BudgetProvider()
    try:
        def job(slot):
            author=next(m for m in MODELS if m.id==slot["author"])
            result=author_slot(provider,author,MODELS,slot["target"],directory/"authors"/slot["id"],
                validation=ValidationConfig(min_pushes=min_pushes),solver=SOLVER,max_output_tokens=8000)
            return {"slot":slot, "result":result}
        results=parallel(job,jobs,desc=f"Author/{name}")
        # Assignment order, not completion order, determines duplicate acceptance.
        by_id={r["slot"]["id"]:r for r in results}
        seen=set()
        accepted=[]
        for slot in jobs:
            item=by_id[slot["id"]]
            r=item["result"]
            item["deduplicated_valid"]=bool(r["valid"] and r["puzzle_id"] not in seen)
            if item["deduplicated_valid"]: seen.add(r["puzzle_id"])
            accepted.append(item)
        write_json(directory/"authored.json",accepted)
        print(dumps({"round":name,"valid":sum(r["deduplicated_valid"] for r in accepted),
                     "slots":len(accepted),"failures":[{"slot":r["slot"]["id"],"reason":r["result"].get("reason")}
                     for r in accepted if not r["deduplicated_valid"]],"cost":cost()}))
    finally: provider.close()


def solve_round(name,repeats=2):
    directory=DATA/name
    authored=read_json(directory/"authored.json")
    jobs=[]
    for item in authored:
        if not item["deduplicated_valid"]: continue
        for model in MODELS:
            for repeat in range(repeats):
                jobs.append({"slot":item["slot"],"model":model.id,"repeat":repeat,
                             "id":f"{item['slot']['id']}-{model.id}-r{repeat}"})
    path=directory/"solve_manifest.json"
    random.Random(2026111502).shuffle(jobs)
    if path.exists() and read_json(path)!=jobs: raise ValueError("Solve manifest changed")
    write_json(path,jobs)
    by_slot={x["slot"]["id"]:x for x in authored}
    provider=BudgetProvider()
    try:
        def job(j):
            p=by_slot[j["slot"]["id"]]["result"]["proposal"]
            model=next(m for m in MODELS if m.id==j["model"])
            seed=int(digest([j["slot"]["id"],j["repeat"]])[:10],16)
            result=run_episode(provider,model,Level.from_rows(p["board"]),directory/"solvers"/j["id"],
                bad_push=Push.from_dict(p["predicted_first_push"]),config=SOLVER,seed=seed)
            return {"job":j,"result":result}
        results=parallel(job,jobs,desc=f"Solve/{name}")
        write_json(directory/"solved.json",results)
        summary={m.id:{"n":sum(r["job"]["model"]==m.id for r in results),
                      "solved":sum(r["job"]["model"]==m.id and bool(r["result"]["full_success"]) for r in results),
                      "predicted_errors":sum(r["job"]["model"]==m.id and bool(r["result"]["predicted_error"]) for r in results)} for m in MODELS}
        print(dumps({"round":name,"summary":summary,"cost":cost()}))
    finally: provider.close()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("stage",choices=["feasibility","author","solve","cost"])
    p.add_argument("--round",default="r1_freeform")
    p.add_argument("--slots",type=int,default=1)
    p.add_argument("--min-pushes",type=int,default=6)
    p.add_argument("--repeats",type=int,default=2)
    args=p.parse_args()
    if args.stage=="feasibility": feasibility()
    elif args.stage=="author": author_round(args.round,args.slots,args.min_pushes)
    elif args.stage=="solve": solve_round(args.round,args.repeats)
    else: print(dumps(cost()))


if __name__=="__main__": main()
