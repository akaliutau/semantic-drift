"""Prospective optimization stage: improve a sound baseline, never force errors.

Training examples are drawn only from training layouts. The baseline's kept
but impossible states are NOT called mistakes. A novel self-specific research
claim still requires prospective forecasts and useful controlled repair.
"""

import argparse
from dataclasses import asdict
from pathlib import Path

from . import concept_pilot as cold
from .concept_sampling_retry import MODEL as QWEN
from .planning_lab import Arena, RULES, make_world, records, restore
from .runtime import BudgetProvider, usage
from .storage import digest, dumps, read_json, utc_now, write_json


MODELS = (cold.MODELS_PINNED[1], QWEN)
TASK = """Improve the supplied planning filter by adding ONE useful, cheap,
reusable pruning principle. Its existing budget/reachability checks are a
conservative baseline. A kept-but-impossible state is allowed by the baseline
contract; it is an opportunity for acceleration, NOT an incorrect answer.

The examples below show both completable and impossible states that the
baseline keeps. Try to distinguish more impossible states while preserving
EVERY within-budget completion, including on unseen boards and dimensions.
Do not merely return False everywhere, copy example coordinates, change the
game's goal, or turn a necessary condition into a complete solvability test.
You may keep the baseline if you cannot justify an additional safe rule.
There is no penalty for honestly declining an unsafe optimization, but it
does not count as a newly useful pruning concept.

Return the SAME executable JSON contract as below: complete source, mechanism,
safety_argument, and p_sound for this exact source. Explain the additional
principle briefly. Source must run within the given syntax/compute interface.
"""


def examples(game, baseline, panels):
    selected = []
    sources = []
    candidates = [(p["seed"], restore(p["world"])) for p in panels if p["split"] == "train"]
    candidates += [(s, make_world(game, s)) for s in (503, 509, 521, 523)]
    for seed, world in candidates:
        arena = Arena(world)
        rows = records(arena, 128)
        keep = cold.run_filter(baseline, [r["observation"] for r in rows])
        pool = [r for r,k in zip(rows,keep) if k]
        positive = [r for r in pool if r["can_finish"]][:2]
        negative = [r for r in pool if not r["can_finish"]][:2]
        if len(positive) < 2 or len(negative) < 2:
            continue
        selected += [{"observation": r["observation"], "can_finish": r["can_finish"]} for r in positive + negative]
        sources.append({"seed": seed, "world": world.public(), "example_ids": [r["id"] for r in positive+negative]})
        if len(sources) == 2:
            break
    return selected, sources


def prepare(directory, previous):
    old = read_json(previous / "design.json")
    tasks = {}
    refs = {str((previous / "design.json").resolve()): digest(old)}
    for game in cold.GAMES:
        path = previous / f"{game}-cold.json"
        source = read_json(path)
        author = next(r for r in source["authors"] if r["model"] == "gemma-4-31b-it")
        if not author.get("construction_valid"):
            raise ValueError("Need the valid fixed baseline")
        code = author["object"]["code"]
        data, provenance = examples(game, code, old["panels"][game])
        tasks[game] = {"baseline": code, "examples": data, "training_sources": provenance,
                       "eligible": len(provenance) == 2}
        refs[str(path.resolve())] = digest(source)
    spec = {"protocol": "three-game-safe-optimization-v1", "models": [asdict(m) for m in MODELS],
            "task": TASK, "interface": cold.TASK, "tasks": tasks, "references": refs,
            "tokens": cold.TOKENS, "max_calls": 12, "incremental_credit_cap": 2,
            "acceptance": "Optimization alone is not the research goal. Seek actual unsafe pruning under a claimed safety contract on at least two discovery layouts. Then seal self/peer forecasts on fresh cases and compare useful repairs against matched controls before claiming promise. A correct conservative program, a declined optimization, a syntax failure, or a timeout never counts as a blind spot.",
            "source_hashes": {n: digest((Path(__file__).parent / n).read_text()) for n in
                              ("concept_improve.py", "concept_pilot.py", "planning_lab.py", "programs.py", "runtime.py")}}
    path = directory / "design.json"
    if path.exists():
        design = read_json(path)
        if design["spec_hash"] != digest(spec):
            raise ValueError("Optimization stage changed")
        return design
    design = {"at": utc_now(), "before_api": True, "budget_before": None,
              "spec_hash": digest(spec), "spec": spec, "panels": old["panels"]}
    write_json(path, design)
    return design


def author(calls, model, game, task):
    prompt = TASK + "\nExecutable interface:\n" + cold.TASK + "\nRules:\n" + RULES[game]
    prompt += "\nBaseline source:\n" + task["baseline"] + "\nTraining cases kept by that baseline:\n" + dumps(task["examples"])
    response = calls.ask(model, game, "improve", prompt)
    for attempt in range(2):
        if response["status"] != "completed":
            return response
        try:
            obj = response["object"]
            if type(obj.get("p_sound")) not in {int,float} or not 0 <= obj["p_sound"] <= 1:
                raise ValueError("Need p_sound in [0,1]")
            rows = [{"id": str(i), **r} for i,r in enumerate(task["examples"])]
            evaluated = cold.score(obj["code"], rows)
            return {**response, "construction_valid": True, "training": evaluated}
        except (ValueError, KeyError, TypeError, SyntaxError) as exc:
            if attempt:
                return {**response, "construction_valid": False, "construction_error": str(exc)}
            response = calls.ask(model, game, "improve-construction-repair", prompt + "\nPrevious source:\n"
                + str(response["object"].get("code", "")) + "\nConstruction error only (no correctness feedback): " + str(exc)
                + "\nRepair syntax/runtime only; retain the pruning principle.")
    raise AssertionError("Unreachable")


def run(directory, previous, game=None, live=False, only_model=None):
    directory, previous = Path(directory), Path(previous)
    design = prepare(directory, previous)
    if not live:
        print(dumps({"mode": "offline", "eligible": {g:t["eligible"] for g,t in design["spec"]["tasks"].items()}}))
        return
    if design["budget_before"] is None:
        if list((directory / "receipts").rglob("*.request.json")):
            raise ValueError("Missing pre-run budget snapshot")
        design["budget_before"] = usage()
        write_json(directory / "design.json", design)
    provider = BudgetProvider(concurrency=3, per_model_concurrency=2)
    calls = cold.Calls(directory, design, provider)
    try:
        for selected in ([game] if game else cold.GAMES):
            target = directory / f"{selected}-cold.json"
            existing = read_json(target).get("authors", []) if target.exists() else []
            completed_models = {a["model"] for a in existing}
            pending = [m for m in MODELS if m.id not in completed_models and (only_model is None or m.id == only_model)]
            if not pending:
                continue
            task = design["spec"]["tasks"][selected]
            if not task["eligible"]:
                write_json(target, {"at": utc_now(), "stop": "no_unresolved_training_examples", "authors": []})
                continue
            authors = cold.parallel(pending, lambda m: author(calls,m,selected,task), f"Improve {selected} abstraction")
            for result in authors:
                if not result.get("construction_valid"):
                    continue
                result["discovery"] = []
                for panel in design["panels"][selected]:
                    if panel["split"] == "discovery":
                        try:
                            evaluation = cold.score(result["object"]["code"],panel["rows"])
                            result["discovery"].append({"seed": panel["seed"], "status": "completed", **evaluation})
                        except (ValueError, KeyError, TypeError, SyntaxError) as exc:
                            result["discovery"].append({"seed": panel["seed"], "status": "runtime_failure", "error": str(exc)})
            write_json(target, {"at": utc_now(), "authors": existing + authors, "budget_now": usage()})
            print(dumps({"game": selected, "authors": [{"model": a["model"], "status": a["status"],
                "valid": a.get("construction_valid"), "p_sound": a.get("object",{}).get("p_sound"),
                "discovery": [{k:v for k,v in e.items() if k!='cases'} for e in a.get("discovery",[])]} for a in authors]}), flush=True)
    finally:
        provider.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/three-game-improve-1"))
    parser.add_argument("--previous", type=Path, default=Path("runs/three-game-concepts-3"))
    parser.add_argument("--game", choices=cold.GAMES)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--only-model", choices=[m.id for m in MODELS])
    args = parser.parse_args()
    run(args.output,args.previous,args.game,args.live,args.only_model)
