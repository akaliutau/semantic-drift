"""Offline counterexample search over complete small reachable graphs.

Expands the cold screen's sampled states, NOT its unseen confirmation boards.
Every unsafe rejection carries a replayable winning completion. Counts describe
a finite correlated graph/budget set, not independent statistical trials.
"""

import argparse
from pathlib import Path
from time import perf_counter

from tqdm import tqdm

from .concept_pilot import GAMES, run_filter
from .planning_lab import Arena, restore, verify
from .storage import digest, dumps, read_json, write_json


BUDGETS = (1, 2, 3, 5, 8, 16, 32)


def audit_world(world, code, *, budgets=BUDGETS, excluded=(), show_progress=False):
    arena = Arena(world)
    excluded = set(excluded)
    depth = {}
    for state, parent in arena.parent.items():
        depth[state] = 0 if parent is None else depth[parent[0]] + 1
    counts = {"checked": 0, "winning": 0, "unsafe": 0, "dead_pruned": 0, "kept": 0}
    examples = {}
    started = perf_counter()
    batch = []

    def evaluate(items):
        keep_flags = run_filter(code, [r["observation"] for r in items])
        for record, keep in zip(items, keep_flags):
            state, budget = record["state"], record["budget"]
            win = state in arena.distance and arena.distance[state] <= budget
            counts["checked"] += 1
            counts["winning"] += win
            counts["kept"] += keep
            counts["dead_pruned"] += not keep and not win
            if not keep and win:
                counts["unsafe"] += 1
                key = (depth[state] + arena.distance[state], budget, record["id"])
                if state not in examples and (len(examples) < 8 or key < max(e["rank"] for e in examples.values())):
                    certificate = arena.certificate(state, budget)
                    assert verify(certificate)
                    examples[state] = {"id": record["id"], "rank": key, "certificate": certificate}
                    if len(examples) > 8:
                        worst = max(examples, key=lambda s: examples[s]["rank"])
                        del examples[worst]

    states = [s for s in arena.parent if not world.goal(s)]
    for state in tqdm(states, desc=f"Audit {world.game} reachable states", disable=not show_progress):
        base = world.observe(state)
        for budget in budgets:
            observation = {**base, "moves_left": budget}
            ident = digest(observation)[:12]
            if ident in excluded:
                continue
            batch.append({"state": state, "budget": budget, "id": ident, "observation": observation})
            if len(batch) == 128:
                evaluate(batch)
                batch = []
    if batch:
        evaluate(batch)
    return {**counts, "reachable_states": len(arena.parent), "budgets": list(budgets),
            "seconds": perf_counter()-started, "examples": sorted(examples.values(), key=lambda r: r["rank"])}


def run(directory, game=None):
    directory = Path(directory)
    design = read_json(directory / "design.json")
    for selected in ([game] if game else GAMES):
        source = directory / f"{selected}-cold.json"
        if not source.exists():
            continue
        authors = read_json(source)["authors"]
        seen = set()
        for panel in design["panels"][selected]:
            if panel["split"] == "train":
                for truth in (True, False):
                    seen.update(r["id"] for r in [r for r in panel["rows"] if r["can_finish"] == truth][:2])
        for author in authors:
            if not author.get("construction_valid"):
                continue
            code = author["object"]["code"]
            target = directory / "audits" / selected / f"{author['model']}.json"
            if target.exists():
                if read_json(target)["code_hash"] != digest(code):
                    raise ValueError("Audited code changed")
                continue
            evaluations = []
            for panel in design["panels"][selected]:
                if panel["split"] != "discovery":
                    continue
                try:
                    result = audit_world(restore(panel["world"]), code, excluded=seen, show_progress=True)
                    evaluations.append({"seed": panel["seed"], "status": "completed", **result})
                except (ValueError, KeyError, TypeError, SyntaxError) as exc:
                    evaluations.append({"seed": panel["seed"], "status": "runtime_failure", "error": str(exc)})
            write_json(target, {"model": author["model"], "game": selected, "code_hash": digest(code),
                                "source_authoring": str(source), "evaluations": evaluations})
            print(dumps({"game": selected, "model": author["model"], "evaluations": [
                {k:v for k,v in e.items() if k not in {"examples", "budgets"}} for e in evaluations]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/three-game-concepts-3"))
    parser.add_argument("--game", choices=GAMES)
    args = parser.parse_args()
    run(args.output, args.game)
