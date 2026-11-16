"""Bounded cold screen for model-invented planning abstractions/pruning lemmas.

The exact planner is the reference. A model supplies a cheap keep(state) filter:
False deletes a search state, so rejecting any state with a short completion
is unsound. Always-True is allowed and scored as no acceleration, not a failure.
Later self/peer prediction and prospective repair phases require a new seal.
"""

import argparse
import ast
from dataclasses import asdict
from pathlib import Path
import threading
from time import perf_counter

from tqdm import tqdm

from .journal import AdmissionJournal
from .maze_pilot import parallel
from .parsing import submission
from .planning_lab import Arena, RULES, make_world, records, render, verify
from .programs import run_generator, validate_source
from .providers import ModelSpec
from .runtime import BudgetProvider, usage, response_ok
from .storage import digest, dumps, read_json, utc_now, write_json
from .two_directions import MODELS_PINNED


GAMES = ("dots", "ricochet", "numberlink")
TOKENS = {"gemma-4-31b-it": 10000, "qwen3.5-397b": 24000, "glm-5.2": 24000,
          "qwen3.8-27b": 10000, "deepseek-v4-flash": 10000, "glm-5.3-flash": 10000}
GLM_PAIR = (MODELS_PINNED[1], ModelSpec("glm-5.2", input_per_million=1.4,
                                    output_per_million=4.4, settings={"temperature": 0}))
TASK = """Invent a CHEAP, reusable state-pruning lemma for the supplied game.
An exact graph-search planner already exists. It checks goals first, generates
legal successors correctly and decrements moves_left on every action. We want
to skip hopeless branches more cheaply, while preserving EVERY solution within
the remaining move budget. All observations and rules are supplied accurately.

Write pure Python def generate(batch): returning a list of bools, one per
observation. True = keep exploring; False = safely discard because NO completion
within moves_left exists. A state that could finish MUST return True. True does
not assert solvability. Returning True everywhere is allowed but adds no useful
pruning. A simple safe rule is preferable to a fast unsound one. Do NOT enumerate
future action sequences or implement the full solution search inside the filter.
Cheap graph reachability, bounds, components and other derived features are fine.
The game and suffix budget define the contract; intermediates are not goals.

Cells use row*width+column. Directions are orthogonal. A wall [a,b] blocks that
adjacent edge in both directions. In dots, board.numbers lists numbered cells
in order; state.next_number is the next label, 1-based, or null after the last.
In Numberlink, state.pair_index is zero-based; state.used includes all drawn
paths and the current starting endpoint. Other remaining endpoints are still
reserved. In ricochet, state.robots[0] is the target robot and others are helpers.

Keep source <=6000 characters. Helper functions, for loops, indexing, sets,
lists, dicts, comprehensions, comparisons, + - * // % and standard collection
methods are supported. Builtins include range,len,min,max,sum,abs,int,str,bool,
list,tuple,dict,set,enumerate,zip,sorted,all,any,reversed. NO imports, inspect,
random, files, model calls, classes, while, bitwise operations or exponentiation.
For a queue use a list and a for loop; it may append while iterating.
Return ONLY JSON with code, mechanism (brief), safety_argument (brief), and
p_sound: your probability in [0,1] that this EXACT code never falsely discards
a completable state on fresh boards under these rules. No claim of proof follows
from passing a few examples. Fresh layouts, dimensions and partial paths follow.
"""


def run_filter(code, observations):
    if not isinstance(code, str) or len(code) > 6000:
        raise ValueError("Pruner source must be <=6000 characters")
    validate_source(code)
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.Import) or isinstance(node, ast.Name) and node.id in {"inspect", "random"}:
            raise ValueError("No imports, inspector or random in a pruning lemma")
    output = []
    for start in range(0, len(observations), 128):
        result = run_generator(code, observations[start:start+128])["value"]
        if not isinstance(result, list) or len(result) != len(observations[start:start+128]) or any(type(v) is not bool for v in result):
            raise ValueError("generate(batch) must return one bool per input")
        output += result
    return output


def score(code, rows):
    started = perf_counter()
    predictions = run_filter(code, [r["observation"] for r in rows])
    cases = [{"id": r["id"], "keep": keep, "can_finish": r["can_finish"],
              "unsafe_rejection": not keep and r["can_finish"]} for r, keep in zip(rows, predictions)]
    return {"cases": cases, "unsafe": sum(r["unsafe_rejection"] for r in cases),
            "dead_states_pruned": sum(not r["keep"] and not r["can_finish"] for r in cases),
            "kept": sum(predictions), "count": len(rows), "seconds": perf_counter()-started}


def prepare(directory, models=MODELS_PINNED, reuse_from=None):
    settings = [("train", 11, 4, 4), ("train", 17, 4, 4),
                ("discovery", 41, 4, 4), ("discovery", 43, 4, 4),
                ("confirmation", 101, 4, 5), ("confirmation", 107, 4, 5),
                ("repair_transfer", 211, 5, 4), ("repair_transfer", 223, 5, 4)]
    spec = {"protocol": "three-game-pruner-cold-v1", "models": [asdict(m) for m in models],
            "games": GAMES, "seeds": settings, "rules": RULES, "task": TASK,
            "tokens": TOKENS, "seconds": 300, "max_calls": 12, "incremental_credit_cap": 2.0,
            "budget_before": None, "source_hashes": {n: digest((Path(__file__).parent / n).read_text()) for n in
                ("concept_pilot.py", "planning_lab.py", "programs.py", "runtime.py", "streaming.py", "storage.py")},
            "stops": "Six cold authors, at most one construction-only repair each. No resampling a valid program after an unsafe result. Always-keep and correct simple bounds are valid baselines, not new concepts.",
            "promise_gate": "A cold error alone never passes. A subsequent prospectively sealed stage must test self-specific error predictions against a peer with identical evidence, pinpoint the omitted distinction mechanically, and compare useful repairs on untouched boards. No claim of novel predicate invention, universal safety, or statistical significance from this screen.",
            "limits": "A pruner is a Boolean abstraction of bounded completion, not a complete solver or an exact bisimulation. Record panels sample reachable states; absence of a counterexample is not a proof of soundness."}
    if reuse_from is not None:
        references = []
        for game in GAMES:
            for model in models:
                for label in ("cold", "construction-repair"):
                    previous = Path(reuse_from) / "responses" / game / model.id / f"{label}.json"
                    if previous.exists():
                        response = read_json(previous)
                        if response["status"] == "completed":
                            references.append({"game": game, "model": model.id, "label": label,
                                               "path": str(previous.resolve()), "hash": digest(response)})
        spec["reused_authorings"] = references
    path = directory / "design.json"
    if path.exists():
        design = read_json(path)
        if design["spec_hash"] != digest(spec):
            raise ValueError("Source or protocol changed; use a fresh directory")
        return design
    panels = {g: [] for g in GAMES}
    for game, config in tqdm([(g, c) for g in GAMES for c in settings], desc="Build exact small game panels"):
        split, seed, width, height = config
        world = make_world(game, seed, width, height)
        arena = Arena(world)
        rows = records(arena, 32)
        assert rows and all(verify(r["certificate"]) for r in rows if r["can_finish"])
        panels[game].append({"split": split, "seed": seed, "world": asdict(world),
                             "reachable_states": len(arena.parent), "rows": rows})
        if split == "train" and seed == 11:
            directory.mkdir(parents=True, exist_ok=True)
            render(world, world.initial, directory / f"{game}.png")
    for game in GAMES:
        seen = set()
        for panel in panels[game]:
            panel["rows"] = [r for r in panel["rows"] if r["id"] not in seen]
            seen.update(r["id"] for r in panel["rows"])
            if len(panel["rows"]) < 8:
                raise ValueError("Too few distinct probe observations in a sealed panel")
    design = {"at": utc_now(), "before_api": True, "budget_before": None,
              "spec_hash": digest(spec), "spec": spec, "panels": panels}
    write_json(path, design)
    return design


class Calls:
    def __init__(self, directory, design, provider):
        self.directory, self.design, self.provider = directory, design, provider
        self.lock = threading.Lock()
        self.count = len(list((directory / "receipts").rglob("*.request.json")))

    def ask(self, model, game, label, prompt):
        path = self.directory / "responses" / game / model.id / f"{label}.json"
        if path.exists():
            row = read_json(path)
            if row["prompt_hash"] != digest(prompt):
                raise ValueError("Cached prompt changed")
            return row
        for reference in self.design.get("spec", {}).get("reused_authorings", []):
            if (reference["game"], reference["model"], reference["label"]) != (game, model.id, label):
                continue
            row = read_json(reference["path"])
            if digest(row) != reference["hash"] or row["prompt_hash"] != digest(prompt):
                raise ValueError("Reused author response or prompt differs")
            row = {**row, "reused_from": reference["path"]}
            write_json(path, row)
            return row
        receipts = self.directory / "receipts" / game / model.id
        restoring = (receipts / f"{label}-admission0-0.request.json").exists()
        with self.lock:
            if not restoring:
                now, old = usage(), self.design["budget_before"]
                spent = now["reported_credits"] + now["uncertain_reserve"] - old["reported_credits"] - old["uncertain_reserve"]
                reserve = 2*((len(prompt.encode())+4096)*model.input_per_million + TOKENS[model.id]*model.output_per_million)/1e6
                if self.count >= 12 or spent + reserve > 2:
                    raise RuntimeError("Cold-screen cap reached")
                self.count += 1
        row = {"model": model.id, "game": game, "label": label, "prompt_hash": digest(prompt)}
        try:
            response = AdmissionJournal(receipts).complete(label, self.provider, model,
                [{"role": "user", "content": prompt}], tools=[], tool_choice="none",
                max_output_tokens=TOKENS[model.id], timeout=300, retries=0)
            row.update(api_status=response.raw.get("status"), served_model=response.raw.get("model"),
                       seconds=response.seconds, usage=response.usage)
            if response_ok(response, model):
                obj, encoding = submission(response, [])
                row.update(status="completed", object=obj, encoding=encoding)
            else:
                row["status"] = "incomplete_or_wrong_model"
        except Exception as exc:
            row.update(status="request_or_submission_error", error=f"{type(exc).__name__}: {exc}")
        write_json(path, row)
        return row


def author(calls, model, game, design):
    examples = []
    for panel in design["panels"][game]:
        if panel["split"] == "train":
            for truth in (True, False):
                examples += [{"observation": r["observation"], "can_finish": truth} for r in panel["rows"] if r["can_finish"] == truth][:2]
    prompt = TASK + "\nGame rules:\n" + RULES[game] + "\nLabeled examples:\n" + dumps(examples)
    response = calls.ask(model, game, "cold", prompt)
    for attempt in range(2):
        if response["status"] != "completed":
            return response
        try:
            obj = response["object"]
            if type(obj.get("p_sound")) not in {int, float} or not 0 <= obj["p_sound"] <= 1:
                raise ValueError("Need p_sound in [0,1]")
            evaluation = score(obj["code"], [{"observation": e["observation"], "can_finish": e["can_finish"], "id": str(i)} for i,e in enumerate(examples)])
            return {**response, "construction_valid": True, "training": evaluation}
        except (ValueError, KeyError, TypeError, SyntaxError) as exc:
            if attempt:
                return {**response, "construction_valid": False, "construction_error": str(exc)}
            response = calls.ask(model, game, "construction-repair", prompt + "\nPrevious source:\n" + str(response["object"].get("code", "")) + "\nRuntime/interface error, NOT an oracle correctness label: " + str(exc) + "\nFix construction only; retain your pruning principle.")
    raise AssertionError("Unreachable")


def run(directory, game=None, live=False, *, models=MODELS_PINNED, reuse_from=None):
    directory = Path(directory)
    design = prepare(directory, models, reuse_from)
    if not live:
        print(dumps({"mode": "offline", "games": GAMES, "max_calls": 12, "credit_cap": 2}))
        return
    if design["budget_before"] is None:
        if list((directory / "receipts").rglob("*.request.json")):
            raise ValueError("Existing receipts lack a pre-run budget snapshot")
        design["budget_before"] = usage()
        write_json(directory / "design.json", design)
    provider = BudgetProvider(concurrency=3, per_model_concurrency=2)
    calls = Calls(directory, design, provider)
    try:
        for selected in ([game] if game else GAMES):
            result_path = directory / f"{selected}-cold.json"
            if result_path.exists():
                continue
            authors = parallel(models, lambda m: author(calls, m, selected, design), f"Invent {selected} pruning rules")
            for row in authors:
                if not row.get("construction_valid"):
                    continue
                evaluated = []
                for panel in design["panels"][selected]:
                    if panel["split"] != "discovery":
                        continue
                    try:
                        result = score(row["object"]["code"], panel["rows"])
                        evaluated.append({"seed": panel["seed"], "status": "completed", **result})
                    except (ValueError, TypeError, KeyError, SyntaxError) as exc:
                        evaluated.append({"seed": panel["seed"], "status": "runtime_failure", "error": str(exc)})
                row["discovery"] = evaluated
            write_json(result_path, {"at": utc_now(), "authors": authors, "budget_now": usage()})
            print(dumps({"game": selected, "authors": [{"model": r["model"], "status": r["status"],
                         "construction_valid": r.get("construction_valid"), "p_sound": r.get("object", {}).get("p_sound"),
                         "discovery": [{"seed": e["seed"], "status": e["status"], "unsafe": e.get("unsafe"), "dead_pruned": e.get("dead_states_pruned")} for e in r.get("discovery", [])]} for r in authors]}), flush=True)
    finally:
        provider.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/three-game-concepts-1"))
    parser.add_argument("--game", choices=GAMES)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--pair", choices=("qwen", "glm"), default="qwen")
    parser.add_argument("--reuse-from", type=Path)
    args = parser.parse_args()
    run(args.output, args.game, args.live, models=GLM_PAIR if args.pair == "glm" else MODELS_PINNED,
        reuse_from=args.reuse_from)
