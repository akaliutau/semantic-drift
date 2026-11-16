"""Small, prospectively bounded feasibility check; defaults to offline only.

python -m selftrap.maze_pilot --output runs/maze-abstraction-1
python -m selftrap.maze_pilot --output runs/maze-abstraction-1 --live

Two unassisted submissions first. If both models already produce sufficient
representations, stop for a ceiling effect. Otherwise compare generic failure
feedback with a replayable collision certificate, up to six requests in total.
All paid requests use the existing provider, throttle, receipts and 190-credit
ledger. No training, large dataset, or automatically repeated failed request.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path

from tqdm import tqdm

from .journal import AdmissionJournal
from .maze import discover, evaluate, held_out, render, verify_certificate
from .parsing import submission
from .runtime import BudgetProvider, MODELS, response_ok, usage
from .storage import digest, dumps, read_json, utc_now, write_json


MAX_CALLS = 6
CREDIT_CAP = 0.5
TOKENS = 12000
TIMEOUT = 180
MODELS_PINNED = tuple(replace(m, settings={"temperature": 0}) for m in MODELS)

TASK = """Construct the graph-node identity for a shortest-path BFS planner.
The simulator already computes every legal move and state update correctly.
The only component you supply is a small Python expression defining when two
states count as the SAME graph node. BFS keeps one visited set of these node
identifiers and does not expand a state if that identifier was already visited.

Available observations: room (current room name, a string), and has_key (bool).
Your expression may use these names, tuples, constants, comparisons, not, and,
or, and conditional expressions. No function calls, attributes, indexing,
arithmetic, imports, code blocks, or extra variables. The representation must
preserve goal status and the legal next moves, including their successor-node
identities, for every reachable state. It must work on new room names and new
layouts governed by the SAME rules. Do not hard-code the supplied room names.

Return ONLY a JSON object with node_expression (the expression as a string)
and diagnosis (one brief sentence). You are constructing a reusable state
representation, not listing a route or reimplementing the planner.
"""


def accounted(budget):
    return budget["reported_credits"] + budget["uncertain_reserve"]


def prepare(directory):
    found = discover()
    maze = found["world"]
    certificate = found["evaluation"]["certificate"]
    assert verify_certificate(maze, "room", certificate)
    spec = {
        "protocol": "key-maze-abstraction-v1",
        "models": [asdict(m) for m in MODELS_PINNED],
        "max_calls": MAX_CALLS, "incremental_credit_cap": CREDIT_CAP,
        "max_output_tokens": TOKENS, "timeout_seconds": TIMEOUT,
        "discovery_candidates": found["considered"],
        "training_world": maze.public(), "held_out_worlds": [w.public() for w in held_out()],
        "provided_bad_representation": "room", "certificate": certificate,
        "task_prompt": TASK,
        "stopping_rule": "Two cold successes: stop for ceiling. Otherwise four paired repair submissions, then stop regardless of outcome.",
        "primary_success": "No representation collisions on training and all four sealed transfer cases, and an optimal valid route on every case.",
        "feedback_control": "Same full rules and map, same room-only starter, and the fact that BFS falsely reports no route. Only the certificate arm receives the two reachable states and their distinguishing move.",
        "scope": "State-abstraction feasibility only. Correct transitions are supplied by the simulator; the key bit is an available observation. This does not test open-ended predicate invention or natural-language rule extraction.",
        "inference_limit": "Two model configurations, one mechanic, and four shared transfer cases; no population significance claim or proof of algorithmic novelty.",
        "source_hashes": {name: digest((Path(__file__).parent / name).read_text())
                          for name in ("maze.py", "maze_pilot.py", "runtime.py", "streaming.py")},
    }
    path = directory / "design.json"
    if path.exists():
        design = read_json(path)
        if design["spec_hash"] != digest(spec):
            raise ValueError("Design or source changed; use a fresh output directory")
    else:
        design = {"created_at": utc_now(), "before_api": True, "budget_before": None,
                  "spec": spec, "spec_hash": digest(spec)}
        write_json(path, design)
    offline = {"discovery_candidates": found["considered"], "bad_encoder": found["evaluation"],
               "known_repair": evaluate(maze, "(room, has_key)"),
               "held_out_bad": [evaluate(w, "room") for w in held_out()],
               "held_out_known_repair": [evaluate(w, "(room, has_key)") for w in held_out()]}
    write_json(directory / "offline.json", offline)
    render(maze, directory / "maze.png")
    return design, maze


def score(expression, maze):
    rows = [evaluate(world, expression) for world in [maze, *held_out()]]
    return {"cases": rows,
            "sufficient_cases": sum(r["sufficient_on_reachable_states"] for r in rows),
            "optimal_cases": sum(r["optimal"] for r in rows),
            "passed": all(r["sufficient_on_reachable_states"] and r["optimal"] for r in rows)}


def ask(directory, provider, model, maze, arm, design):
    result_path = directory / "trials" / model.id / f"{arm}.json"
    if result_path.exists():
        return read_json(result_path)
    prompt = TASK + "\nWorld:\n" + dumps(maze.public())
    if arm != "cold":
        prompt += ("\nCurrent representation: room\n"
                   "Observed failure: this representation makes BFS report no route, "
                   "but the maze is solvable. Repair the representation.\n")
        if arm == "certificate":
            prompt += "Mechanically checked counterexample certificate:\n" + dumps(design["spec"]["certificate"])
        else:
            prompt += "Check the rules carefully and make a reusable repair."
    requests = list((directory / "receipts").rglob("*.request.json"))
    restoring = (directory / "receipts" / model.id / f"{arm}-admission0-0.request.json").exists()
    # Exactly six planned labels, no admission retries or hidden recovery runs.
    # The shared budget counts all in-flight reservations before concurrent work.
    current = usage()
    reserve = (len(prompt.encode()) * model.input_per_million + TOKENS * model.output_per_million) / 1_000_000
    if not restoring and (len(requests) >= MAX_CALLS or accounted(current) - accounted(design["budget_before"]) + reserve > CREDIT_CAP):
        raise RuntimeError("The small-pilot call or credit guard has been reached")
    journal = AdmissionJournal(directory / "receipts" / model.id)
    row = {"model": model.id, "arm": arm, "prompt_hash": digest(prompt)}
    try:
        response = journal.complete(arm, provider, model,
                                    [{"role": "user", "content": prompt}], tools=[],
                                    max_output_tokens=TOKENS, timeout=TIMEOUT, retries=0)
        row.update(api_status=response.raw.get("status"), response_id=response.raw.get("id"),
                   served_model=response.raw.get("model"), usage=response.usage,
                   seconds=response.seconds, estimated_cost=response.estimated_cost)
        if not response_ok(response, model):
            row.update(status="api_incomplete_or_wrong_model", passed=False)
        else:
            obj, encoding = submission(response, [])
            expression = obj["node_expression"]
            row.update(status="completed", object=obj, output_encoding=encoding,
                       evaluation=score(expression, maze))
            row["passed"] = row["evaluation"]["passed"]
    except Exception as exc:
        row.update(status="request_or_submission_error", error=f"{type(exc).__name__}: {exc}", passed=False)
    write_json(result_path, row)
    return row


def parallel(jobs, function, label):
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(function, job) for job in jobs]
        return [future.result() for future in tqdm(as_completed(futures), total=len(futures), desc=label)]


def summarize(directory, design, trials, stop):
    after = usage()
    before = design["budget_before"]
    result = {"created_at": utc_now(), "stop": stop, "trials": trials,
              "new_requests": after["calls"] - before["calls"],
              "reported_credits": after["reported_credits"] - before["reported_credits"],
              "new_uncertain_reserve": after["uncertain_reserve"] - before["uncertain_reserve"],
              "input_tokens": after["input_tokens"] - before["input_tokens"],
              "output_tokens": after["output_tokens"] - before["output_tokens"],
              "budget_after": after}
    write_json(directory / "result.json", result)
    lines = ["Minimal key-maze representation pilot", "", f"Stop: {stop}",
             "One new mechanic: a reusable key opens one undirected door.",
             "The simulator supplies transitions; the model supplies graph-node identity.", ""]
    for trial in sorted(trials, key=lambda r: (r["model"], r["arm"])):
        expression = trial.get("object", {}).get("node_expression", "<none>")
        evaluation = trial.get("evaluation", {})
        lines.append(f"{trial['model']} / {trial['arm']}: {trial['status']}; expression={expression!r}; "
                     f"sufficient={evaluation.get('sufficient_cases', 0)}/5; optimal={evaluation.get('optimal_cases', 0)}/5")
    lines += ["", f"New API requests: {result['new_requests']}; reported credits: {result['reported_credits']:.8f}; "
              f"uncertain reserve: {result['new_uncertain_reserve']:.8f}.",
              "All raw prompts, responses, native usage, and timing are in receipts/.", "",
              "The room-only counterexample and the known inventory-bit repair were checked mechanically.",
              "The five scored cases share one mechanic and each submission is reused across them; they are not five independent model trials."]
    if stop == "cold_ceiling":
        lines += ["Both models already construct sufficient representations without feedback.",
                  "Certificate-versus-generic repair calls were skipped. This testbed validates the plumbing but supplies no evidence that the proposed intervention helps models discover a missing concept.",
                  "Do not enlarge this same easy-maze sample or present it as a novel blind-spot discovery."]
    elif stop == "repair_panel_complete":
        wins = []
        for model in MODELS_PINNED:
            pairs = {t["arm"]: t for t in trials if t["model"] == model.id}
            if all(a in pairs and pairs[a]["status"] == "completed" for a in ("generic", "certificate")):
                wins.append((model.id, pairs["generic"]["passed"], pairs["certificate"]["passed"]))
        lines += [f"Paired repair results (model, generic pass, certificate pass): {wins}",
                  "This is a small feasibility gate, not a statistical demonstration of general superiority."]
    else:
        lines += ["Incomplete API/submission trials are recorded separately from reasoning failures. No failed request was silently replayed."]
    (directory / "report.txt").write_text("\n".join(lines) + "\n")
    print(dumps({k: v for k, v in result.items() if k not in {"trials", "budget_after"}}), flush=True)
    return result


def run(directory, live=False):
    directory = Path(directory)
    design, maze = prepare(directory)
    if not live:
        print(dumps({"mode": "offline", "directory": str(directory), "discovery_candidates": design["spec"]["discovery_candidates"],
                     "live_call_cap": MAX_CALLS, "incremental_credit_cap": CREDIT_CAP}), flush=True)
        return
    if (directory / "result.json").exists():
        print((directory / "report.txt").read_text())
        return
    if design["budget_before"] is None:
        if list((directory / "receipts").rglob("*.request.json")):
            raise ValueError("Receipts exist without a pre-run budget snapshot")
        design["budget_before"] = usage()
        design["live_started_at"] = utc_now()
        write_json(directory / "design.json", design)
    provider = BudgetProvider(concurrency=3, per_model_concurrency=2)
    trials = []
    try:
        trials = parallel(MODELS_PINNED, lambda m: ask(directory, provider, m, maze, "cold", design), "Unassisted representations")
        if any(t["status"] != "completed" for t in trials):
            return summarize(directory, design, trials, "incomplete_cold_panel")
        if all(t["passed"] for t in trials):
            return summarize(directory, design, trials, "cold_ceiling")
        jobs = [(m, arm) for m in MODELS_PINNED for arm in ("generic", "certificate")]
        trials += parallel(jobs, lambda j: ask(directory, provider, j[0], maze, j[1], design), "Matched repair feedback")
        return summarize(directory, design, trials, "repair_panel_complete")
    finally:
        provider.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/maze-abstraction-1")
    parser.add_argument("--live", action="store_true", help="Authorize up to six actual provider requests")
    arguments = parser.parse_args()
    run(arguments.output, arguments.live)
