"""Two small, gated pilots: certified false rejection and executable self-models.

Offline preparation is the default. --live authorizes at most ten requests
and one incremental credit, within the ORIGINAL shared budget and throttle.
The expected base cost is four calls: two reviewer panels, two self-surrogates.
Peer authors and confirmation panels run only if a concrete candidate exists.
"""

import argparse
import ast
from dataclasses import asdict, replace
from pathlib import Path
import threading

from .journal import AdmissionJournal
from .maze_pilot import parallel
from .parsing import submission
from .programs import run_generator, validate_source
from .runtime import BudgetProvider, MODELS, response_ok, usage
from .sortnet import RULES, certify, discovery_panel, execute, pool, render
from .storage import digest, dumps, read_json, utc_now, write_json


MODELS_PINNED = tuple(replace(m, settings={"temperature": 0}) for m in MODELS)
LABELS = {"VALID", "INVALID", "UNKNOWN"}
MAX_CALLS = 10
CREDIT_CAP = 1.0
TOKENS = 12000
SECONDS = 180
ACTOR = """Judge each four-slot compare-and-swap program below independently.
Determine whether it always sorts. These are not necessarily familiar sorting
algorithms. Use the precise supplied rules, including directed index pairs.
Do not assume that all programs have the same answer. If you conclude INVALID,
provide one four-element input on which the FINAL output is not sorted. Use
UNKNOWN if you cannot determine the verdict. No execution tools are available.
Return ONLY JSON: {"rows":[{"id":"...","verdict":"VALID|INVALID|UNKNOWN",
"counterexample":null or [four integers]}]}. Include every supplied ID once.
"""

AUTHOR = """Write an executable behavioral surrogate for the TARGET reviewer.
Predict the reviewer's actual verdicts, INCLUDING ITS MISTAKES; predicting the
mathematically correct answer alone is not the objective. Both self and peer
authors receive the same target observations. There is no claim that you can
copy weights. We test whether one frozen small program predicts useful new
target-specific failures more cheaply than repeated model queries.

Write pure Python def generate(panel): returning {case_id: verdict_string} for
every case. panel is a list of {"id":str,"steps":[[i,j],...]}, exactly the
reviewer's input format. Values must be VALID, INVALID, or UNKNOWN. Match the
observed target behavior, not your preferred ideal behavior. Keep source below
6000 characters. You may define helper functions. No imports, model calls,
files, network, random, inspect, classes, while, exponentiation or bitwise ops.
Use for loops, ordinary collections, indexing, comparisons, + - * // %,
len/range/list/dict/sorted/all/any. Standard list/dict methods are available.
For binary enumeration, digits can be obtained with (mask // divisor) % 2
for divisor in (8,4,2,1); no unsupported syntax is needed.

One frozen program will be applied to new programs without calling any model.
It must reproduce at least 3 of the 4 observed verdicts to enter the search.
You cannot see the held-out pool. We search it for your predicted mistakes,
then test selected cases on a fresh target. A constant negative guess is not
evidence of self-knowledge. Peer, exact-checker and constant baselines matter.
Return ONLY JSON: {"code":"...","hypothesis":"one brief sentence"}.
"""


def charged(budget):
    return budget["reported_credits"] + budget["uncertain_reserve"]


def prepare(directory, *, tokens=TOKENS, seconds=SECONDS, reuse_discovery=None):
    if not 1 <= tokens <= 24000 or not 0 < seconds <= 300:
        raise ValueError("Pilot limits: at most 24000 output tokens and 300 seconds")
    discovery = discovery_panel()
    hidden = pool(101, 12, 12, exclude=[r["public"]["id"] for r in discovery["cases"]])
    spec = {
        "protocol": "truth-and-selfclone-v2", "models": [asdict(m) for m in MODELS_PINNED],
        "max_calls": MAX_CALLS, "incremental_credit_cap": CREDIT_CAP,
        "tokens": tokens, "seconds": seconds, "rules": RULES,
        "actor_prompt": ACTOR, "author_prompt": AUTHOR,
        "discovery": discovery, "held_out": hidden,
        "stops": ["API/submission failure is not a reasoning error; stop dependent work for that target. Completed targets continue independently.",
                  "No cold false rejection: stop direction 2 discovery; no larger adaptive search.",
                  "No eligible self-surrogate predicts an error in the sealed pool: stop self-surrogate search.",
                  "If candidates exist, obtain peer surrogates and at most one fresh confirmation panel per target.",
                  "At most one construction-only repair per self author. Never repair after held-out target feedback."],
        "case_selection": "Shortest predicted-error program, then ID; at most one self and one peer selection per target. Confirmation forecasts are recomputed on the exact final panel before querying the target.",
        "source_hashes": {n: digest((Path(__file__).parent / n).read_text()) for n in
                          ("sortnet.py", "two_directions.py", "programs.py", "runtime.py", "streaming.py")},
        "limits": "Four verdicts in one request are correlated; this is a feasibility screen. The proof certificate is retained for researchers but is shown to the target only in selected confirmation cases. Initial false rejection measures verification error, not irrational refusal of visible proof.",
    }
    if reuse_discovery is not None:
        previous = Path(reuse_discovery)
        original = read_json(previous / "design.json")["spec"]
        for key in ("models", "actor_prompt", "rules", "discovery"):
            if original[key] != spec[key]:
                raise ValueError(f"Discovery reuse changes {key}")
        reused = []
        for model in MODELS_PINNED:
            saved = previous / "responses" / model.id / "discovery.json"
            if not saved.exists():
                continue
            response = read_json(saved)
            if response["status"] == "completed":
                reused.append({"model": model.id, "path": str(saved.resolve()),
                               "hash": digest(response)})
        spec["reused_discovery"] = reused
    path = directory / "design.json"
    if path.exists():
        design = read_json(path)
        if design["spec_hash"] != digest(spec):
            raise ValueError("Design changed; use a new directory")
    else:
        design = {"at": utc_now(), "before_api": True, "budget_before": None,
                  "spec_hash": digest(spec), "spec": spec}
        write_json(path, design)
    unusual = next(r for r in discovery["cases"] if r["certificate"]["valid"]
                   and any(i > j for i, j in r["public"]["steps"]))
    render(unusual, directory / "sorting-program.png")
    return design


class Requests:
    """Small additional cap; provider still enforces global pacing and credits."""
    def __init__(self, directory, design, provider):
        self.directory, self.design, self.provider = directory, design, provider
        self.lock = threading.Lock()
        self.new_admissions = 0
        self.existing = len(list((directory / "receipts").rglob("*.request.json")))

    def ask(self, model, label, prompt):
        result_path = self.directory / "responses" / model.id / f"{label}.json"
        if result_path.exists():
            return read_json(result_path)
        if label == "discovery":
            for reference in self.design["spec"].get("reused_discovery", []):
                if reference["model"] != model.id:
                    continue
                row = read_json(reference["path"])
                if digest(row) != reference["hash"] or row["prompt_hash"] != digest(prompt):
                    raise ValueError("Reused discovery receipt or prompt changed")
                row = {**row, "reused_from": reference["path"]}
                write_json(result_path, row)
                return row
        tokens, seconds = self.design["spec"]["tokens"], self.design["spec"]["seconds"]
        receipts = self.directory / "receipts" / model.id
        # Journal uses Path.with_suffix; a dot in a model-derived label would
        # otherwise truncate both the label and a construction-repair suffix.
        receipt_label = label.replace(".", "_")
        restoring = (receipts / f"{receipt_label}-admission0-0.request.json").exists()
        with self.lock:
            if not restoring:
                spent = charged(usage()) - charged(self.design["budget_before"])
                reserve = (len(prompt.encode()) * model.input_per_million + tokens * model.output_per_million) / 1_000_000
                if self.existing + self.new_admissions >= MAX_CALLS or spent + reserve > CREDIT_CAP:
                    raise RuntimeError("Small-pilot guard reached")
                self.new_admissions += 1
        row = {"model": model.id, "label": label, "prompt_hash": digest(prompt)}
        try:
            response = AdmissionJournal(receipts).complete(receipt_label, self.provider, model,
                [{"role": "user", "content": prompt}], tools=[], max_output_tokens=tokens,
                timeout=seconds, retries=0)
            row.update(api_status=response.raw.get("status"), served_model=response.raw.get("model"),
                       seconds=response.seconds, usage=response.usage, response_id=response.raw.get("id"))
            if response_ok(response, model):
                obj, encoding = submission(response, [])
                row.update(status="completed", object=obj, encoding=encoding)
            else:
                row["status"] = "api_incomplete_or_wrong_model"
        except Exception as exc:
            row.update(status="request_or_submission_error", error=f"{type(exc).__name__}: {exc}")
        write_json(result_path, row)
        return row


def score_actor(obj, cases):
    expected = {r["public"]["id"]: r for r in cases}
    rows = obj["rows"]
    if len(rows) != len(expected) or {r["id"] for r in rows} != set(expected):
        raise ValueError("Need exactly one verdict for every case")
    scored = []
    for row in rows:
        if row["verdict"] not in LABELS:
            raise ValueError("Unknown verdict label")
        case = expected[row["id"]]
        truth = "VALID" if case["certificate"]["valid"] else "INVALID"
        witness = row.get("counterexample")
        witness_valid = (isinstance(witness, list) and len(witness) == 4
                         and all(type(v) is int and abs(v) <= 100000 for v in witness)
                         and execute(case["public"]["steps"], witness) != sorted(witness))
        scored.append({**row, "truth": truth, "correct": row["verdict"] == truth,
                       "false_rejection": truth == "VALID" and row["verdict"] == "INVALID",
                       "false_acceptance": truth == "INVALID" and row["verdict"] == "VALID",
                       "counterexample_valid": witness_valid})
    return scored


def actor(requests, model, cases, label):
    panel = [r["public"] for r in cases]
    proof_note = ("\nA supplied certificate contains every binary input and the exact final output. "
                  "The threshold argument explains why these cover arbitrary integers. "
                  "Take this additional evidence into account.\n") if any("certificate" in p for p in panel) else ""
    result = requests.ask(model, label, ACTOR + "\n" + RULES + proof_note + "\nPrograms:\n" + dumps(panel))
    if result["status"] == "completed":
        try:
            result["scored"] = score_actor(result["object"], cases)
        except (ValueError, KeyError, TypeError) as exc:
            result.update(status="invalid_submission", error=str(exc))
    return result


def clone_predictions(code, panel):
    if not isinstance(code, str) or len(code) > 6000:
        raise ValueError("Code exceeds 6000 characters")
    validate_source(code)
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.Import) or isinstance(node, ast.Name) and node.id in {"inspect", "random"}:
            raise ValueError("No imports, random, or inspector in a behavioral surrogate")
    result = run_generator(code, panel)["value"]
    if (not isinstance(result, dict) or set(result) != {r["id"] for r in panel}
            or any(v not in LABELS for v in result.values())):
        raise ValueError("generate(panel) must map every case ID to a verdict string")
    return result


def author(requests, model, target, observations, discovery, kind):
    prompt = AUTHOR + "\n" + dumps({"you": model.id, "target": target.id, "actor_prompt": ACTOR,
                                    "rules": RULES, "observed_programs": [r["public"] for r in discovery],
                                    "observations": observations})
    label = f"{kind}-clone-for-{target.id}"
    result = requests.ask(model, label, prompt)
    for attempt in range(2 if kind == "self" else 1):
        if result["status"] != "completed":
            return result
        try:
            code = result["object"]["code"]
            predicted = clone_predictions(code, [r["public"] for r in discovery])
            observed = {r["id"]: r["verdict"] for r in observations}
            matches = sum(predicted[k] == v for k, v in observed.items())
            return {**result, "target": target.id, "kind": kind, "code": code,
                    "observed_matches": matches, "eligible": matches >= 3}
        except (ValueError, TypeError, KeyError, SyntaxError) as exc:
            if attempt or kind != "self":
                return {**result, "target": target.id, "kind": kind, "eligible": False,
                        "construction_error": str(exc)}
            result = requests.ask(model, label + "-construction-repair",
                prompt + "\nPrevious source:\n" + str(result["object"].get("code", ""))
                + "\nExecution error (no target feedback): " + str(exc)
                + "\nReturn complete corrected source. Keep the same behavioral hypothesis.")
    raise RuntimeError("Unreachable author state")


def candidate(clone, cases):
    if not clone.get("eligible"):
        return None
    try:
        predictions = clone_predictions(clone["code"], [r["public"] for r in cases])
    except (ValueError, KeyError, TypeError, SyntaxError) as exc:
        clone.update(eligible=False, search_error=str(exc))
        return None
    wrong = [r for r in cases if predictions[r["public"]["id"]] in {"VALID", "INVALID"}
             and predictions[r["public"]["id"]] != ("VALID" if r["certificate"]["valid"] else "INVALID")]
    return min(wrong, key=lambda r: (len(r["public"]["steps"]), r["public"]["id"])) if wrong else None


def finish(directory, design, data, stop):
    after, before = usage(), design["budget_before"]
    result = {"at": utc_now(), "stop": stop, **data, "budget_after": after,
              "new_requests": after["calls"] - before["calls"],
              "reported_credits": after["reported_credits"] - before["reported_credits"],
              "uncertain_reserve": after["uncertain_reserve"] - before["uncertain_reserve"],
              "input_tokens": after["input_tokens"] - before["input_tokens"],
              "output_tokens": after["output_tokens"] - before["output_tokens"]}
    write_json(directory / "result.json", result)
    print(dumps({k: result[k] for k in ("stop", "new_requests", "reported_credits", "uncertain_reserve", "input_tokens", "output_tokens")}), flush=True)
    return result


def run(directory, live=False, *, tokens=TOKENS, seconds=SECONDS, reuse_discovery=None):
    directory = Path(directory)
    if live and (directory / "result.json").exists():
        result = read_json(directory / "result.json")
        print(dumps({"cached": True, "stop": result["stop"], "new_requests": result["new_requests"]}))
        return
    design = prepare(directory, tokens=tokens, seconds=seconds, reuse_discovery=reuse_discovery)
    if not live:
        print(dumps({"mode": "offline", "directory": str(directory), "max_calls": MAX_CALLS,
                     "max_additional_credits": CREDIT_CAP, "discovery_cases": 4, "held_out_candidates": 24}))
        return
    if design["budget_before"] is None:
        if list((directory / "receipts").rglob("*.request.json")):
            raise ValueError("Existing receipts without pre-run budget snapshot")
        design["budget_before"] = usage()
        design["live_started_at"] = utc_now()
        write_json(directory / "design.json", design)
    discovery = design["spec"]["discovery"]["cases"]
    hidden = design["spec"]["held_out"]["cases"]
    provider = BudgetProvider(concurrency=3, per_model_concurrency=2)
    requests = Requests(directory, design, provider)
    data = {}
    try:
        actors = parallel(MODELS_PINNED, lambda m: actor(requests, m, discovery, "discovery"), "Review four tiny programs")
        data["discovery"] = actors
        write_json(directory / "discovery.json", actors)
        active = tuple(m for m in MODELS_PINNED
                       if any(r["model"] == m.id and r["status"] == "completed" for r in actors))
        data["unscored_targets"] = [r["model"] for r in actors if r["status"] != "completed"]
        if not active:
            return finish(directory, design, data, "incomplete_discovery")
        observations = {r["model"]: r["scored"] for r in actors if r["status"] == "completed"}
        self_clones = parallel(active,
            lambda m: author(requests, m, m, observations[m.id], discovery, "self"), "Export self-surrogates")
        data["self_clones"] = self_clones
        write_json(directory / "self-clones.json", self_clones)
        selected = {r["target"]: candidate(r, hidden) for r in self_clones if r.get("target")}
        false_rejects = {r["model"]: [s["id"] for s in r["scored"] if s["false_rejection"]]
                        for r in actors if r["status"] == "completed"}
        data["direction2_discovery_false_rejections"] = false_rejects
        data["self_selected_ids"] = {k: None if v is None else v["public"]["id"] for k, v in selected.items()}
        if not any(selected.values()) and not any(false_rejects.values()):
            return finish(directory, design, data, "no_false_rejections_or_self_predicted_traps")
        peers = []
        if any(selected.values()):
            jobs = [(next(m for m in MODELS_PINNED if m.id != target.id), target) for target in active]
            peers = parallel(jobs, lambda j: author(requests, j[0], j[1], observations[j[1].id], discovery, "peer"), "Matched peer-surrogates")
        data["peer_clones"] = peers
        write_json(directory / "peer-clones.json", peers)
        panels = {}
        forecasts = {}
        for target in active:
            choices = {}
            labels = {}
            for clone in [*self_clones, *peers]:
                if clone.get("target") != target.id or not clone.get("eligible"):
                    continue
                chosen = candidate(clone, hidden)
                if chosen:
                    ident = chosen["public"]["id"]
                    choices[ident] = chosen
                    labels.setdefault(ident, []).append(clone["kind"])
            for ident in false_rejects[target.id][:1]:
                original = next(r for r in discovery if r["public"]["id"] == ident)
                choices[ident] = {**original, "public": {**original["public"], "certificate": original["certificate"]}}
                labels.setdefault(ident, []).append("prior_false_rejection_with_visible_certificate")
            if not choices:
                continue
            panels[target.id] = list(choices.values())
            forecasts[target.id] = {"selection_sources": labels,
                "predictions": {clone["kind"]: clone_predictions(clone["code"], [r["public"] for r in panels[target.id]])
                                for clone in [*self_clones, *peers] if clone.get("target") == target.id and clone.get("eligible")}}
        write_json(directory / "confirmation-seal.json", {"at": utc_now(), "before_target_calls": True,
                                                        "panels": panels, "frozen_forecasts": forecasts})
        confirmation = parallel([m for m in active if m.id in panels],
            lambda m: actor(requests, m, panels[m.id], "confirmation"), "Confirm selected cases")
        data.update(confirmation=confirmation, frozen_forecasts=forecasts)
        return finish(directory, design, data, "bounded_confirmation_complete")
    finally:
        provider.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/truth-and-selfclone-1")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--tokens", type=int, default=TOKENS)
    parser.add_argument("--seconds", type=float, default=SECONDS)
    parser.add_argument("--reuse-discovery", type=Path,
                        help="Fresh phase: reuse completed, identical discovery prompts only")
    args = parser.parse_args()
    run(args.output, args.live, tokens=args.tokens, seconds=args.seconds,
        reuse_discovery=args.reuse_discovery)
