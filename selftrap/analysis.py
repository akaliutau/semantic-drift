"""Stage 6: the declared outcomes only; no model-based judging.

The independent unit is a generation slot/puzzle. Repetitions remain together.
An unfinished or infrastructure-failed observation is missing, never a success
or a zero-error observation. Reports on incomplete runs say so explicitly.
"""

from collections import Counter, defaultdict
import csv
from pathlib import Path
import random
from statistics import mean

from tqdm import tqdm

from .experiment import completed_slots, load_run
from .storage import read_json, usage_summary, write_json


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered)-1)*probability
    low = int(position)
    high = min(low+1, len(ordered)-1)
    return ordered[low]+(ordered[high]-ordered[low])*(position-low)


def interval(values: list[float]) -> list[float] | None:
    return [percentile(values, .025), percentile(values, .975)] if values else None


def contrast(cells: dict[tuple[str, str], list[float]], roster: list[str]) -> float | None:
    keys = [(a, t) for a in roster for t in roster]
    if any(not cells.get(k) for k in keys): return None
    return mean(mean(cells[(a, a)]) for a in roster)-mean(
        mean(cells[(a, t)]) for a in roster for t in roster if a != t)


def observations(directory: Path) -> dict:
    result = {}
    for path in (directory/"episodes").glob("*/result.json"):
        job = read_json(path.parent/"job.json")
        key = job["slot"]["id"], job["solver"], job["arm"], job["repeat"]
        if key in result: raise ValueError("Duplicate scored episode key")
        value = read_json(path)
        if value.get("model") != job["solver"]: raise ValueError("Episode solver identity mismatch")
        result[key] = value
    return result


def values_for(data: dict, slot: str, model: str, arm: str, repeats: int, field: str) -> list:
    return [data.get((slot, model, arm, r), {}).get(field) for r in range(repeats)]


def rates(data, slot, model, arm, repeats, field) -> tuple[float | None, float, float]:
    values = values_for(data, slot, model, arm, repeats, field)
    known = [float(v) for v in values if v is not None]
    missing = repeats-len(known)
    return (mean(known) if not missing else None, sum(known)/repeats, (sum(known)+missing)/repeats)


def primary(directory: Path, manifest: dict, data: dict, resamples: int, progress: bool = True) -> dict:
    roster = [m["id"] for m in manifest["config"]["models"]]
    repeats = manifest["config"]["repeats"]
    cells, conditional, lower, upper = (defaultdict(list) for _ in range(4))
    coverage = Counter()
    for slot, authored in completed_slots(directory, manifest):
        if slot["source"] != "author" or slot["target"] is None: continue
        key = slot["author"], slot["target"]
        if authored is None or authored["status"] == "service_failure":
            lower[key].append(-1.0)
            upper[key].append(1.0)
            coverage["missing_author_slots"] += 1
        elif not authored["valid"]:
            cells[key].append(0.0)
            lower[key].append(0.0)
            upper[key].append(0.0)
            coverage["invalid_slots_zero_credit"] += 1
        else:
            values = {m: rates(data, slot["id"], m, "baseline", repeats, "predicted_error") for m in roster}
            target = values[slot["target"]]
            peers = [values[m] for m in roster if m != slot["target"]]
            lower[key].append(target[1]-mean(v[2] for v in peers))
            upper[key].append(target[2]-mean(v[1] for v in peers))
            if all(v[0] is not None for v in values.values()):
                value = target[0]-mean(v[0] for v in peers)
                cells[key].append(value)
                conditional[key].append(value)
                coverage["complete_valid_puzzles"] += 1
            else: coverage["incomplete_valid_puzzles"] += 1
    point = contrast(cells, roster)
    samples = []
    if point is not None:
        rng = random.Random(manifest["config"]["seed"]+123)
        for _ in tqdm(range(resamples), desc="Slot bootstrap", disable=not progress, leave=False):
            boot = {key: rng.choices(values, k=len(values)) for key, values in cells.items()}
            samples.append(contrast(boot, roster))
    # D's off-diagonal cells have negative coefficients, so exchange bounds there.
    bounds = [mean(mean(lower[(a,a)]) for a in roster)-mean(mean(upper[(a,t)]) for a in roster for t in roster if a != t),
              mean(mean(upper[(a,a)]) for a in roster)-mean(mean(lower[(a,t)]) for a in roster for t in roster if a != t)]
    return {"D_complete_panels": point, "D_95_cluster_bootstrap": interval(samples),
            "D_conditional_valid": contrast(conditional, roster),
            "D_missing_data_bounds": bounds, "coverage": dict(coverage),
            "complete": not coverage["missing_author_slots"] and not coverage["incomplete_valid_puzzles"],
            "cells": [{"author": a, "target": t, "delta": mean(cells[(a,t)]) if cells[(a,t)] else None,
                       "slots_in_estimate": len(cells[(a,t)])} for a in roster for t in roster],
            "interpretation": "D compares diagonal vs off-diagonal target specificity; invalid slots get zero credit. Partial-data estimates need missing-data bounds."}


def brier_and_rates(directory: Path, manifest: dict, data: dict, pilot_rates: dict | None = None) -> dict:
    repeats = manifest["config"]["repeats"]
    briers, baseline_briers, groups, bins = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    validity = defaultdict(Counter)
    for slot, authored in completed_slots(directory, manifest):
        if slot["source"] == "author":
            validity[(slot["author"], slot["target"])][authored["status"] if authored else "pending"] += 1
        if not authored or not authored["valid"]: continue
        forecasts = {f["solver_id"]: f for f in authored["proposal"].get("forecasts", [])}
        for m in manifest["config"]["models"]:
            model = m["id"]
            for r in range(repeats):
                result = data.get((slot["id"], model, "baseline", r))
                if not result: continue
                if result["full_success"] is not None:
                    groups[(slot["source"], slot["author"], slot["target"], model)].append(result)
                if model not in forecasts: continue
                for field, forecast in (("predicted_error", "p_predicted_first_push"), ("full_success", "p_full_solution")):
                    if result[field] is None: continue
                    p, y = forecasts[model][forecast], float(result[field])
                    briers[field].append((p-y)**2)
                    bins[(field, min(int(p*5), 4))].append((p,y))
                    if pilot_rates and pilot_rates.get(model, {}).get(field) is not None:
                        baseline_briers[field].append((pilot_rates[model][field]-y)**2)
    return {"brier": {f: {"score": mean(v), "n_forecasts_scored": len(v),
                          "pilot_base_rate_brier": mean(baseline_briers[f]) if baseline_briers[f] else None} for f,v in briers.items()},
            "calibration": [{"event": f, "bin": b, "n": len(v), "mean_forecast": mean(x for x,y in v),
                             "observed": mean(y for x,y in v)} for (f,b),v in sorted(bins.items())],
            "validity": [{"author": a, "target": t, "counts": dict(v)} for (a,t),v in validity.items()],
            "behavior": [{"source": s, "author": a, "target": t, "solver": m, "episodes": len(v),
                          "error_rate": mean(r["predicted_error"] for r in v),
                          "solve_rate": mean(r["full_success"] for r in v)} for (s,a,t,m),v in groups.items()]}


def warning_effects(directory: Path, manifest: dict, data: dict, resamples: int, progress: bool = True) -> dict:
    repeats = manifest["config"]["repeats"]
    effects = []
    for arm in tqdm(("generic", "author", "peer"), desc="Warning analysis", disable=not progress, leave=False):
        solve_diffs, error_diffs = [], []
        eligible = 0
        for slot, authored in completed_slots(directory, manifest):
            if not authored or not authored["valid"] or slot["source"] != "author" or slot["target"] != slot["author"]:
                continue
            eligible += 1
            before = rates(data, slot["id"], slot["target"], "baseline", repeats, "full_success")[0]
            after = rates(data, slot["id"], slot["target"], arm, repeats, "full_success")[0]
            if before is None or after is None: continue
            solve_diffs.append(after-before)
            before_error = rates(data, slot["id"], slot["target"], "baseline", repeats, "predicted_error")[0]
            after_error = rates(data, slot["id"], slot["target"], arm, repeats, "predicted_error")[0]
            error_diffs.append(after_error-before_error)
        effect = {"arm": arm, "eligible_puzzles": eligible, "paired_complete_puzzles": len(solve_diffs),
                  "solve_improvement": mean(solve_diffs) if solve_diffs else None,
                  "error_change": mean(error_diffs) if error_diffs else None,
                  "p_value": None, "p_holm": None, "solve_95_cluster_bootstrap": None}
        if solve_diffs:
            rng, n = random.Random(manifest["config"]["seed"]+sum(map(ord, arm))), len(solve_diffs)
            draws = [mean(rng.choices(solve_diffs, k=n)) for _ in range(resamples)]
            # Paired randomization test: one sign per puzzle, not per episode.
            observed = abs(mean(solve_diffs))
            extreme = sum(abs(mean(v*rng.choice([-1,1]) for v in solve_diffs)) >= observed-1e-12
                          for _ in range(resamples))
            effect.update(solve_95_cluster_bootstrap=interval(draws), p_value=(extreme+1)/(resamples+1))
        effects.append(effect)
    # Keep the predeclared family of three even when an arm has no data.
    ordered = sorted((e for e in effects if e["p_value"] is not None), key=lambda e:e["p_value"])
    adjusted = 0
    for index, item in enumerate(ordered):
        adjusted = max(adjusted, min(1, (3-index)*item["p_value"]))
        item["p_holm"] = adjusted
    return {"arms": effects, "test": "Paired puzzle-level sign randomization; Holm family=3 for full-solve effects",
            "warning": "Missing/unapproved warnings are unavailable, not zero effects. Warning trials are not conditioned on baseline failure."}


def analyse(directory: str | Path, output: str | Path, *, resamples: int = 10_000,
            pilot: str | Path | None = None, progress: bool = True) -> dict:
    if resamples < 1: raise ValueError("resamples must be positive")
    directory, output = Path(directory), Path(output)
    manifest = load_run(directory)
    data = observations(directory)
    pilot_rates = None
    if pilot:
        pilot_path = Path(pilot)
        pilot_manifest = load_run(pilot_path, check_sources=False)
        if pilot_manifest["config"]["phase"] != "pilot": raise ValueError("Calibration baseline must be a pilot run")
        if manifest["config"]["phase"] != "main": raise ValueError("Use a disjoint pilot only for main analysis")
        pilot_data = observations(pilot_path)
        pilot_rates = {}
        for m in manifest["config"]["models"]:
            pilot_rates[m["id"]] = {}
            for field in ("predicted_error", "full_success"):
                values = [v[field] for k,v in pilot_data.items() if k[1] == m["id"] and k[2] == "baseline" and v[field] is not None]
                pilot_rates[m["id"]][field] = mean(values) if values else None
    primary_result = primary(directory, manifest, data, resamples, progress)
    symmetry = []
    for key, result in data.items():
        slot, model, arm, repeat = key
        if arm != "symmetry": continue
        baseline = data.get((slot, model, "baseline", repeat))
        if baseline and baseline["full_success"] is not None and result["full_success"] is not None:
            symmetry.append({"solve_change": float(result["full_success"])-float(baseline["full_success"]),
                             "error_change": float(result["predicted_error"])-float(baseline["predicted_error"])})
    report = {"manifest_hash": digest_manifest(manifest), "phase": manifest["config"]["phase"],
              "primary": primary_result, "secondary": brier_and_rates(directory, manifest, data, pilot_rates),
              "warnings": warning_effects(directory, manifest, data, resamples, progress),
              "symmetry": {"paired_episodes": len(symmetry),
                           "solve_change": mean(v["solve_change"] for v in symmetry) if symmetry else None,
                           "error_change": mean(v["error_change"] for v in symmetry) if symmetry else None},
              "terminal_status_counts": dict(Counter(v["status"] for v in data.values())),
              "usage": usage_summary(directory), "bootstrap_resamples": resamples,
              "limitations": ["Only the selected model+harness configurations are tested.",
                              "Error targeting and calibration do not establish privileged introspection.",
                              "Partial runs and missing infrastructure outcomes need sensitivity bounds."]}
    write_json(output/"summary.json", report)
    for name, rows in (("specificity", report["primary"]["cells"]), ("behavior", report["secondary"]["behavior"]),
                       ("calibration", report["secondary"]["calibration"]), ("warnings", report["warnings"]["arms"])):
        if rows:
            with (output/f"{name}.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows({k: row.get(k) for k in rows[0]} for row in rows)
    return report


def digest_manifest(manifest: dict) -> str:
    from .storage import digest
    return digest(manifest)
