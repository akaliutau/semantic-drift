"""Mechanical author validation; never consult solver outcomes."""

from functools import lru_cache
import json
import math
from pathlib import Path
from time import perf_counter

from jsonschema import Draft202012Validator

from .game import Level, Push, canonical_id, parse_plan
from .oracle import Oracle

ROOT = Path(__file__).resolve().parent.parent


@lru_cache
def proposal_validator() -> Draft202012Validator:
    schema = json.loads((ROOT/"schemas/puzzle.schema.json").read_text())
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def validate(proposal: dict, roster: list[str], min_pushes: int = 6,
             max_pushes: int = 30, max_states: int = 100_000,
             timeout: float = 5.0, seen: set[str] | None = None) -> dict:
    started = perf_counter()
    certificate = {}
    def finish(errors: list[str]) -> dict:
        return {"valid": not errors, "errors": errors, "certificate": certificate,
                "seconds": perf_counter()-started}
    errors = sorted((e.message for e in proposal_validator().iter_errors(proposal)))
    if errors: return finish(errors)
    forecasts = proposal["forecasts"]
    if sorted(f["solver_id"] for f in forecasts) != sorted(roster):
        return finish(["Forecasts must name each frozen solver exactly once"])
    for f in forecasts:
        probs = f["p_predicted_first_push"], f["p_full_solution"]
        if not all(math.isfinite(p) for p in probs) or sum(probs) > 1+1e-9:
            return finish(["Forecast probabilities must be finite and sum to at most one"])
    result = validate_puzzle(proposal["board"], proposal["solution"], proposal["predicted_first_push"],
                             min_pushes, max_pushes, max_states, timeout, seen)
    result["certificate"]["warning_word_count"] = len(proposal["diagnostic_warning"].split())
    return result


def validate_puzzle(rows: list[str], solution: list[dict], predicted_first_push: dict,
                    min_pushes: int = 6, max_pushes: int = 30, max_states: int = 100_000,
                    timeout: float = 5.0, seen: set[str] | None = None) -> dict:
    """Shared mechanical checker for authored and procedural puzzles."""
    started = perf_counter()
    certificate = {}
    def finish(errors):
        return {"valid": not errors, "errors": errors, "certificate": certificate,
                "seconds": perf_counter()-started}
    try:
        level = Level.from_rows(rows)
        if (level.board.height, level.board.width) != (7, 7):
            return finish(["Require a 7x7 board"])
        if len(level.state.boxes) not in (2, 3) or level.board.solved(level.state):
            return finish(["Require 2-3 boxes and an unsolved start"])
        puzzle_id = canonical_id(level)
        certificate["puzzle_id"] = puzzle_id
        if seen is not None and puzzle_id in seen: return finish(["Duplicate puzzle"])
        plan = parse_plan(solution)
        grid_wins, oracle_wins = level.replay(plan)["completed"], Oracle(level).replay(plan)
        if grid_wins != oracle_wins:
            raise RuntimeError("Independent witness checkers disagree")
        if not grid_wins:
            return finish(["Winning witness fails independent replay"])
        optimal = Oracle(level).search(max_states, timeout)
        certificate["solution_search"] = optimal.to_dict()
        if optimal.status != "SOLVED": return finish([f"Solution oracle: {optimal.status}"])
        if not level.replay(optimal.plan)["completed"]:
            raise RuntimeError("Independent oracle and replay checker disagree")
        if not min_pushes <= len(optimal.plan) <= max_pushes:
            return finish([f"Optimal solution outside {min_pushes}..{max_pushes} pushes"])
        bad = Push.from_dict(predicted_first_push)
        after = level.board.push(level.state, bad)
        trap = Oracle(Level(level.board, after)).search(max_states, timeout)
        certificate["trap_search"] = trap.to_dict()
        if trap.status == "SOLVED" and not Level(level.board, after).replay(trap.plan)["completed"]:
            raise RuntimeError("Independent trap search and replay checker disagree")
        if trap.status != "UNSOLVABLE":
            return finish([f"Declared trap not certified unsolvable: {trap.status}"])
    except ValueError as exc:
        return finish([str(exc)])
    return finish([])
