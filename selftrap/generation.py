"""Stage 4: fixed-budget authors, procedural controls, and independent warnings."""

from dataclasses import asdict, dataclass
from pathlib import Path
import random
from time import perf_counter

from tqdm import tqdm

from .game import Board, DIRECTIONS, Level, State, canonical_id
from .oracle import Oracle
from .providers import ModelSpec, Provider, ProviderError
from .solver import SolverConfig, render, solver_template
from .storage import Journal, digest, dumps, read_json, strict_json, write_json
from .validation import ROOT, validate, validate_puzzle


@dataclass(frozen=True)
class ValidationConfig:
    min_pushes: int = 6
    max_pushes: int = 30
    max_states: int = 100_000
    timeout: float = 5.0

    def __post_init__(self):
        if not 1 <= self.min_pushes <= self.max_pushes <= 40 or self.max_states < 1 or self.timeout <= 0:
            raise ValueError("Invalid difficulty bounds or oracle caps")


def solver_contract(roster: list[ModelSpec], solver: SolverConfig) -> str:
    rules = render(solver_template(solver), public_puzzle_json="[a supplied puzzle]", warning_block="")
    return rules+"\nModel configurations: "+dumps([asdict(m) for m in roster])


def author_slot(provider: Provider, author: ModelSpec, roster: list[ModelSpec], target: str | None,
                directory: str | Path, *, validation: ValidationConfig | None = None,
                solver: SolverConfig | None = None, seen: set[str] | None = None,
                max_output_tokens: int = 8000, retry_uncertain: bool = False) -> dict:
    directory = Path(directory)
    validation, solver = validation or ValidationConfig(), solver or SolverConfig()
    ids = [m.id for m in roster]
    if author.id not in ids or target is not None and target not in ids:
        raise ValueError("Author and target must belong to the frozen core roster")
    prompt = render((ROOT/"prompts/author.txt").read_text(), author_model_id=author.id,
        solver_roster_json=dumps([asdict(m) for m in roster]),
        assignment=f"TARGET {target}" if target else "GENERIC_HARD",
        min_optimal_pushes=validation.min_pushes, max_optimal_pushes=validation.max_pushes,
        solver_contract=solver_contract(roster, solver),
        proposal_schema_json=(ROOT/"schemas/puzzle.schema.json").read_text())
    identity = {"prompt_hash": digest(prompt), "author": asdict(author),
                "validation": asdict(validation), "output_budget": max_output_tokens}
    metadata, output = directory/"author.json", directory/"result.json"
    if metadata.exists():
        if read_json(metadata)["identity"] != identity: raise ValueError("Author configuration changed on resume")
        if output.exists(): return read_json(output)
    else: write_json(metadata, {"identity": identity, "prompt": prompt})
    journal = Journal(directory/"calls", retry_uncertain)
    messages = [{"role": "user", "content": prompt}]
    used_tokens = 0
    result = {"valid": False, "status": "invalid", "author": author.id, "target": target}
    for attempt in range(2):
        remaining = max_output_tokens-used_tokens
        if remaining <= 0: break
        try:
            response = journal.complete(f"proposal-{attempt}", provider, author, messages,
                tools=[], max_output_tokens=remaining, timeout=180, tool_choice="none")
        except ProviderError as exc:
            result.update(status="service_failure", reason=str(exc))
            break
        actual = (response.usage or {}).get("output_tokens")
        used_tokens += actual if isinstance(actual, int) and actual >= 0 else remaining
        if response.raw.get("status") not in {"completed", "incomplete"} or response.raw.get("model") != author.id:
            result.update(status="service_failure", reason="Failed response or model ID mismatch")
            break
        if response.raw.get("status") == "incomplete" or used_tokens > max_output_tokens:
            result.update(reason="Author output budget exhausted or response incomplete")
            break
        proposal = None
        try:
            proposal = strict_json(response.text)
            checked = validate(proposal, ids, **asdict(validation), seen=seen)
        except (ValueError, TypeError) as exc:
            checked = {"valid": False, "errors": [str(exc)]}
        write_json(directory/f"proposal-{attempt}.json", {"proposal": proposal, "validation": checked})
        if checked["valid"]:
            result.update(valid=True, status="valid", puzzle_id=checked["certificate"]["puzzle_id"],
                          proposal=proposal, validation=checked, accepted_attempt=attempt)
            break
        result["reason"] = checked["errors"]
        # Validity feedback only: no solver queries or outcome-based selection.
        messages.extend(response.output or [{"role": "assistant", "content": response.text}])
        messages.append({"role": "user", "content": "Mechanical validation failed: "+dumps(checked["errors"])
                         +". You have one validity-only repair. Return the complete proposal JSON."})
    write_json(output, result)
    return result


def reverse_scramble(rng: random.Random, steps: int = 20) -> Level | None:
    """Start solved and apply reverse pushes; all generated starts are solvable."""
    interior = {(r, c) for r in range(1, 6) for c in range(1, 6)}
    floor = {p for p in interior if rng.random() > 0.12}
    if len(floor) < 12: return None
    boxes = frozenset(rng.sample(sorted(floor), rng.choice([2, 3])))
    board = Board(7, 7, frozenset((r, c) for r in range(7) for c in range(7) if (r, c) not in floor), boxes)
    state = State(rng.choice(sorted(floor-boxes)), boxes)
    for _ in range(steps):
        reached = board.reachable(state)
        pulls = []
        for r, c in sorted(state.boxes):
            for dr, dc in DIRECTIONS.values():
                prev_box, prev_player = (r-dr, c-dc), (r-2*dr, c-2*dc)
                if prev_box in reached and board.floor(prev_player) and prev_player not in state.boxes:
                    pulls.append(State(prev_player, (state.boxes-{(r, c)}) | {prev_box}))
        if not pulls: break
        state = rng.choice(pulls)
    return Level(board, state) if not board.solved(state) else None


def procedural_puzzle(seed: int, *, validation: ValidationConfig | None = None,
                      seen: set[str] | None = None, max_attempts: int = 200,
                      max_seconds: float = 60, progress: bool = True) -> dict:
    validation = validation or ValidationConfig()
    rng, started = random.Random(seed), perf_counter()
    attempted = 0
    for attempt in tqdm(range(max_attempts), desc="Procedural proposals", disable=not progress, leave=False):
        if perf_counter()-started >= max_seconds: break
        attempted += 1
        level = reverse_scramble(rng)
        if level is None or seen is not None and canonical_id(level) in seen: continue
        remaining = max_seconds-(perf_counter()-started)
        if remaining <= 0: break
        solution = Oracle(level).search(validation.max_states, min(validation.timeout, remaining))
        if solution.status != "SOLVED" or not validation.min_pushes <= len(solution.plan) <= validation.max_pushes:
            continue
        actions = level.board.legal_pushes(level.state)
        rng.shuffle(actions)
        for bad in actions:
            remaining = max_seconds-(perf_counter()-started)
            if remaining <= 0: break
            trap = Oracle(Level(level.board, level.board.push(level.state, bad))).search(
                validation.max_states, min(validation.timeout, remaining))
            if trap.status != "UNSOLVABLE": continue
            puzzle = {"board": level.board.render(level.state), "solution": [p.to_dict() for p in solution.plan],
                      "predicted_first_push": bad.to_dict()}
            checked = validate_puzzle(**{"rows": puzzle["board"], "solution": puzzle["solution"],
                                        "predicted_first_push": puzzle["predicted_first_push"]},
                                      **asdict(validation), seen=seen)
            if checked["valid"]:
                return {"valid": True, "status": "valid", "source": "procedural", "seed": seed,
                        "puzzle_id": checked["certificate"]["puzzle_id"], "proposal": puzzle,
                        "validation": checked, "attempts": attempt+1, "seconds": perf_counter()-started}
    return {"valid": False, "status": "invalid", "source": "procedural", "seed": seed,
            "attempts": attempted, "seconds": perf_counter()-started,
            "reason": "No valid nonduplicate control within attempt/time budget"}


def peer_warning(provider: Provider, peer: ModelSpec, roster: list[ModelSpec], proposal: dict,
                 directory: str | Path, solver: SolverConfig | None = None,
                 retry_uncertain: bool = False) -> dict:
    # Strict allowlist: no original author identity, warning, forecasts or outcomes.
    info = {k: proposal[k] for k in ("board", "solution", "predicted_first_push")}
    prompt = render((ROOT/"prompts/warning_peer.txt").read_text(),
        verified_puzzle_and_bad_push_json=dumps(info), solver_contract=solver_contract(roster, solver or SolverConfig()))
    try:
        response = Journal(directory, retry_uncertain).complete("warning", provider, peer,
            [{"role": "user", "content": prompt}], tools=[], max_output_tokens=1000, timeout=180, tool_choice="none")
    except ProviderError as exc:
        return {"status": "service_failure", "reason": str(exc)}
    if response.raw.get("status") != "completed" or response.raw.get("model") != peer.id:
        return {"status": "unavailable", "reason": "Incomplete, failed or mismatched peer response"}
    return {"status": "pending_review", "text": response.text.strip(),
            "word_count": len(response.text.split())}
