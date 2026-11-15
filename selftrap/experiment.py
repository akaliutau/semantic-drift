"""Stage 5: a frozen, resumable experiment. Nothing runs at import or init."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import importlib.metadata
from pathlib import Path
import random
import subprocess
import threading

from tqdm import tqdm

from .game import Level, Push
from .generation import ValidationConfig, author_slot, peer_warning, procedural_puzzle
from .providers import ModelSpec, ProviderError, create_provider, estimated_cost
from .solver import SolverConfig, run_episode
from .storage import digest, dumps, read_json, usage_summary, utc_now, write_json
from .validation import ROOT


def source_hashes() -> dict:
    paths = (list((ROOT/"selftrap").glob("*.py"))+list((ROOT/"prompts").glob("*.txt"))
             +list((ROOT/"schemas").glob("*.json")))
    return {str(p.relative_to(ROOT)): digest(p.read_text()) for p in sorted(paths)}


@contextmanager
def run_lock(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/".lock").open("w") as handle:
        try: fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise ValueError("Another command is already modifying this run") from None
        try: yield
        finally: fcntl.flock(handle, fcntl.LOCK_UN)


def init_run(directory: str | Path, config: dict) -> dict:
    directory = Path(directory)
    roster = [ModelSpec(**m) for m in config["models"]]
    if len(roster) != 3 or len({m.id for m in roster}) != 3:
        raise ValueError("The crossed design requires three distinct core models")
    if any(m.provider != "adaption" or m.unit != "credits" or m.input_per_million <= 0
           or m.output_per_million <= 0 or m.id.startswith("stack:") for m in roster):
        raise ValueError("This study is restricted to sponsor credits")
    if not 1 <= config.get("concurrency_per_model", 1) <= 3:
        raise ValueError("concurrency_per_model must be 1..3")
    if config["phase"] not in {"pilot", "main"}:
        raise ValueError("phase must be pilot or main")
    if min(config["slots_per_condition"], config["repeats"], config["credit_cap"]) <= 0:
        raise ValueError("Sample size, repeats and credit cap must be positive")
    if config["controls"] < 0: raise ValueError("controls cannot be negative")
    SolverConfig(**config["solver"])
    ValidationConfig(**config["validation"])
    with run_lock(directory):
        path = directory/"manifest.json"
        if path.exists(): raise ValueError("Run already exists; use a new directory")
        slots = []
        for a, author in enumerate(roster):
            for t, target in enumerate(roster+[None]):
                for i in range(config["slots_per_condition"]):
                    slots.append({"id": f"a{a}-t{t}-n{i:03}", "author": author.id,
                                  "target": target.id if target else None, "source": "author", "index": i})
        for i in range(config["controls"]):
            slots.append({"id": f"control-{i:03}", "source": "procedural",
                          "seed": config["seed"]+10_000+i, "author": None, "target": None})
        random.Random(config["seed"]).shuffle(slots)
        manifest = {"created_at": utc_now(), "config": config, "config_hash": digest(config),
                    "source_hashes": source_hashes(), "slots": slots,
                    "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    "code_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
                    "packages": {p: importlib.metadata.version(p) for p in ("adaption", "httpx", "pydantic", "jsonschema", "tqdm", "python-dotenv")}}
        manifest["manifest_hash"] = digest(manifest)
        write_json(path, manifest)
        # Preserve the exact protocol and prompts, even after later code changes.
        for name in manifest["source_hashes"]:
            target = directory/"snapshot"/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text((ROOT/name).read_text())
        write_json(directory/"snapshot/protocol.json", read_json(ROOT/"experiment_protocol.json"))
    return manifest


def load_run(directory: str | Path, check_sources: bool = True) -> dict:
    manifest = read_json(Path(directory)/"manifest.json")
    unsigned = {k:v for k,v in manifest.items() if k != "manifest_hash"}
    if digest(unsigned) != manifest.get("manifest_hash"):
        raise ValueError("Manifest assignments or provenance changed after init")
    if digest(manifest["config"]) != manifest["config_hash"]:
        raise ValueError("Manifest config was modified after init")
    if check_sources and source_hashes() != manifest["source_hashes"]:
        raise ValueError("Code/prompts changed after init; restore the frozen revision or start a new run")
    return manifest


def reserve_cost(model: ModelSpec, messages: list[dict], tools: list[dict], tokens: int) -> float:
    # Conservative planning reservation: serialized UTF-8 bytes plus overhead.
    # This is not a tokenizer or a guarantee about provider billing.
    inputs = len(dumps([messages, tools]).encode())+1024
    return estimated_cost(model, {"input_tokens": inputs, "output_tokens": tokens})


class RunProvider:
    """Small shared catalog-cost guard and per-model concurrency limiter."""
    def __init__(self, directory: Path, cap: float, concurrency: int = 1):
        self.cap, self.spent = cap, 0.0
        self.lock = threading.Lock()
        self.limits = {}
        self.concurrency = concurrency
        self.provider = create_provider("adaption", ROOT/".env")
        for path in directory.rglob("*.request.json"):
            receipt = path.with_name(path.name.replace(".request.json", ".response.json"))
            if receipt.exists():
                value = read_json(receipt)
                if value["estimated_cost"] is not None:
                    self.spent += value["estimated_cost"]
                    continue
            error = path.with_name(path.name.replace(".request.json", ".error.json"))
            if error.exists() and read_json(error).get("billing") == "not_sent": continue
            req = read_json(path)["request"]
            self.spent += reserve_cost(ModelSpec(**req["model"]), req["messages"], req["tools"], req["max_output_tokens"])

    def complete(self, model, messages, *, tools, max_output_tokens, **kwargs):
        reservation = reserve_cost(model, messages, tools, max_output_tokens)
        with self.lock:
            if self.spent+reservation > self.cap:
                error = ProviderError("Run credit reservation cap reached; no API request sent")
                error.billing = "not_sent"
                raise error
            self.spent += reservation
            limit = self.limits.setdefault(model.id, threading.Semaphore(self.concurrency))
        with limit:
            result = self.provider.complete(model, messages, tools=tools,
                                            max_output_tokens=max_output_tokens, **kwargs)
        if result.estimated_cost is not None:
            with self.lock: self.spent += result.estimated_cost-reservation
        return result

    def close(self): self.provider.close()


def completed_slots(directory: Path, manifest: dict) -> list[tuple[dict, dict | None]]:
    return [(slot, read_json(p) if (p := directory/"slots"/slot["id"]/"result.json").exists() else None)
            for slot in manifest["slots"]]


def generate(directory: str | Path, limit: int = 1, retry_uncertain: bool = False):
    directory = Path(directory)
    with run_lock(directory):
        manifest = load_run(directory)
        cfg = manifest["config"]
        roster = [ModelSpec(**m) for m in cfg["models"]]
        seen = {r["puzzle_id"] for _, r in completed_slots(directory, manifest) if r and r["valid"]}
        for excluded in cfg.get("exclude_runs", []):
            other = Path(excluded)
            excluded_manifest = load_run(other, check_sources=False)
            seen.update(r["puzzle_id"] for _, r in completed_slots(other, excluded_manifest) if r and r["valid"])
        pending = [s for s, r in completed_slots(directory, manifest) if r is None][:limit]
        provider = RunProvider(directory, cfg["credit_cap"])
        try:
            for slot in tqdm(pending, desc="Generation", unit="slot"):
                path = directory/"slots"/slot["id"]
                if slot["source"] == "procedural":
                    result = procedural_puzzle(slot["seed"], validation=ValidationConfig(**cfg["validation"]), seen=seen)
                    write_json(path/"result.json", result)
                else:
                    author = next(m for m in roster if m.id == slot["author"])
                    result = author_slot(provider, author, roster, slot["target"], path,
                        validation=ValidationConfig(**cfg["validation"]), solver=SolverConfig(**cfg["solver"]),
                        seen=seen, max_output_tokens=cfg["author_output_tokens"], retry_uncertain=retry_uncertain)
                if result["valid"]: seen.add(result["puzzle_id"])
        finally: provider.close()


def prepare_warnings(directory: str | Path, limit: int = 1, retry_uncertain: bool = False):
    directory = Path(directory)
    with run_lock(directory):
        manifest = load_run(directory)
        cfg, selected = manifest["config"], []
        roster = [ModelSpec(**m) for m in cfg["models"]]
        for slot, result in completed_slots(directory, manifest):
            if (result and result["valid"] and slot["source"] == "author" and slot["target"] == slot["author"]
                    and not (directory/"warnings"/slot["id"]/"review.json").exists()):
                selected.append((slot, result))
        provider = RunProvider(directory, cfg["credit_cap"])
        try:
            for slot, result in tqdm(selected[:limit], desc="Peer warnings", unit="puzzle"):
                path = directory/"warnings"/slot["id"]
                author_index = next(i for i, m in enumerate(roster) if m.id == slot["author"])
                peer = roster[(author_index+1+slot["index"] % 2) % len(roster)]
                peer_result = peer_warning(provider, peer, roster, result["proposal"], path/"calls",
                    SolverConfig(**cfg["solver"]), retry_uncertain)
                write_json(path/"review.json", {"puzzle_id": result["puzzle_id"], "peer_model": peer.id,
                    "author": {"text": result["proposal"]["diagnostic_warning"], "decision": "pending"},
                    "peer": {**peer_result, "decision": "pending"}})
        finally: provider.close()


def review_warning(directory: Path, slot_id: str, kind: str, decision: str, reviewer: str):
    with run_lock(directory):
        load_run(directory)
        if kind not in {"author", "peer"} or decision not in {"approve", "reject"}:
            raise ValueError("Review kind/decision invalid")
        if any((directory/"episodes").glob("*/result.json")):
            raise ValueError("Record blind warning reviews before any solver outcomes in this run")
        path = directory/"warnings"/slot_id/"review.json"
        review = read_json(path)
        text = review[kind].get("text", "")
        if decision == "approve" and (not text or len(text.split()) > 40):
            raise ValueError("Approved warning must contain 1-40 words")
        review[kind].update(decision=decision, reviewer=reviewer, at=utc_now(), text_hash=digest(text))
        write_json(path, review)


def build_jobs(directory: Path, manifest: dict, arm: str) -> list[dict]:
    cfg, jobs = manifest["config"], []
    roster = [m["id"] for m in cfg["models"]]
    symmetry_slots = set()
    if arm == "symmetry":
        # Two slots from each of the 12 cells, chosen before solving outcomes.
        for author in roster:
            for target in roster+[None]:
                candidates = [s["id"] for s in manifest["slots"] if s["source"] == "author"
                              and s["author"] == author and s["target"] == target]
                random.Random(str([cfg["seed"], author, target])).shuffle(candidates)
                symmetry_slots.update(candidates[:2])
    for slot, puzzle in completed_slots(directory, manifest):
        if not puzzle or not puzzle["valid"]: continue
        solvers = roster
        warning = ""
        if arm == "symmetry":
            if slot["id"] not in symmetry_slots: continue
        elif arm != "baseline":
            if slot["source"] != "author" or slot["target"] != slot["author"]: continue
            solvers = [slot["target"]]
            if arm == "generic": warning = (ROOT/"prompts/generic_warning.txt").read_text().strip()
            else:
                path = directory/"warnings"/slot["id"]/"review.json"
                if not path.exists(): continue
                checked = read_json(path)[arm]
                if checked["decision"] != "approve": continue
                warning = checked["text"]
                if digest(warning) != checked["text_hash"]: raise ValueError("Reviewed warning text changed")
        for model in solvers:
            for repeat in range(cfg["repeats"]):
                episode_id = f"{slot['id']}-s{roster.index(model)}-r{repeat}-{arm}"
                seed = int(digest([cfg["seed"], slot["id"], repeat])[:12], 16)
                jobs.append({"id": episode_id, "slot": slot, "puzzle_id": puzzle["puzzle_id"],
                             "solver": model, "repeat": repeat, "arm": arm, "seed": seed, "warning": warning,
                             "symmetry": (seed % 7)+1 if arm == "symmetry" else 0})
    random.Random(cfg["seed"]+sum(map(ord, arm))).shuffle(jobs)
    return jobs


def solve(directory: str | Path, arm: str = "baseline", limit: int = 1, workers: int = 1,
          retry_uncertain: bool = False):
    directory = Path(directory)
    if arm not in {"baseline", "generic", "author", "peer", "symmetry"}: raise ValueError("Unknown warning arm")
    if not 1 <= workers <= 9: raise ValueError("workers must be between 1 and 9")
    with run_lock(directory):
        manifest = load_run(directory)
        cfg = manifest["config"]
        slots = completed_slots(directory, manifest)
        if any(result is None for _, result in slots):
            raise ValueError("Finish fixed author/control slots before solving; no result-based generation")
        jobs = build_jobs(directory, manifest, arm)
        if not jobs: raise ValueError("No eligible jobs (diagnostic warnings need blind approval)")
        jobs_path = directory/f"jobs-{arm}.json"
        if jobs_path.exists() and read_json(jobs_path) != jobs:
            raise ValueError("Job set changed after it was frozen")
        write_json(jobs_path, jobs)
        pending = [j for j in jobs if not (directory/"episodes"/j["id"]/"result.json").exists()][:limit]
        roster = {m["id"]: ModelSpec(**m) for m in cfg["models"]}
        puzzles = {s["id"]: r for s, r in slots if r and r["valid"]}
        provider = RunProvider(directory, cfg["credit_cap"], cfg.get("concurrency_per_model", 1))
        def execute(job):
            path = directory/"episodes"/job["id"]
            write_json(path/"job.json", job)
            p = puzzles[job["slot"]["id"]]["proposal"]
            level, bad = Level.from_rows(p["board"]), Push.from_dict(p["predicted_first_push"])
            if job["symmetry"]:
                from .game import transform_level, transform_push
                level = transform_level(level, job["symmetry"])
                bad = transform_push(bad, 7, job["symmetry"])
            return run_episode(provider, roster[job["solver"]], level, path,
                bad_push=bad, config=SolverConfig(**cfg["solver"]),
                seed=job["seed"], warning=job["warning"], retry_uncertain=retry_uncertain)
        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for _ in tqdm(pool.map(execute, pending), total=len(pending), desc=f"Solve/{arm}", unit="episode"): pass
        finally: provider.close()


def status(directory: str | Path) -> dict:
    directory = Path(directory)
    manifest = load_run(directory, check_sources=False)
    slots = completed_slots(directory, manifest)
    episodes = [read_json(p) for p in (directory/"episodes").glob("*/result.json")]
    return {"phase": manifest["config"]["phase"], "slots_assigned": len(slots),
            "slots_completed": sum(r is not None for _, r in slots),
            "valid_puzzles": sum(bool(r and r["valid"]) for _, r in slots),
            "episodes_completed": len(episodes),
            "service_failures": sum(r["status"] == "service_failure" for r in episodes),
            "usage": usage_summary(directory)}
