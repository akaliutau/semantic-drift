"""Explicit stage commands; all API stages default to one job."""

import argparse
from pathlib import Path

from . import experiment
from .storage import dumps, read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="Freeze a local run manifest; no API requests")
    init.add_argument("--config", type=Path, default=Path("configs/pilot.json"))
    init.add_argument("--output", type=Path, required=True)
    for name in ("generate", "prepare-warnings", "solve"):
        command = commands.add_parser(name)
        command.add_argument("run", type=Path)
        command.add_argument("--limit", type=int, default=1, help="Maximum jobs to execute; default 1")
        command.add_argument("--retry-uncertain", action="store_true")
        if name == "solve":
            command.add_argument("--arm", choices=["baseline", "generic", "author", "peer", "symmetry"], default="baseline")
            command.add_argument("--workers", type=int, default=1)
    review = commands.add_parser("review-warning", help="Record a human's blind warning review")
    review.add_argument("run", type=Path)
    review.add_argument("--slot", required=True)
    review.add_argument("--kind", required=True, choices=["author", "peer"])
    review.add_argument("--decision", required=True, choices=["approve", "reject"])
    review.add_argument("--reviewer", required=True)
    status = commands.add_parser("status")
    status.add_argument("run", type=Path)
    analysis = commands.add_parser("analyse", help="Offline scoring and puzzle-cluster uncertainty intervals")
    analysis.add_argument("run", type=Path)
    analysis.add_argument("--output", type=Path, required=True)
    analysis.add_argument("--resamples", type=int, default=10000)
    analysis.add_argument("--pilot", type=Path)
    export = commands.add_parser("export", help="Export assignments, proposals, checks and outcomes as JSONL")
    export.add_argument("run", type=Path)
    export.add_argument("--output", type=Path, required=True)
    replay = commands.add_parser("replay", help="Print a puzzle solution, without an API")
    replay.add_argument("puzzle", type=Path)
    replay.add_argument("--plan", type=Path, help="Solver result JSON; otherwise replay the reference witness")
    args = parser.parse_args()
    if hasattr(args, "limit") and args.limit <= 0: parser.error("limit must be positive")
    if args.command == "init":
        value = experiment.init_run(args.output, read_json(args.config))
        print(dumps({"output": str(args.output), "slots": len(value["slots"]), "api_requests": 0}))
    elif args.command == "generate": experiment.generate(args.run, args.limit, args.retry_uncertain)
    elif args.command == "prepare-warnings": experiment.prepare_warnings(args.run, args.limit, args.retry_uncertain)
    elif args.command == "solve": experiment.solve(args.run, args.arm, args.limit, args.workers, args.retry_uncertain)
    elif args.command == "review-warning":
        experiment.review_warning(args.run, args.slot, args.kind, args.decision, args.reviewer)
    elif args.command == "status": print(dumps(experiment.status(args.run)))
    elif args.command == "analyse":
        from .analysis import analyse
        result = analyse(args.run, args.output, resamples=args.resamples, pilot=args.pilot)
        print(dumps({"output": str(args.output), "primary": result["primary"], "usage": result["usage"]}))
    elif args.command == "export":
        from .reporting import export_dataset
        export_dataset(args.run, args.output)
    elif args.command == "replay":
        from .reporting import replay_text
        print(replay_text(args.puzzle, args.plan))


if __name__ == "__main__": main()
