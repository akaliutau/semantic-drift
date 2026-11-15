"""Small offline exports and board replays; never calls a provider."""

from pathlib import Path

from tqdm import tqdm

from .experiment import completed_slots, load_run
from .game import Level, parse_plan
from .storage import dumps, read_json, write_json


def export_dataset(directory: str | Path, output: str | Path):
    directory, output = Path(directory), Path(output)
    manifest = load_run(directory, check_sources=False)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/"manifest.json", manifest)
    episodes = {}
    for path in (directory/"episodes").glob("*/job.json"):
        job = read_json(path)
        result = path.parent/"result.json"
        episodes.setdefault(job["slot"]["id"], []).append({
            "job": job, "result": read_json(result) if result.exists() else None})
    with (output/"dataset.jsonl").open("w") as stream:
        for slot, result in tqdm(completed_slots(directory, manifest), desc="Dataset export", unit="slot"):
            attempts = [read_json(p) for p in sorted((directory/"slots"/slot["id"]).glob("proposal-*.json"))]
            review_path = directory/"warnings"/slot["id"]/"review.json"
            stream.write(dumps({"assignment": slot, "authored": result, "attempts": attempts,
                                "warning_review": read_json(review_path) if review_path.exists() else None,
                                "episodes": episodes.get(slot["id"], [])})+"\n")


def replay_text(puzzle_path: str | Path, plan_path: str | Path | None = None) -> str:
    puzzle = read_json(puzzle_path)
    puzzle = puzzle.get("proposal", puzzle)
    level = Level.from_rows(puzzle["board"])
    obj = read_json(plan_path) if plan_path else {"plan": puzzle["solution"]}
    plan = parse_plan(obj["plan"])
    state = level.state
    frames = ["Start\n"+"\n".join(level.board.render(state))]
    for index, action in enumerate(plan):
        try: state = level.board.push(state, action)
        except ValueError as exc:
            frames.append(f"Illegal action {index+1}: {exc}")
            break
        frames.append(f"Push {index+1}: {action.to_dict()}\n"+"\n".join(level.board.render(state)))
    frames.append(f"Complete legal solution: {level.replay(plan)['completed']}")
    return "\n\n".join(frames)
