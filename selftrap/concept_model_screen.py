"""Bounded technical model selection, independent of pruning correctness."""

import argparse
from dataclasses import asdict
from pathlib import Path

from .concept_pilot import run
from .providers import ModelSpec
from .storage import digest, dumps, read_json, utc_now, write_json


PRIORITY = (
    ModelSpec("qwen3.8-27b", input_per_million=.039, output_per_million=1.25, settings={"temperature": 0}),
    ModelSpec("deepseek-v4-flash", input_per_million=.045, output_per_million=.14, settings={"temperature": 0}),
    ModelSpec("glm-5.3-flash", input_per_million=.15, output_per_million=.5, settings={"temperature": 0}),
)


def screen(directory, live=False):
    directory = Path(directory)
    policy = {"priority": [asdict(m) for m in PRIORITY], "game": "numberlink",
              "rule": "Choose the first model in this predeclared priority that returns executable code. Ignore oracle correctness, p_sound and pruning coverage when selecting. Retain all authors, including unsafe ones and incomplete responses.",
              "source_hash": digest(Path(__file__).read_text())}
    path = directory / "selection-policy.json"
    if path.exists():
        if read_json(path)["policy_hash"] != digest(policy):
            raise ValueError("Technical selection policy changed")
    else:
        write_json(path, {"at": utc_now(), "before_api": True, "policy_hash": digest(policy), "policy": policy})
    run(directory, "numberlink", live=live, models=PRIORITY)
    if live:
        authors = read_json(directory / "numberlink-cold.json")["authors"]
        eligible = {a["model"] for a in authors if a.get("construction_valid") and a["status"] == "completed"}
        selected = next((m.id for m in PRIORITY if m.id in eligible), None)
        write_json(directory / "model-selection.json", {"selected": selected, "eligible": sorted(eligible),
                                                       "policy_hash": digest(policy)})
        print(dumps({"technical_model_selection": selected}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/three-game-model-screen-1"))
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    screen(args.output, args.live)
