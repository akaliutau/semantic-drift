"""Retry incomplete authoring at the upstream recommended Qwen temperature."""

import argparse
from dataclasses import asdict
from pathlib import Path

from .concept_pilot import run
from .providers import ModelSpec
from .storage import digest, read_json, utc_now, write_json


MODEL = ModelSpec("qwen3.8-27b", input_per_million=.039, output_per_million=1.25,
                  settings={"temperature": 1.0})


def retry(directory, game="numberlink", live=False):
    directory = Path(directory)
    policy = {"model": asdict(MODEL), "max_output_tokens": 32768,
              "selection": "First-priority model from the previous completion-only screen; no pruning correctness criterion.",
              "reason": "Both the temperature-zero screen and the temperature-one 10000-token call exhausted reasoning without a final answer. Retain temperature one, increase the output budget to 32768, and keep the task and datasets identical. Prior incomplete outputs remain unscored.",
              "source": "https://huggingface.co/Qwen/Qwen3.8-27B#best-practices",
              "unsupported_controls": "Adaption exposes no top_p/top_k controls here; do not claim the complete upstream decoding recipe.",
              "source_hash": digest(Path(__file__).read_text())}
    path = directory / "sampling-policy.json"
    if path.exists():
        if read_json(path)["policy_hash"] != digest(policy):
            raise ValueError("Sampling policy changed")
    else:
        write_json(path, {"at": utc_now(), "before_api": True, "policy_hash": digest(policy), "policy": policy})
    run(directory, game, live=live, models=(MODEL,))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/three-game-sampling-1"))
    parser.add_argument("--game", choices=("dots", "ricochet", "numberlink"), default="numberlink")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    retry(args.output, args.game, args.live)
