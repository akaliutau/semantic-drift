"""One text request or puzzle episode; never starts an experiment/dataset."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tqdm import tqdm

from selftrap.game import Level, Push
from selftrap.providers import ModelSpec, ProviderError, create_provider
from selftrap.solver import SolverConfig, run_episode
from selftrap.storage import Journal, dumps, read_json, usage_summary
from selftrap.validation import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Explicitly execute the chosen sponsor API check")
    parser.add_argument("--mode", choices=["text", "episode"], default="text")
    parser.add_argument("--model", default="qwen3.5-397b")
    parser.add_argument("--output", type=Path, help="Defaults to a separate directory per mode/model")
    parser.add_argument("--force-first-simulation", action="store_true",
                        help="Episode only: require a tool call instead of letting the solver choose")
    parser.add_argument("--max-output-tokens", type=int, default=2000)
    parser.add_argument("--retry-uncertain", action="store_true")
    args = parser.parse_args()
    if not args.live:
        parser.error("No network request made. Add --live to run the selected check.")
    if args.force_first_simulation and args.mode != "episode":
        parser.error("--force-first-simulation requires --mode episode")
    output = args.output or ROOT/"runs"/f"api-smoke-{args.mode}-{args.model}"
    protocol = read_json(ROOT/"experiment_protocol.json")
    specs = protocol["model_profiles"]["sponsored_core"]+protocol["model_profiles"]["optional_small_model_audit"]
    matches = [m for m in specs if m["id"] == args.model]
    if len(matches) != 1: parser.error("Choose an explicitly priced sponsor model from the protocol")
    model = ModelSpec(**matches[0])
    level = Level.from_rows(["#######", "#_____#", "#_.$__#", "#_____#",
                            "#_@$._#", "#_____#", "#######"])
    provider = create_provider(model.provider, ROOT/".env")
    try:
        if args.mode == "text":
            # Same model/prompt as the user's working api.py, through our wrapper.
            with tqdm(total=1, desc="API text check", unit="request") as progress:
                try:
                    response = Journal(output/"calls").complete("text", provider, model,
                        "Write a one-sentence bedtime story about a unicorn.",
                        tools=[], max_output_tokens=args.max_output_tokens, timeout=60, retries=0)
                except ProviderError as exc:
                    print(dumps({"mode": "text", "error": str(exc), "usage": usage_summary(output),
                                 "artifacts": str(output)}))
                    raise SystemExit(1) from None
                progress.update()
            print(dumps({"mode": "text", "status": response.raw.get("status"), "text": response.text,
                         "usage": usage_summary(output), "artifacts": str(output)}))
            if (response.raw.get("status") != "completed" or response.raw.get("model") != model.id
                    or not response.text.strip()):
                raise SystemExit(1)
            return
        with tqdm(total=1, desc="API smoke episode", unit="episode") as progress:
            result = run_episode(provider, model, level, output, bad_push=Push((2, 3), "up"),
                config=SolverConfig(max_output_tokens=args.max_output_tokens),
                force_first_simulation=args.force_first_simulation, retry_uncertain=args.retry_uncertain)
            progress.update()
        print(dumps({"result": result, "usage": usage_summary(output), "artifacts": str(output)}))
        if (result["status"] not in {"correct", "unsolved", "illegal"}
                or args.force_first_simulation and result["tool_calls"] < 1):
            raise SystemExit(1)
    finally:
        provider.close()


if __name__ == "__main__": main()
