# SelfTrap

Can a model construct a puzzle that causes a **particular predicted mistake in
itself**, beyond making puzzles that defeat everyone? This repository implements
the crossed author/solver experiment in `experiment_plan_24h.txt`.

Implementation is complete for the core experiment. **No pilot or main study has
been collected.** One live smoke episode was attempted with
`deepseek-v4-flash`: two Adaption HTTP 504s and a final timeout, with no model
response or token usage. Its local receipts are in `runs/api-smoke/`. Local
tests use tiny fixtures and in-memory providers; they are not research results.
The adapter was then aligned with the official quickstart and verified live on
Qwen: one text request and one tiny tool-enabled puzzle episode both succeeded.
The puzzle used one simulation and returned the correct final solution. This
does not identify the cause of the earlier DeepSeek/tool timeout. No pilot or
main experiment has been run.

## Stages and modules

| Stage | Module | Responsibility |
|---|---|---|
| 1 | `game.py`, `oracle.py`, `validation.py` | Push rules, independent exhaustive search, witness/trap checks, symmetry deduplication |
| 2 | `providers.py` | Provider interface and stateless Adaption SDK adapter |
| 3 | `storage.py`, `solver.py` | Atomic JSON receipts, bounded tool loop, final-plan grading |
| 4 | `generation.py` | Fixed-budget authors, validity-only repair, procedural controls, peer warnings |
| 5 | `experiment.py`, `__main__.py` | Frozen assignments, progress, credit reservation, staged commands, resume |
| 6 | `analysis.py` | Specificity contrast, clustered intervals, forecasts, warning effects |
| 7 | `reporting.py` | JSONL dataset export and readable puzzle replays |

## Installation and local checks

Use the local environment; there is no training, GPU, Modal job, or paid external
API in this implementation. Run commands from this checkout:

```bash
python -m pip install -e '.[test]'
python -m pytest -q
python -m selftrap replay examples/tiny_puzzle.json
```

The supplied fixture takes two pushes, intentionally below the research
difficulty minimum. It is only for implementation checks. Runtime dependency
versions are saved in every manifest. The version pins used during development
are in `requirements-experiment.txt`; the existing cloud requirements are kept
separate.

## API smoke checks

Set `ADAPTION_API_KEY` in `.env` or the process environment, matching `Adaption()`.
The wrapper never writes credentials to traces. This command makes live calls:

```bash
python scripts/test_api.py --live
```

By default this sends **one plain-text Qwen request**, with the same input as
the user's working `api.py`. It uses our wrapper and saves usage/timing receipts.
No tool parameters are sent and no retries run in this mode. The response has a
2,000-token cap and a 60-second timeout.

Test the full puzzle loop explicitly, with automatic tool choice:

```bash
python scripts/test_api.py --live --mode episode
```

Add `--force-first-simulation` to require a native tool call as in the original
smoke attempt. This runs **one puzzle episode**. Bounds: three model
responses, two tool calls, 2,000 generated tokens total, and 180 seconds. A
transport failure can use up to two retries within that deadline. It never
generates a research dataset. A wrong puzzle answer is not retried.

The existing DeepSeek smoke attempt failed at the provider. Defaults now use a
separate directory per mode/model, so it cannot be confused with that cached
failure. Repeating a check with the same output path reuses its saved receipts.
To intentionally make a new attempt, supply a fresh `--output` directory.
Changing `--model`, mode or forcing a simulation also requires a new output
directory. Failed/timeout requests have **unknown billing**, not a
recorded cost of zero.

Verified through the refactored wrapper on 15 November 2026:

| Check | Seconds | Input/output tokens | Estimated credits |
|---|---:|---:|---:|
| Qwen text, one request | 10.48 | 21 / 1,006 | 0.00353255 |
| Qwen puzzle, two requests and one simulation | 18.92 | 3,800 / 1,668 | 0.007928 |

Receipts are in `runs/api-refactor-qwen/` and
`runs/api-refactor-qwen-episode/`. Output totals include reasoning tokens.
These are implementation checks on the two-push fixture, not research results.

The adapter follows the [official quickstart](https://docs.adaptionlabs.ai/inference-stacks/quickstart/)
and [Responses API reference](https://docs.adaptionlabs.ai/api/python/resources/responses/methods/create).
It uses `Adaption(max_retries=0)` with the standard environment configuration,
`responses.create(model=..., input=...)`, and the required output bound. Tools
and tool choice are added when applicable. Direct model calls have no stack
memory; response storage defaults to false. Local function-call limits are
enforced by our solver loop; Adaption's `max_tool_calls` controls its hosted/MCP
loop and is omitted here.

## Experiment commands, for later execution

Nothing below has been run against the API during implementation. Every command
that launches generation or solving defaults to **one job**. `init`, `status`,
`analyse`, `export`, and `replay` are offline.

First edit `configs/pilot.json`: verify the three account-accessible direct
model IDs and freeze provider-native settings. Empty `settings` means the
provider defaults, explicitly recorded as such; it does not mean temperature
zero. Only `temperature` is currently exposed by the adapter. There is no hidden
model substitution, memory stack, arbitrary code execution, or routing.

```bash
python -m selftrap init --config configs/pilot.json --output runs/pilot
python -m selftrap generate runs/pilot --limit 1
python -m selftrap status runs/pilot
```

The pilot has 24 assigned author slots. Repeat generation in explicitly sized
batches when authorized to run it. Each slot permits two proposals total and
accepts the first valid, nonduplicate puzzle. All failed attempts remain saved.
Authors see structural/mechanical errors only, never solver feedback. Search
caps yield `UNKNOWN`, which fails validity rather than proving a deadlock.

Finish generation before solving. For the warning follow-up, generate and
review warnings **before any solver results**:

```bash
python -m selftrap prepare-warnings runs/pilot --limit 1
```

Read the corresponding `runs/pilot/warnings/<slot>/review.json`. A reviewer must
check that a diagnostic warning is at most 40 words and contains no coordinates,
directions, solution prefix, or explicit ban on a particular move. Record the
actual human decision with:

```bash
python -m selftrap review-warning runs/pilot --slot a0-t0-n000 --kind author --decision approve --reviewer YOUR_NAME
python -m selftrap review-warning runs/pilot --slot a0-t0-n000 --kind peer --decision approve --reviewer YOUR_NAME
```

Use `reject` for noncompliant warnings. Unapproved warnings are unavailable;
their puzzles remain in baseline analyses. Reviews after solver outcomes are
rejected. The author/peer source label is withheld from solvers.

```bash
python -m selftrap solve runs/pilot --arm baseline --limit 1
python -m selftrap solve runs/pilot --arm generic --limit 1
python -m selftrap solve runs/pilot --arm author --limit 1
python -m selftrap solve runs/pilot --arm peer --limit 1
```

`--workers` controls simultaneous episode jobs (default 1, maximum 9). The
frozen `concurrency_per_model` further limits requests to each model (1–3).
tqdm shows progress/ETA; every call saves elapsed time and returned usage.
The smoke configuration's forced simulation is never used in research episodes.

Only after the disjoint pilot confirms feasibility, choose the main sample
size and settings, edit `configs/main.json`, and initialize a new run. The
template has 288 author slots, 32 controls, and two solver repeats. Its
`exclude_runs` rejects puzzles already present in `runs/pilot`. The main roster
must be chosen before generation; do not switch a model mid-run.

```bash
python -m selftrap init --config configs/main.json --output runs/main
```

Use the same explicit stage commands with `runs/main`. A larger `--limit N`
authorizes up to N jobs for that invocation. Symmetry robustness is optional:
`solve runs/main --arm symmetry --limit N`. It uses two preselected slots per
author/objective cell and a nonidentity rotation/reflection, with correctly
transformed actions. A fourth-model audit is not implemented in this core run.

## Bounds, persistence and cost

Each solver sees only board/state coordinates and legal pushes. Each
`simulate_plan` starts from the original board; final submission does too.
Exploring a bad move is legitimate. The scored error is a structurally parsed
final plan beginning with the declared legal deadlocking push. A later illegal
move does not erase that first-push event. A successful solution requires the
entire final plan to replay legally and finish with all boxes on goals.

Output limits and request/tool counts are enforced. Missing output usage
consumes the remaining output budget conservatively. Equal nominal limits do
not establish equal compute across models. No tool provides deadlock labels,
optimal distances, reference plans, or author forecasts to the actor.

One run directory contains:

```text
manifest.json                    immutable configuration, assignments, versions
snapshot/                        exact source, prompts, schema and protocol
slots/<slot>/                    proposals, validity certificates, final status
warnings/<slot>/review.json       warning text and blind review decisions
jobs-<arm>.json                   frozen randomized solver jobs
episodes/<job>/job.json           assignment and warning arm
episodes/<job>/episode.json       exact solver prompt and identity hash
episodes/<job>/calls/             timestamped request/response/error JSON
episodes/<job>/tool-*.json        simulation feedback
episodes/<job>/result.json        final plan, exact outcomes and elapsed time
```

Atomic replacements avoid half-written artifacts. A process lock prevents two
commands modifying the same run. Completed jobs/calls are reused. Changed
configurations or source files abort resume. An interrupted in-flight request
with no response/error requires inspection and explicit `--retry-uncertain`;
it is never silently repeated. Completed infrastructure failures stay recorded
after the fixed retry budget; rerunning a stage does not erase them.

The credit cap uses returned usage and uncached catalog rates. Before sending a
call, the guard reserves input bytes plus requested output tokens; unknown
billing remains reserved. This is a planning guard, not an invoice guarantee.
It is local to each run, so cap allocations across pilot/main must fit the
same sponsor balance. No credit-to-USD conversion is assumed.

## Analysis and export

```bash
python -m selftrap analyse runs/main --pilot runs/pilot --output results/main
python -m selftrap export runs/main --output data/release
python -m selftrap replay runs/main/slots/SLOT_ID/result.json --plan runs/main/episodes/JOB_ID/result.json
```

Analysis writes JSON and CSV, without an API or model judge. It reports:

- Target-specific trap frequency and the predeclared diagonal-minus-off-diagonal
  contrast D; invalid construction slots receive zero generation credit.
- A 95% bootstrap interval over slots within cells, keeping repeated solver
  runs together. This is uncertainty for these configurations, not all LLMs.
- Brier scores and calibration bins for precommitted error/success forecasts;
  optional disjoint-pilot base-rate comparison.
- Solve/error rates by author, objective and solver, including generic and
  procedural controls; paired warning effects, with Holm correction for the
  three full-solve comparisons; optional symmetry differences.
- Validity yield, terminal failure counts, coverage, runtime, usage and estimated
  cost. Missing infrastructure observations stay missing; the primary estimate
  includes complete panels and reports worst-case missing-data bounds.

An incomplete run is explicitly marked incomplete. The tests' synthetic outcomes
are confined to temporary directories. They must never be presented as findings.
There is no automated paper generator or promise of a positive result.
