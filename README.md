# SelfTrap

Research target: a novel, mechanically checkable test of whether models can
predict and diagnose their own characteristic failures. Candidate ideas pass
a novelty check and a small empirical gate before larger experiments.

Reusable API code lives in `selftrap/`:

| Module | Purpose |
|---|---|
| `providers.py` | Provider interface and official Adaption adapter |
| `streaming.py` | Streaming completion and usage accounting |
| `throttle.py` | Shared pacing, 429 cooldown, credit reservations |
| `runtime.py` | Composition, model IDs, original shared budget |
| `storage.py`, `journal.py` | Atomic receipts and admission-only retry |
| `parsing.py` | Terminal JSON extraction without correcting answers |

Install with `python -m pip install -e '.[test]'`. Local checks:
`python -m pytest -q`. Credentials belong in `.env` as `ADAPTION_API_KEY`.
One explicitly live API check:

```bash
python -m selftrap.smoke --live --model gemma-4-31b-it --output runs/api-check-new
```

Completed receipts are reused. Interrupted requests are not silently repeated.
All paid work reuses the original 190-credit ledger under ignored `runs/`.
Run manifests, response IDs, native tokens, latency, costs, failed candidates
and stopping decisions are recorded. Unknown billing retains its reserve.

See the current [game-selection review](SEARCH.md),
[ordered pilot results](ORDERED_PILOT_REPORT.md),
[failed directions](FAILED_DIRECTIONS.txt), the supplied
[challenge](challenge_descr.txt), and [environment](env_description.txt).
Prior implementations remain recoverable through Git history.

Small maze representation pilot (isolated research branch):

Result: both models passed without feedback; stopped after two requests.
See [the short report](MAZE_PILOT_REPORT.txt) and
[the three-room diagram](examples/maze-abstraction.png). The mechanical
prototype works; this version is too easy to demonstrate the proposed benefit.

```bash
python -m pip install -e '.[test,maze-demo]'
python -m selftrap.maze_pilot --output runs/maze-abstraction-1
python -m selftrap.maze_pilot --output runs/maze-abstraction-1 --live
```

Offline mode prepares a three-room diagram and checked state
collision. Live mode tests two models without feedback first and stops if both
already solve the representation task. Otherwise it compares generic feedback
with the certificate, up to six requests and 0.5 additional credits. The exact
worlds, prompts, source hashes, and stopping rule are sealed in `design.json`
before API calls. Four transfer cases reuse each submitted representation;
they are not independent model trials. Transitions are supplied correctly by
the simulator, so this tests state abstraction rather than full formalization.

Two further small pilots use four-slot comparison programs: search for correct
solutions a model rejects, and model-written programs predicting its own
mistakes. Gemma rejected a correct sorter even with a true exhaustive
certificate. Its peer predicted the same failure; Qwen's peer found a trap
that Qwen's self-selected candidate did not. No self-specific advantage or
winning novelty was established. See [the report](TRUTH_SELFCLONE_PILOT_REPORT.txt)
and [the program diagram](examples/sorting-program.png).

```bash
python -m selftrap.two_directions --output runs/new-sortnet-pilot
```

This prepares checked examples offline. `--live` admits at most ten new
requests and one additional credit under the original shared ledger.
The protocol freezes four discovery cases, 24 offline candidates and the
stopping gates. Only candidates selected before target feedback are confirmed.
Completed targets can continue when another fails to submit. An explicit new
phase can set `--tokens 24000 --seconds 300 --reuse-discovery PATH`; it reuses
only completed responses to identical discovery prompts. The pilot stopped
after ten total admissions, including one interrupted and one incomplete call.
