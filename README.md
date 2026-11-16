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

The decisive [evidence-revision follow-up](EVIDENCE_REVISION_REPORT.txt) rejected
the stronger persistence claim: Gemma corrected both false rejections after
native execution evidence; ordinary rechecking did not. Qwen stayed correct.
Nine calls including a declared tool-interface repair cost 0.05001168 credits.
Offline preparation: `python -m selftrap.evidence_revision --output runs/new-evidence`.

For the missing-abstraction direction, the [game review](CONCEPT_GAME_REVIEW.txt)
compares routing, sliding, topology, programming and history-dependent games.
Ordered dots with walls and mini Ricochet Robots are the recommended first
testbeds. Existing puzzle benchmarks constrain the novelty claim; these games
have not yet established a model abstraction failure in our experiments.

Kaggle model-style pilot (isolated research branch): the token-backed download,
prompt-disjoint CPU baselines and a four-call masking attempt are described in
[the report](KAGGLE_STYLE_PILOT_REPORT.txt). Source attribution works above chance;
the masked rewrites changed length and meaning, so they do not establish a
style-only effect or an epistemic finding.

```bash
python -m pip install -e '.[test,style-lab]'
python -m selftrap.style_lab --download --output runs/kaggle/new-baseline
python -m selftrap.style_mask_pilot --baseline runs/kaggle/new-baseline --output runs/kaggle/new-masks
```

The final command prepares the fixed four-passage design offline; `--live`
executes at most four requests under a 0.1-credit local guard and the original
shared ceiling. Kaggle credentials are read from `.env` by its official SDK.
Raw competition data, trained classifiers and request receipts stay in `runs/`.

The [provenance trust pilot](PROVENANCE_TRUST_PILOT_REPORT.txt) found no decision
failure in two model families: editor identity and authentication left copied
claim confidence unchanged. All 22 checking choices, including controls and a
separate diagnostic, were correct. The predeclared gate stopped larger runs.

```bash
python -m selftrap.provenance_pilot --output runs/new-provenance
python -m selftrap.provenance_diagnostic --output runs/new-provenance-diagnostic
```

Both commands prepare sealed designs offline. `--live` runs the corresponding
18-call screen or four-call diagnostic under the original shared budget.

The [consensus and style-control pilots](CONSENSUS_STYLE_AUDIT_REPORT.txt) found
no effect of equivalent wording on evidence counting in the controlled cases.
Generic meaning checks did admit altered claims; an explicit commitment check
reduced those errors. This is a validator result, not a replicated bias estimate.

The [executable-counterfeit study](EXECUTABLE_COUNTERFEIT_RESEARCH.txt) implements
no. 4 as an adversarial variant of provenance-based trust. Two models generated
working swap/search programs. Neither achieved a targeted style-label flip on
six transfer passages, so epistemic trust probes did not pass their entry gate.
The report distinguishes initial interface failures, unreachable edit spaces,
expanded contextual synonyms, simple search baselines and the remaining novelty
requirements. All research remains isolated from the main branch.

```bash
python -m selftrap.consensus_pilot --output runs/new-consensus
python -m selftrap.style_control_audit --output runs/new-style-control-audit
python -m selftrap.executable_mask_pilot --output runs/new-executable-mask
```

These commands seal designs offline. `--live` admits the bounded API jobs.
`expanded_swap_audit` evaluates frozen programs offline with a reviewed edit bank.
