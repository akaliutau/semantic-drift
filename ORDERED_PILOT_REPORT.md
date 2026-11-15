# Ordered game pilot — 15 November 2026

**Outcome: no self-specific failure law passed the discovery gate.** We found
real errors and a small forecast lead, but no confirmed personal diagnosis or
behavioral repair. These results do not justify a 15/15 claim or a larger run.

We attempted Mastermind, Black Box, Minesweeper and Zendo in that order. Request
models were `qwen3.5-397b` and `gemma-4-31b-it`; the response IDs were checked
against `qwen3.5-397b-a17b-nvfp4` and `gemma-4-31b-it`. All calls used the existing
Adaption wrapper and original shared credit ledger.

## Method and changes

Authors were asked for a seeded Python generator and forecasts of a specific wrong
answer and a proposed repair. Programs were frozen before fresh seeds. Exact
oracles checked consistency, identifiability and the proposed wrong answer.
Both models then forecast both actors from identical evidence, including the
oracle answer; actors received only rules and public clues. Forecasting versus
solving therefore had different information. Self-versus-peer forecasters had
the same information.

The original gate required a self-authored exact error on both new seeds, with
the peer correct at least once. No author passed. Construction failures were
kept separate from reasoning errors. We fixed overly restrictive pure-Python
validation and a repair prompt that omitted the previous source. Unchanged
submissions were rechecked locally; original records were preserved. Later
construction checked three development seeds to catch constant generators.

When construction blocked Black Box and Zendo, we used small oracle-generated
controls. These do not establish model invention or satisfy the original gate.
A separately declared event distinguished a *possible but unestablished*
answer from an inconsistent answer. That distinction was fixed before new
screening responses; original exact-trap outcomes were not relabeled.

## Findings

| Panel | Complete correct / complete answers | Incomplete actors | Interpretation |
|---|---:|---:|---|
| Mastermind, model-written generators | 5/7 | 1 | Neither author's exact predicted trap occurred. |
| Minesweeper, model-written generators | 3/3 | 1 | No predicted error. Two other held-out generator slots failed validity or uniqueness-of-instance checks. |
| Black Box, oracle controls | 0/2 | 2 | Gemma made both declared epistemic errors; Qwen exhausted 12,000 tokens twice. |
| Zendo, oracle controls | 1/2 | 2 | Gemma gave one unjustified definite answer; Qwen exhausted 12,000 tokens twice. |
| Fresh Mastermind controls | 4/6 | 2 | Gemma's two wrong codes were inconsistent, so neither matched the narrower epistemic event. |

On Black Box's two Gemma outcomes, Gemma's self-forecast Brier loss was 0.05
versus Qwen's 0.34. This is a discovery lead only: both outcomes were errors,
the sample is two selected cases, and Qwen supplied no complete actor answers.
Generic pessimism and forecaster differences remain explanations. On the
available fresh Mastermind outcomes, self-forecasts were worse than peer
forecasts for both targets. No stable advantage was established.

Two originally truncated puzzles were repeated at 12,000 rather than 6,500
tokens, on both models. Both Qwen repeats completed correctly. These repeats are
budget diagnostics, not independent confirmation. Excluded Qwen responses
were checked: they contained only reasoning, no completed answer call, and
reported `max_output_tokens` exhaustion.

Gemma's instant incorrect native calls suggested a harness issue. We forecast
and tested forced versus automatic tool choice on the two selected Mastermind
puzzles, retaining identical prompts and budgets. Gemma was wrong on both
forced repeats and both auto trials. Qwen's auto trials exhausted tokens or
the deadline. Automatic tool choice did not provide the proposed escape.

## Cost, parallelism and reproducibility

- **67 API requests:** 17 construction, 12 forecasting and 38 actor attempts.
- **0.780348 reported credits**, plus **0.135622 uncertain reserve** for an HTTP
  502 and a stream deadline: **0.915970 newly accounted credits**. No automatic
  replay of these uncertain requests occurred.
- Original ledger: **36.472080 accounted / 190 hard cap**; no budget reset.
- Verified **three overlapping requests**, including same-model overlap.
  Starts retained 12-second shared pacing. **No HTTP 429** occurred. Overlap
  establishes accepted concurrent requests, not a provider's maximum capacity.
- **22 offline checks passed.** Raw prompts, code seals, generated cases,
  native usage, responses and stopping decisions remain in ignored
  `runs/new-search/ordered-games-1/`. No requests remain active.
- All new work is isolated on `research/ordered-diagnostic-games`.

The internal call guard was extended once from 60 to 68 for the predeclared
eight-request harness diagnostic. The four-credit round guard and original
shared cap remained unchanged; the amendment is saved before its outcomes.

## Decision

Do not scale these variants or pitch ordinary uncertainty mistakes as novel.
Uncertainty attribution already appears in
[ConfuseBench](https://aclanthology.org/2025.acl-long.840/); counterfactual
explanation testing in [CHIVE](https://arxiv.org/abs/2608.16747), and generic
self-report limitations in [Strangers to Themselves](https://arxiv.org/abs/2609.09899).

The requested contribution still needs a model-discovered executable failure
profile, prediction beyond generic difficulty or bias, and a prospective repair
that transfers to fresh cases. None is established here. Further discovery
should provide direct executable oracle tools to authors and predict completion
separately from incorrect answers. Those are design fixes, not a research win.
