# Target and current search

Target: a novel, clear AI epistemics study with a credible path to the rubric's
15/15. A fresh toy game alone does not qualify. A candidate must identify a new
failure mechanism, verify it mechanically, beat obvious alternative explanations,
and survive a small fresh replication. No score is guaranteed before evidence.

Current candidate: **Does a model know what its harness makes it do?**
A model invents game mechanics. A compiler constructs solvable worlds with
certified misleading shortcuts. Before acting, each model predicts its own and
a peer's action distributions under two *exactly disclosed* native API contracts.
Only tool_choice changes; inputs, tool schema and budgets match. Test whether
self-models predict this intervention and whether a self-diagnosis predicts a
repair that actually works. Parser failures and token exhaustion remain separate.

Initial probe: both targets committed a certified greedy error. Gemma predicted
its own actual move with .60 probability, peer .45; Qwen self .25, peer .50.
That failed its predeclared self-advantage gate. Native tool_choice was not fully
specified in those forecast instructions, so it cannot confirm the new claim.
The new experiment uses three fresh states and fresh forecasts, twelve actors.
No original outcome is relabelled. Raw protocols and all rejected rounds are
kept in ignored runs/new-search/; active source retains reusable components.

Further approaches, only if this candidate fails:

- **Executable self-diagnosis.** Replace prose explanations with a small
  executable predicate predicting exactly where *this model* will fail. Test
  its precision on fresh invented worlds, against peer predicates and a
  model-agnostic difficulty predictor. A diagnosis that fits its training trace
  but fails under a semantics-preserving change is rejected.
- **Forecast the cure.** Before a known mistake recurs, ask whether the model's
  proposed diagnostic intervention will help itself or only its peer. Score
  predicted treatment effects and actual corrections on held-out worlds.
  An accurate mistake forecast without a useful causal repair is insufficient.

Closest primary work and novelty exclusions:

- [The Token Games](https://arxiv.org/abs/2602.17831): invented verifiable puzzle
  duels. We need model-specific foresight and diagnosis, not invention rankings.
- [Looking Inward](https://proceedings.iclr.cc/paper_files/paper/2025/file/0a6059857ae5c82ea9726ee9282a7145-Paper-Conference.pdf): trained self-prediction;
  generic self-prediction is already established.
- [Evaluating and Improving LLM Self-Modeling](https://arxiv.org/abs/2608.30980)
  and [CHIVE](https://arxiv.org/abs/2608.16747): counterfactual self-modeling and
  explanation simulatability. Prompt-edit forecasts alone would overlap.
- [Self-Authored Verification](https://arxiv.org/abs/2607.24300): self-tests can
  miss deployment regressions. Inventing a poor verifier alone would overlap.
- [Reality Check](https://arxiv.org/abs/2605.26242): public input cues can imitate
  introspection. No privileged-access claim without matched information controls.
- [Masked by Consensus](https://arxiv.org/abs/2604.12373): task/domain and model
  agreement affect apparent self-knowledge. Raw confidence bias is insufficient.

Searches found no direct match for *prospective self-versus-peer prediction of
native tool-choice intervention effects*. This is a provisional novelty claim,
not an exhaustive absence claim. A strong empirical finding is still required.
