# Game selection for self-specific prediction and diagnosis

Design review, 15 November 2026. This is a proposal, not a positive result.
No paid trials were started during this review.

Subsequent status: all four directions have been attempted in the
[ordered pilot](ORDERED_PILOT_REPORT.md). No self-specific failure law passed.
The rationale below is retained as the original design, not a claim that the
experiment succeeded.

Question: can a model discover a reproducible mistake in its own behavior,
predict the specific error on new instances, and predict a repair that works
better for that model than generic advice does?

Keep game rules fixed. Let the model write an instance generator, an error
predictor, and a proposed intervention. Freeze these before drawing new seeds.
The interesting open space is the family of mistakes it discovers.

## What the game must make measurable

- An independent oracle must distinguish insufficient evidence from a wrong
  deduction despite sufficient evidence. A successful guess can still have an
  unjustified claim of certainty.
- Actions and intermediate claims should be small and exactly checkable.
  Long action lists mix reasoning with memory, formatting, and execution.
- Generators must produce valid instances cheaply. Require uniqueness only
  when the question demands it: a queried fact can be determined even when
  the entire hidden world is not.
- Prefer inverse puzzles: sample a solution or hidden world, derive truthful
  clues, then independently check what those clues determine. This makes
  construction easy while leaving inference challenging.
- Support targeted interventions: additional evidence, computation on existing
  evidence, or an equivalent representation. Record which kind each supplies.
- Error mechanisms need distinguishing cases. One wrong answer fits many
  explanations. A diagnosis must predict cases where alternatives disagree,
  then predict a repair before observing its effect.

## Games selected by those requirements

| Game | What a model-written program generates | Diagnostic advantage |
|---|---|---|
| **Mastermind: first pilot** | Secret codes and truthful guess/feedback transcripts, filtered by an exhaustive candidate-set oracle | Separates inconsistent answers, premature certainty, and failure to combine clues. Repeated colors test feedback multiplicity. Tiny outputs and finite enumeration make controls cheap. |
| **Black Box: toy laboratory** | Hidden atom layouts and initial ray observations under fixed ray physics | Separates a wrong simulator, a wrong inverse inference, and insufficient experiments. New rays distinguish candidate explanations. A 4x4 board with two atoms has 120 candidate layouts. |
| **Minesweeper: constraint-integration replication** | Minefields and reveal masks, with certified safe/mine/undetermined queries | Separates local shortcuts from joint deductions. Score whether safety follows from every consistent board, as well as whether the actual cell contains a mine. |
| **Zendo: rule-discovery extension** | Executable secret rules and labeled examples over a bounded scene language | Supports alternative hypotheses and diagnostic experiments. Check semantic equivalence over the stated domain. Especially close existing scientific-discovery work. |

Secondary choices: Nonograms offer row/column integration and exact symmetry
controls; Slitherlink/Bridges offer local-rule satisfaction versus global
connectivity. Use these if a specific failure law appears. Sokoban/Rush Hour
require more work to separate the relevant mistake from search depth and state
tracking. Flappy Bird and Minecraft add implementation and observation costs.
These rankings are design judgments, not measured model rankings.

## Minimal example

Mastermind has two positions and alphabet {A,B}. Guess AA receives one
correct-position peg and zero misplaced pegs. Both AB and BA remain possible.
Guess AB gives feedback (2,0) in the first world and (0,2) in the second.
This example was checked by exhaustive enumeration locally.

Three claims are independently scoreable: which worlds remain possible,
whether certainty is justified, and which experiment resolves uncertainty.
This is an illustration, not an expected model failure.

The actual author must precommit to a stronger prediction: on its generated
family, fresh copies of A will incorrectly eliminate candidate X; a particular
change will repair A more than B. Actors see only the public puzzle; hidden
solutions and author forecasts remain outside their context.

## Related designs and novelty limits

- [MastermindEval](https://arxiv.org/abs/2503.05891) already evaluates interactive
  play and inference from unique-solution transcripts. Its
  [official code](https://github.com/flairNLP/mastermind) is a starting point.
- [Minesweeper reasoning](https://aclanthology.org/2024.naacl-long.4/) and
  [graceful fallback in Minesweeper](https://arxiv.org/abs/2610.20401) already
  exist. Failure or safe stopping in that game is insufficient novelty.
- [ZendoWorld](https://arxiv.org/abs/2607.08233) and
  [earlier Zendo experiments](https://arxiv.org/abs/2402.06025) already study
  rule induction and informative experimentation.
- [The Token Games](https://arxiv.org/abs/2602.17831) uses verifiable model-created
  puzzle duels. [AI Crucible](https://github.com/dogfood-lab/ai-crucible) is a
  related software design with same-family designer/solver roles and
  cross-family difficulty comparisons. Its README is not independent evidence
  of research effectiveness.
- [Hide and Seek](https://arxiv.org/abs/2508.03396) already generates subtle
  mathematical reasoning errors adversarially for diagnosis.
- [CHIVE](https://arxiv.org/abs/2608.16747) already evaluates explanations through
  counterfactual behavioral predictions. Prompt edits plus plausible diagnosis
  are insufficient novelty.
- [Strangers to Themselves](https://arxiv.org/abs/2609.09899) finds that much
  self-report predicts generic assistant behavior. Self-versus-peer controls
  are central to this question.

[Black Box's existing implementation](https://www.chiark.greenend.org.uk/~sgtatham/puzzles/js/blackbox.html)
is a demo reference. Its author's
[generation notes](https://www.chiark.greenend.org.uk/~sgtatham/puzzles/devel/writing.html)
explain a scoring issue: different physical layouts can be indistinguishable
under every ray. Accept observational equivalence when appropriate; never
require an unknowable hidden arrangement.

## Next empirical gate

Use Mastermind to minimize engineering and measurement uncertainty. Both
models receive the same discovery records and may author generators targeting
either model. Freeze generators, exact error forecasts, repair forecasts,
prompts, budgets, and parsing before held-out trials. Run both targets on every
selected instance. Add generic-agent forecasts and historical-rate baselines.

Hold the target fixed when comparing forecasts: does A predict A better than B
predicts A, and does B predict B better than A predicts B? Differences in actor
ability are insufficient. Test precommitted intervention effects against
generic or shuffled advice on fresh instances. Self advantage is an empirical
question; a stronger observer may outperform both.

A tiny gate can justify confirmation work. It cannot establish significance
after adaptive searching. Retain all failed candidates and use sealed fresh
confirmation cases after selection; distinct seeds must produce genuinely
distinct instances. Transfer the same protocol to Black Box only if this gate
shows promise. Otherwise close the candidate.

Potential contribution: a certified, generative test of model-specific failure
laws and prospective repairs. The search has not established priority for that
complete protocol. A new game, a generic null about introspection, or an
isolated self-trap does not by itself justify the target 15/15 score.
