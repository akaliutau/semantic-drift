PROMPT ASSEMBLY CONTRACT -- DRAFT, NOT FROZEN

All templates use literal {{name}} placeholders. Substitute once from trusted
runner metadata; do not recursively interpret text emitted by a model. Hash
the final rendered prompts and save them with each run. Freeze these files
after the disjoint pilot. Roster IDs and assignments must be factual.

author.txt
  author_model_id: actual provider model ID, never a roleplay identity.
  solver_roster_json: the three fixed IDs and factual configuration labels;
    the same information is provided to every author.
  assignment: TARGET plus the actual target ID, or GENERIC_HARD.
  min_optimal_pushes / max_optimal_pushes: pilot-frozen bounds (draft 6/30).
  solver_contract: rendered solver instructions with an explicit placeholder
    for the board, plus tools, model settings, budgets and the event definition.
    Forecasts are for NO WARNING. Never include another author's predictions.
  proposal_schema_json: contents of schemas/puzzle.schema.json.

solver.txt
  public_puzzle_json: strict allowlist: board, player, boxes, goals and
    legal_initial_pushes. Coordinates are zero-based; action-list order is
    seeded per puzzle/repetition and matched across solvers and warning arms.
  warning_block: empty for baseline. Otherwise "Advice to consider:\n" plus
    the approved warning, with no author/source/condition label.
  Never pass solution witnesses, the bad push, forecasts, author rationale,
    oracle distances, target assignment, identity or validity logs.
  Supply simulate_plan through each provider's native tool interface. The
    adapter enforces a single tool call per response, two calls per episode,
    three responses per episode, plan length and the recorded token budget.
    Preserve and classify contract violations instead of silently repairing
    plans or giving extra turns. Token accounting differs across providers.
  simulate_plan always starts at the original state and stops at its first
    illegal action. Report its index and cause; do not execute later actions.
    Successful final grading requires the entire submitted plan to be legal
    and to end with all boxes on goals.

warning_peer.txt
  verified_puzzle_and_bad_push_json: board/state, verified author witness and
    declared bad push only. Strip author identity, warning, forecasts and
    observed solver outcomes. Use the same information for all peer warnings.
  solver_contract: same no-warning instructions/budgets available to authors.

Warning handling
  Diagnostic warnings are <=40 whitespace-delimited words. The generic text
    is fixed across puzzles. Before outcomes are examined, two reviewers when
    available check diagnostic warnings for direct hints. Record review and
    word count; do not rewrite using observed failures. If a warning violates
    the contract, mark that intervention unavailable; do not drop its puzzle
    from baseline/self-targeting analyses. Report missing intervention counts.
  Warnings can still contain useful solution information: compare author and
    peer warnings, not just against no warning. This intervention alone cannot
    establish self-specific insight. Disclose imperfect length matching.

Scoring contract
  Structural JSON failure is a distinct failure category. If the entire plan
    parses structurally and its first action equals the declared legal bad
    push, predicted_error=true even if a later action is illegal. Tool traces
    never count as committed actions. Full_success requires legal replay of
    the entire final plan and all boxes on goals at the end.
  Keep boolean predicted_error and full_success alongside a separate terminal
    status (completed, malformed, illegal, exhausted, service_failure, etc.).
    Do not force overlapping failure descriptions into one ambiguous label.
  Infrastructure failures are missing observations, not model mistakes. Retry
    at most twice and report remaining missingness by model/condition. Do not
    impute a missing error event as zero. The locked analysis must specify its
    complete-data subset and worst-case sensitivity bounds when needed.

The engine, API adapters and experiment runner are not implemented by these
templates. Use synthetic/mock responses to test rendering, leakage boundaries,
tool limits, event scoring and interrupted-run recovery before live runs.
