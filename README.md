# Semantic Gate Audit

**Can rewording hide a factual change from a model that rejects the same change when shown plainly?**

Example: “a grandchild **may** inherit…” becomes “a grandchild **will** inherit…”.
The stronger assertion stays verbatim while the surrounding report is rewritten.
A judge must decide whether this still qualifies as a style-only rewrite.

The contribution is a paired audit of the **meaning-preservation gate itself**:
fixed changes, fresh source topics, multiple judge families, and cheap repairs.
It tests a prerequisite of experiments that attribute score changes to writing
style. It does not claim that all published style effects are invalid.

Exploratory evidence: Gemma approved 5/6 altered rewrites; GLM approved 1/6.
Those six rewrites cover four topics. **This is a pilot, not confirmation.**

## Run once

```bash
python -m pip install -e '.[test]'
# Set ADAPTION_API_KEY in .env, or export it in your shell.
python -m pipeline plan -n 30
python -m pipeline run -n 30 --output results/study-30
```

N counts independent source topics; the bundled dataset supports **1–40**.
The default judges are Gemma 4 31B, GLM 5.3 Flash, DeepSeek V4 Flash, and
Qwen 3.8 27B. Edit `config.json` or select a subset:

```bash
python -m pipeline run -n 40 --models gemma-4-31b-it,glm-5.3-flash --output results/two-judges
```

Six workers, at most two calls per model, global request/token throttling,
and at most **three total attempts** per request. Transient API failures retry;
429s honor Retry-After and slow the whole queue. Model answers, malformed
submissions, and incomplete generations are never retried to obtain a better result.

The default budget is **110 minutes**, including an analysis reserve; 120 is the
maximum. `plan` estimates time before spending, and tqdm shows actual progress.
A hard deadline saves partial results rather than exceeding the budget.
Keep the same `cache/` across runs to preserve the cumulative spending limit.

## What is measured

Each topic has four matched conditions: the bare factual edit, the broader
rewrite, the rewrite with a commitment checklist, and the rewrite with an
automatic word diff. Each call also contains a heading-only valid control and
an explicit denial. Candidate order is fixed across conditions and randomized
across topics. Rewrites are frozen before any judge sees them.

Primary contrasts: rewrite versus bare edit; rewrite versus word diff.
The analysis uses exact paired McNemar tests, Holm correction across all judges
and primary contrasts, paired topic bootstrap intervals, and missing-outcome
bounds. A confirmation label requires N ≥ 20, ≥95% complete pairs, and passing
positive/negative controls. Models are reported separately.

## Everything needed for analysis stays local

`results/study-30/` contains the frozen cases/config, source-code snapshot and
SHA256 hashes, complete prompts and raw API responses, failures/retries,
`observations.csv`, `summary.json`, and `report.md`. Only essential timing,
token, cost, and decision data are recorded. No API rerun is needed:

```bash
python -m pipeline analyze results/study-30
```

`share/` contains a small audit subset you can commit. It includes the full-run
summary, clearly distinguished from the subset. The complete local cache is
ignored by Git; three preexisting generation receipts are bundled in
`examples/cache/` and reused only when the request hash matches exactly.

The dataset is a curated stress test with single-assistant reference annotation.
Construction checks verify retention of an unquoted changed assertion; they do
not prove the semantics of arbitrary prose. Additional wording changes may
contribute to errors, so the paired effect does not isolate typography alone.
Independent human review and a larger confirmation remain necessary for broad
claims. See [data attribution](data/README.md).

Related work: [BITE](https://arxiv.org/abs/2605.26156) studies style attacks and
semantic controls; [Epistemic Erosion](https://aclanthology.org/2026.aacl-long.223/)
studies uncertainty loss. Our target is false approval by the validity check.

Checks: `python -m pytest -q`. Provider-specific code is confined to
`pipeline/providers.py` and `pipeline/streaming.py`.
