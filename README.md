# Semantic Gate Audit

**Can rewording make an LLM approve a factual change that it rejects when shown plainly?**

Example: “This code **may** extend beyond physical space” becomes “This code
**will** extend beyond physical space.” The stronger claim remains verbatim
while a model rewrites the surrounding report. A meaning-preservation judge
then decides whether this qualifies as a style-only rewrite.

The contribution is a paired audit of the **validation gate itself**, including
cheap repairs. Style-attack research uses semantic checks
([BITE](https://arxiv.org/abs/2605.26156)); uncertainty loss is already known
([Epistemic Erosion](https://aclanthology.org/2026.aacl-long.223/)). Our question
is whether broader rewording conceals an otherwise detectable commitment change.

Exploratory evidence: Gemma approved **5/6** altered rewrites; GLM **1/6**.
These cover only four topics. [Pilot evidence](examples/pilot.json) is exploratory,
not confirmation or evidence against a published bias estimate.

## Run

```bash
python -m pip install -e '.[test]'
# Set ADAPTION_API_KEY in .env or your shell.
python -m pipeline plan -n 40
python -m pipeline run -n 40 --output results/study-40
```

N counts distinct source-topic clusters; **1–40** are bundled. Four judges are
configured: Gemma 4 31B, GLM 5.3 Flash, DeepSeek V4 Flash, Qwen 3.8 27B.
Edit [config.json](config.json), or pass `--models gemma-4-31b-it,glm-5.3-flash`.

The live API smoke completed **17 calls in 60 seconds**, across all four models,
without retries or 429s. With the local generation cache, N=40 plans **640 judge
calls in about 50 minutes**, including a planning margin. N=30 plans about
38 minutes. These are estimates; tqdm shows actual progress.

Six workers, two calls/model, request/token throttling, shared 429 backoff, and
**three total attempts maximum**. Only transient API failures retry; model
answers, malformed outputs, and incomplete generations do not. Default runtime
is **110 minutes**, with an analysis reserve; **120 minutes is the maximum**.
Partial outcomes are saved at the deadline. Keep the same local `cache/` to
preserve the cumulative spending limit.

## Design

Each topic has four matched conditions: **bare edit, broader rewrite, checklist,
word diff**. Every call also judges a heading-only valid control and an explicit
denial. Candidate order is matched across conditions; rewrites freeze before judging.

Primary contrasts: rewrite–bare and rewrite–diff. Exact paired McNemar tests,
Holm correction across all judges/primary contrasts, topic bootstrap intervals,
and missing-outcome bounds are generated automatically. A confirmation label
requires N≥20, ≥95% complete pairs, and passing controls. Models stay separate.

## Results you can analyse without another API call

The output folder contains frozen cases/config, **source-code snapshot + SHA256**,
full prompts/raw responses, failures/retries, `observations.csv`, `summary.json`,
and `report.md`. Only decisions, timing, token and cost data are collected.

```bash
python -m pipeline analyze results/study-40
```

`share/` is a committable audit subset. The full local cache is Git-ignored;
three generation receipts are bundled in `examples/cache/`. A completed
[end-to-end smoke example](examples/smoke_result/) is also included.

The dataset has single-assistant reference annotation. Verbatim-retention guards
are not complete semantic proofs. Additional content changes can contribute to
the effect; the experiment does not isolate typography alone. See
[data provenance and license](data/README.md).
