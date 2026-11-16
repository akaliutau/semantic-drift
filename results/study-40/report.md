# Semantic Gate Audit

40 source topics; 4 judges. 40/40 rewrites passed the construction guard.

| Judge | Condition | Altered approved | Valid control approved | Denial approved | Missing |
|---|---|---:|---:|---:|---:|
| gemma-4-31b-it | bare | 11/39 | 39/39 | 0/39 | 1 |
| gemma-4-31b-it | rewritten | 32/37 | 37/37 | 0/37 | 3 |
| gemma-4-31b-it | checklist | 4/39 | 39/39 | 0/39 | 1 |
| gemma-4-31b-it | diff | 25/38 | 38/38 | 0/38 | 2 |
| glm-5.3-flash | bare | 0/37 | 37/37 | 0/37 | 3 |
| glm-5.3-flash | rewritten | 5/39 | 38/39 | 0/39 | 1 |
| glm-5.3-flash | checklist | 0/40 | 37/40 | 0/40 | 0 |
| glm-5.3-flash | diff | 5/40 | 39/40 | 0/40 | 0 |
| deepseek-v4-flash | bare | 5/37 | 37/37 | 0/37 | 3 |
| deepseek-v4-flash | rewritten | 33/39 | 39/39 | 0/39 | 1 |
| deepseek-v4-flash | checklist | 1/38 | 38/38 | 0/38 | 2 |
| deepseek-v4-flash | diff | 26/38 | 38/38 | 0/38 | 2 |
| qwen3.8-27b | bare | 0/39 | 39/39 | 0/39 | 1 |
| qwen3.8-27b | rewritten | 10/40 | 40/40 | 0/40 | 0 |
| qwen3.8-27b | checklist | 0/40 | 40/40 | 0/40 | 0 |
| qwen3.8-27b | diff | 4/40 | 40/40 | 0/40 | 0 |

Paired contrasts compare the same source topics. Holm correction covers all configured judges and both primary contrasts. Full intervals, discordant pairs and missing-outcome bounds are in summary.json.

- gemma-4-31b-it:rewording_effect: n=37, corrected p=3.33786e-06, eligible for confirmation=False.
- gemma-4-31b-it:diff_repair: n=37, corrected p=0.0390625, eligible for confirmation=False.
- glm-5.3-flash:rewording_effect: n=36, corrected p=0.1875, eligible for confirmation=False.
- glm-5.3-flash:diff_repair: n=39, corrected p=1, eligible for confirmation=True.
- deepseek-v4-flash:rewording_effect: n=36, corrected p=4.76837e-07, eligible for confirmation=False.
- deepseek-v4-flash:diff_repair: n=37, corrected p=0.125, eligible for confirmation=False.
- qwen3.8-27b:rewording_effect: n=39, corrected p=0.0117188, eligible for confirmation=True.
- qwen3.8-27b:diff_repair: n=40, corrected p=0.291992, eligible for confirmation=True.

Construction checks verify an exact unquoted assertion; they do not prove arbitrary prose semantics. Labels have single-assistant reference review. This is a curated challenge set, not a population estimate. A significant contrast alone does not establish novelty or deployment impact.

Elapsed: 33.5 min; new API attempts: 644; cached requests: 40; cost recorded: 0.473402; unknown-cost attempts: 4.
