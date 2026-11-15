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
