# FEW-<id>: <experiment title>

- **Linear:** https://linear.app/tinnifo/issue/FEW-<id>
- **Hypothesis:** FEW-<hyp-id> (<claim>)
- **Notebook:** `Notebooks/FEW-<id>_<slug>.py`
- **Owner / reviewer:** Ezra / Theo

## Prediction (LOCKED, do not edit after the run)
- Locked at commit: `<sha>` on <YYYY-MM-DD>
- Prediction:
- Confidence (%):
- Falsified if:

## Setup
- Where: shared box, clean checkout `/workspace/repos/ISBM` (never Tinni's Mac)
- Commit run from: `<sha>`
- `uv.lock` sha256: `<hash>`
- Command: `uv run marimo export html Notebooks/FEW-<id>_<slug>.py -o /tmp/out.html`
- Data / generator config:
- Seeds:
- Metrics: ARI, NMI, held-out link log-likelihood, R-hat (4 chains), wall-clock
- Compute budget: CPU, ≤ <N> min (HF Jobs only with Tinni's cost approval)

## Results
See `results.json`. Summary table:

| Metric | Value | Prediction |
|---|---|---|

## Verdict
Supported / refuted / inconclusive, and why.

## Notes (dated, append-only)
