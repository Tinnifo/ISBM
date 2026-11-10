# experiments/

One folder per experiment issue: `experiments/FEW-123/`.

| File | What |
|---|---|
| `record.md` | From `templates/experiment_record.md`. The **prediction is committed before the run** and never edited afterwards. |
| `results.json` | Machine-readable metrics, plus `commit`, `seed(s)`, `uv_lock_sha256`, `started_at`, `wall_clock_s`. |
| `fig_*.png` | Small figures (keep each < ~1 MB). |

## Where experiments run

- **Never on Tinni's Mac.** The Mac is only used for vault notes and calendar entries.
- On the **shared box** (Linux, CPU) in a clean git checkout at `/workspace/repos/ISBM`:

  ```bash
  cd /workspace/repos/ISBM
  git fetch origin && git switch few-123-slug   # or create it from origin/main
  uv sync
  uv run marimo export html Notebooks/FEW-123_slug.py -o /tmp/FEW-123.html
  ```

- No ad hoc copies or snapshots: everything that produced a number is in a commit.
- Outputs are committed through a PR titled `FEW-123: …`. Theo reviews every result PR.
- **Heavy sweeps / GPU:** Hugging Face Jobs only after Tinni explicitly approves the
  estimated cost in the Linear issue (`needs-tinni` until approved).

## Flow

1. Hypothesis issue exists (`type/hypothesis`), experiment sub-issue created (`type/experiment`).
2. Branch `few-123-slug`, copy `templates/experiment_record.md` → `experiments/FEW-123/record.md`,
   fill in the prediction, **commit** (this locks it).
3. Build/run the notebook on the box, write `results.json`, fill in Results and Verdict.
4. PR → Theo review → Tinni merges → result posted in Linear with commit SHA → journal entry in the vault.
