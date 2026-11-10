# Notebooks/

marimo notebooks only (`.py`). Existing study notebooks keep their names.
New research notebooks are named `FEW-123_slug.py` after their Linear issue and start
with the header cells from `templates/notebook_header.py` (issue URL, hypothesis,
commit, seed). Run them on the shared box, never on Tinni's Mac (see `experiments/README.md`).

Check before a PR: `uv run marimo export html Notebooks/FEW-123_slug.py -o /tmp/x.html`.
