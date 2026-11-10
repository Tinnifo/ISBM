# Header cell for every new marimo notebook. Copy as the first two cells of
# Notebooks/FEW-<id>_<slug>.py and fill in the constants.
import marimo

app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import subprocess

    LINEAR_ISSUE = "https://linear.app/tinnifo/issue/FEW-000"
    HYPOTHESIS = "FEW-000: <one-line claim>"
    SEED = 0
    try:
        COMMIT = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        DIRTY = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    except Exception:
        COMMIT, DIRTY = "unknown", True
    return COMMIT, DIRTY, HYPOTHESIS, LINEAR_ISSUE, SEED, mo


@app.cell(hide_code=True)
def _(COMMIT, DIRTY, HYPOTHESIS, LINEAR_ISSUE, SEED, mo):
    mo.md(
        f"""
    **Linear:** [{LINEAR_ISSUE.rsplit('/', 1)[-1]}]({LINEAR_ISSUE}) · **Hypothesis:** {HYPOTHESIS}
    · **Commit:** `{COMMIT}`{' (dirty working tree!)' if DIRTY else ''} · **Seed:** {SEED}
    """
    )
    return


if __name__ == "__main__":
    app.run()
