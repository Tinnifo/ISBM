# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy",
# ]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Gibbs sampling, from scratch

    Worked through with Pólya's four phases. The worked examples are the agent's; the answers and the code are mine.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Understand the problem

    What is the unknown? What is the data? What is the condition?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1a. Warm-up — what the bar means in $p(A \mid B)$

    You keep a diary for 100 days. Each day you note two things: did it rain, and did you carry an umbrella?

    |  | umbrella | no umbrella | total |
    | --- | --- | --- | --- |
    | **rain** | 27 | 3 | 30 |
    | **dry** | 14 | 56 | 70 |
    | **total** | 41 | 59 | 100 |

    The bar means: *keep only the days where the right-hand side is true. Ignore the rest.*

    **Worked:** $p(\text{umbrella} \mid \text{rain})$ — keep only the 30 rainy days.
    On 27 of them you carried an umbrella. So $p(\text{umbrella} \mid \text{rain}) = 27/30 = 0.9$.

    **Your turn:** what is $p(\text{rain} \mid \text{umbrella})$?
    Which days do you keep, and what fraction comes out?
    """)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Devise a plan

    Have I seen a related problem? What is the simplest case?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Devise a plan

    Have I seen a related problem? What is the simplest case?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Carry out the plan

    **Prediction, before running:** I expect ___ because ___.
    """)
    return


if __name__ == "__main__":
    app.run()
