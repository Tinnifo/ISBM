import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The methods
    The foundational goal is to learn or approximate the intractable denominator fo bayes theoreom

    $$P(A \mid B) = \frac{P(B \mid A)\,P(A)}{P(B)}$$

    OR

    $$P(H \mid D) = \frac{P(D \mid H)\,P(H)}{P(D)}$$

    - $P(H)$ is the prior
    - $P(D \mid H)$ is the likelihood
    - $P(H \mid D)$ is the posterior, where $P(D) \neq 0$

    The latest reasearch focuses on the following

    **MCMC**
    - Metropolis-Hastings
    - Gibbs sampling
    - blocked gibbs sampling
    - collapsed gibbs sampling
    - Hamiltonian
    - NUTS

    **Variational**
    - Variationa Inference
    - Guides

    **Energy-Based**
    - EBMs

    **Fow-Based**
    - Normalising flows

    I will be writing the textbook version of these methods using Murphys book Probabilsitic Machine Learning
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The Data

    I will generate synthetic data using the Ising model since it allows us to control important parameters of the data generation
    - the Ising model is described in more detail in the notebook called "ising_model.ipynb"
    """)
    return


if __name__ == "__main__":
    app.run()
