# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.24",
#     "numpy",
#     "matplotlib",
#     "SciencePlots",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="What problem do Gibbs and VI solve?")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # What problem do Gibbs and VI solve?
    **ISBM · research-direction, understanding the landscape · four people, A to D**

    /// note | The problem

    We see who is linked to whom. **Which grouping of the people produced it?** Bayes' rule
    gives the exact answer, but one step of it cannot be computed for any real network. Gibbs
    sampling and variational inference (VI) are two ways around that step.

    ///

    /// tip | How to use this notebook

    This is a [marimo](https://marimo.io) notebook. Open it with
    `marimo edit ad_posterior_walkthrough.py`. If the cells show no output yet, run them all with
    ⌘⇧R.

    - **It's reactive.** Change a dropdown and everything that depends on it updates by itself.
    - **The slow step waits for a button.** Section 7's long chains run only when you press it.
    - **The code is there if you want it.** The cells that define the method show their code.
      The rest hide it to keep the story readable; click a cell and press ⌘H to show it. Edit
      any cell, and the cells that depend on it rerun.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    def figure(content, caption):
        """``content`` above a muted, centred markdown caption."""
        return mo.vstack([
            content,
            mo.md(caption).style(color="var(--muted-foreground)", text_align="center", font_size="0.9rem"),
        ])

    return (figure,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What this notebook shows

    We take a network small enough to solve exactly: four people, two links. We write down
    every possible grouping, score each one, and get the exact posterior. Then we see why the
    same recipe fails for twenty people, and how a Gibbs sampler gets the posterior without
    the step that fails.

    **What you'll take away**

    1. The unknown is a grouping, and the Bayesian answer is a probability for **every** grouping.
    2. Bayes' rule divides each grouping's score by the sum of **all** scores. The Bell number
       counts the terms in that sum, and it explodes.
    3. In a ratio of two groupings the sum cancels. Gibbs uses this: each step compares 2–4 scores.
    4. Counting how often a Gibbs chain visits each grouping estimates the posterior, but only
       after enough steps, and a short chain still remembers where it started.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The problem, framed

    ### Pólya: unknown, data, condition

    | | For A–D | Where it comes from |
    | --- | --- | --- |
    | **Unknown** | The posterior over groupings: one probability for each of the 15 groupings | Settled with Tinni in the phase-1 thread: the unknown is a partition, and the Bayesian answer is a probability for every partition |
    | **Data** | The link table: A–B and C–D linked, the other 4 pairs not | The toy we chose together |
    | **Condition** | The IRM story: CRP groups ($\alpha = 1$), a link probability per pair of groups (Beta(1, 1)), coin flips | The model; section 1 |
    | **Related problem** | The Ising model: Gibbs sampling there also moves one site at a time using only score ratios | Her own route into Gibbs sampling (agent mapping) |

    Pólya's four phases, and where each happens below:

    1. **Understand the problem**: sections 1–2 (the data, the model, the unknown).
    2. **Devise a plan**: sections 3–5. The direct plan, score every grouping and divide by the
       sum, fails as $n$ grows. The related problem suggests the way out: work with ratios, where
       the sum cancels.
    3. **Carry out the plan**: sections 6–7 (one Gibbs step, then a chain).
    4. **Look back**: compare the chain with the exact posterior, then check it *without* the
       exact posterior (do several chains agree?). The "Your turn" question is a look-back question.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Thinking in systems: the Gibbs chain as stock and flow

    /// note | Agent draft

    The diagram and the leverage table are the agent's reading, for Tinni to confirm or change.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(ACCENT, STYLE, figure, plt):
    from matplotlib.patches import FancyBboxPatch

    def _box(ax, x, y, w, h, text, color="0.2", lw=1.0, ls="-"):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                    fc="white", ec=color, lw=lw, ls=ls))
        ax.text(x, y, text, ha="center", va="center", color=color)

    def _arrow(ax, a, b, text=None, color="0.2", ls="-"):
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", lw=1.0, color=color, ls=ls,
                                                        shrinkA=2, shrinkB=2))
        if text:
            ax.annotate(text, xy=((a[0] + b[0]) / 2, (a[1] + b[1]) / 2), xytext=(0, 6),
                        textcoords="offset points", ha="center", va="bottom", color=color, fontsize=7)

    with plt.style.context(STYLE):
        _fig, _ax = plt.subplots(figsize=(6.2, 2.5), constrained_layout=True)
        _box(_ax, 0.8, 1.0, 1.15, 0.42, "start\ngrouping")
        _box(_ax, 3.0, 1.0, 1.35, 0.42, "current\ngrouping", color=ACCENT, lw=1.4)
        _box(_ax, 5.4, 1.0, 1.35, 0.42, "visit tally\n(the stock)", lw=1.4)
        _box(_ax, 5.4, -0.15, 1.6, 0.36, "exact posterior\n(check only)", color="0.5", ls="--")
        _arrow(_ax, (1.38, 1.0), (2.32, 1.0), "begins at")
        _ax.annotate("", xy=(3.35, 1.22), xytext=(2.65, 1.22),
                     arrowprops=dict(arrowstyle="-|>", lw=1.0, color=ACCENT, connectionstyle="arc3,rad=-1.6",
                                     shrinkA=0, shrinkB=0))
        _ax.text(3.0, 1.78, "each step: pick a person,\nmove them in proportion to score", ha="center",
                 va="bottom", color=ACCENT, fontsize=7)
        _arrow(_ax, (3.68, 1.0), (4.72, 1.0), "+1 visit (inflow)")
        _ax.text(6.25, 1.0, "share = tally / steps", ha="left", va="center", fontsize=7)
        _arrow(_ax, (5.4, 0.15), (5.4, 0.78), color="0.5", ls="--")
        _ax.text(5.5, 0.46, "compare", ha="left", va="center", color="0.5", fontsize=7)
        _ax.text(4.2, 0.62, "no arrow back: the tally\nnever changes the moves", ha="center", va="top",
                 color="0.45", fontsize=7, style="italic")
        _ax.set_xlim(0.1, 7.6)
        _ax.set_ylim(-0.45, 2.3)
        _ax.set_axis_off()
    figure(_fig, "The **stock** is the visit tally; its only **flow** is one visit per step. The start "
                 "sets where the tally begins to fill, so it dominates until enough visits pile up "
                 "(burn-in is that **delay**). There is **no loop** from the tally back to the moves: "
                 "Gibbs does not learn. The exact posterior only checks the tally and never drives it.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Leverage points**: what could be changed, from weakest to strongest (Meadows):

    | Leverage | In the Gibbs chain | What it changes |
    | --- | --- | --- |
    | **Parameter** | Number of steps, the seed, the start | How long until the chains agree, not where they end up |
    | **Feedback structure** | The move set: which groupings are one step apart. One person at a time here; merge–split moves (Jain and Neal, 2004) move many people at once | How fast the chain forgets its start and crosses between good groupings |
    | **Goal** | The target itself: the model and its prior ($\alpha$, $a$, $b$). VI also changes the goal: it swaps the posterior for a simpler distribution fitted by optimisation | Where the tally converges to |

    One rule from the systems check holds here: **the evaluation signal stays out of the method.**
    The exact posterior is used only to check the chain, never inside a step.
    """)
    return


@app.cell(hide_code=True)
def _():
    import random
    from collections import Counter
    from itertools import combinations
    from math import comb, exp, factorial, lgamma, prod

    import matplotlib.pyplot as plt
    import numpy as np
    import scienceplots  # noqa: F401  (must be imported before plt.style)

    STYLE = ["science", "no-latex"]
    ACCENT = "#0072B2"  # one accent colour; context in grey
    return (
        ACCENT,
        Counter,
        STYLE,
        comb,
        combinations,
        exp,
        factorial,
        lgamma,
        np,
        plt,
        prod,
        random,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. The data and the model

    **Data.** Four people. Of the 6 possible pairs, only A–B and C–D are linked.

    **Model.** The infinite relational model (IRM) tells a story for how such a table is made:

    1. A Chinese restaurant process (CRP) with concentration $\alpha$ puts the people into groups.
    2. Each pair of groups gets a link probability $\theta$, drawn from Beta$(a, b)$.
    3. For each pair of people, a coin with probability $\theta$ decides whether they are linked.

    Inference runs the story backwards: given the links, which groupings are plausible?
    """)
    return


@app.cell
def _():
    NODES = ("A", "B", "C", "D")
    LINKS = {("A", "B"), ("C", "D")}  # each pair written with the earlier letter first
    ALPHA = 1.0  # CRP concentration
    A_BETA, B_BETA = 1.0, 1.0  # Beta prior on each pair of groups' link probability
    return ALPHA, A_BETA, B_BETA, LINKS, NODES


@app.cell(hide_code=True)
def _(ACCENT, LINKS, NODES, STYLE, combinations, figure, plt):
    _pos = {"A": (0, 1), "B": (1, 1), "C": (0, 0), "D": (1, 0)}
    with plt.style.context(STYLE):
        _fig, _ax = plt.subplots(figsize=(2.2, 2.0), constrained_layout=True)
        for _i, _j in combinations(NODES, 2):
            _linked = (_i, _j) in LINKS
            _ax.plot(*zip(_pos[_i], _pos[_j]), color=ACCENT if _linked else "0.85",
                     lw=2.2 if _linked else 0.8, ls="-" if _linked else ":", zorder=1)
        for _n, (_x, _y) in _pos.items():
            _ax.scatter([_x], [_y], s=520, color="white", edgecolor="0.2", zorder=2)
            _ax.text(_x, _y, _n, ha="center", va="center", zorder=3)
        _ax.set_xlim(-0.35, 1.35)
        _ax.set_ylim(-0.35, 1.35)
        _ax.set_axis_off()
    figure(_fig, f"The data. Solid lines are the {len(LINKS)} links; dotted lines are the "
                 f"{6 - len(LINKS)} pairs that are not linked.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. The unknown is a grouping

    A grouping splits the people into groups. Group names do not matter:
    `A=1, B=1, C=2, D=2` and `A=2, B=2, C=1, D=1` are the same grouping, {AB}{CD}. That is why
    $4^4 = 256$ label assignments overcount.

    The function below lists every grouping once. It lists the groupings of everyone but the
    first person, then puts the first person into one of those groups or into a new group.
    """)
    return


@app.cell
def _(NODES):
    def all_groupings(people):
        """Yield every grouping of `people` exactly once (as lists of lists)."""
        if not people:
            yield []
            return
        first, rest = people[0], people[1:]
        for smaller in all_groupings(rest):
            for i in range(len(smaller)):  # join an existing group
                yield smaller[:i] + [[first] + smaller[i]] + smaller[i + 1 :]
            yield [[first]] + smaller  # or start a new group

    def canon(blocks):
        """One fixed way to write a grouping, so equal groupings compare equal."""
        return tuple(sorted(tuple(sorted(b)) for b in blocks if b))

    GROUPINGS = [canon(g) for g in all_groupings(list(NODES))]
    return GROUPINGS, canon


@app.cell(hide_code=True)
def _():
    def show(g):
        return "".join("{" + "".join(b) + "}" for b in g)

    def shape(g):
        return "+".join(str(s) for s in sorted((len(b) for b in g), reverse=True))

    return shape, show


@app.cell(hide_code=True)
def _(GROUPINGS, figure, mo, shape, show):
    _by_shape = {}
    for _g in GROUPINGS:
        _by_shape.setdefault(shape(_g), []).append(show(_g))
    _rows = "\n".join(
        f"| {s} | {', '.join(gs)} | {len(gs)} |"
        for s, gs in sorted(_by_shape.items(), key=lambda kv: (-max(map(int, kv[0].split('+'))), -min(map(int, kv[0].split('+')))))
    )
    figure(
        mo.md("| Shape | Groupings | How many |\n| --- | --- | --- |\n" + _rows
              + f"\n| **Total** | | **{len(GROUPINGS)}** |"),
        "Every grouping of four people, sorted by group sizes. For 2+2, fix A and choose its "
        "partner: three choices, three groupings.",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. The count explodes

    The number of groupings of $n$ people is the **Bell number** $B(n)$. It satisfies
    $B(n+1) = \sum_k \binom{n}{k} B(k)$: choose which $k$ of the other $n$ people are *not* in the
    newcomer's group, and group them in any way.
    """)
    return


@app.cell
def _(comb):
    def bell_numbers(n_max):
        b = [1]
        for n in range(n_max):
            b.append(sum(comb(n, k) * b[k] for k in range(n + 1)))
        return b

    BELL = bell_numbers(100)
    return (BELL,)


@app.cell(hide_code=True)
def _(ACCENT, BELL, GROUPINGS, STYLE, figure, mo, np, plt):
    assert BELL[4] == len(GROUPINGS)

    def _duration(seconds):
        for unit, size in [("years", 3.156e7), ("days", 86400), ("hours", 3600), ("minutes", 60)]:
            if seconds >= size:
                return f"{seconds / size:,.1f} {unit}"
        return f"{seconds:.6f} s"

    with plt.style.context(STYLE):
        _fig, _ax = plt.subplots(figsize=(3.6, 2.6), constrained_layout=True)
        _n = np.arange(1, 21)
        _ax.semilogy(_n, [BELL[k] for k in _n], "o-", color="0.5", ms=2.5, lw=1)
        _ax.semilogy([4], [BELL[4]], "o", color=ACCENT, ms=5)
        _ax.annotate("A–D: 15", xy=(4, BELL[4]), xytext=(9, -3), textcoords="offset points",
                     ha="left", va="center", color=ACCENT)
        _ax.set_xlabel("number of people, n")
        _ax.set_ylabel("number of groupings, B(n)")
        _ax.set_xticks([1, 5, 10, 15, 20])
        _ax.margins(x=0.05, y=0.15)

    _rows = "\n".join(f"| {n} | {BELL[n]:,} | {_duration(BELL[n] / 1e6)} |" for n in [4, 6, 10, 15, 20])
    _table = mo.md(
        "| $n$ | $B(n)$ | to score them all |\n| --- | --- | --- |\n" + _rows
        + f"\n| 100 | ≈ {BELL[100]:.1e} | ≈ {BELL[100] / 1e6:.0e} s |"
    )
    figure(
        mo.hstack([_fig, _table], widths=[3, 2], align="center"),
        f"The count grows faster than exponentially. At a million scores per second, twenty people "
        f"take **{_duration(BELL[20] / 1e6)}**; a hundred take about {BELL[100] / 1e6:.0e} s, "
        f"against a universe about 4e17 s old.",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. The score of a grouping, and Bayes' rule

    Each grouping $z$ gets a **score**, prior × likelihood. Bayes' rule turns scores into
    probabilities:

    $$p(z \mid \text{data}) = \frac{\text{score}(z)}{\sum_{\text{all groupings } z'} \text{score}(z')}$$

    Bayes says **what** to compute. The Bell number says **how many terms** the bottom sum has.

    /// details | Where the prior and the likelihood come from

    - **Prior (the CRP).** With $K$ groups of sizes $n_1, \dots, n_K$ among $n$ people,
      $p(z) = \alpha^{K} \prod_k (n_k - 1)! \,/\, \big(\alpha(\alpha+1)\cdots(\alpha+n-1)\big)$.
      With $\alpha = 1$ the bottom is $4! = 24$, so one big group gets $3!/24 = 0.25$.
    - **Likelihood.** For each pair of groups, count the linked pairs $n_1$ and the unlinked
      pairs $n_0$ between them. Integrating out that pair's $\theta$ gives
      $B(a + n_1, b + n_0) / B(a, b)$, with $B$ the Beta function. Multiply over pairs of groups.
    ///
    """)
    return


@app.cell
def _(ALPHA, A_BETA, B_BETA, GROUPINGS, LINKS, NODES, combinations, exp, factorial, lgamma, prod):
    def crp_prior(g):
        n = sum(len(b) for b in g)
        return ALPHA ** len(g) * prod(factorial(len(b) - 1) for b in g) / prod(ALPHA + i for i in range(n))

    def beta_fn(x, y):
        return exp(lgamma(x) + lgamma(y) - lgamma(x + y))

    def likelihood(g):
        group_of = {person: k for k, block in enumerate(g) for person in block}
        counts = {}  # (group k, group l) -> [linked pairs, unlinked pairs]
        for i, j in combinations(NODES, 2):
            c = counts.setdefault(tuple(sorted((group_of[i], group_of[j]))), [0, 0])
            c[0 if (i, j) in LINKS else 1] += 1
        return prod(beta_fn(A_BETA + n1, B_BETA + n0) / beta_fn(A_BETA, B_BETA) for n1, n0 in counts.values())

    SCORE = {g: crp_prior(g) * likelihood(g) for g in GROUPINGS}  # all 15, computed once
    TOTAL = sum(SCORE.values())  # the sum that cannot be computed for real networks
    POSTERIOR = {g: s / TOTAL for g, s in SCORE.items()}
    return POSTERIOR, SCORE, TOTAL, crp_prior, likelihood


@app.cell(hide_code=True)
def _(GROUPINGS, POSTERIOR, SCORE, TOTAL, crp_prior, figure, likelihood, mo, show):
    figure(
        mo.ui.table(
            [{"grouping": show(g), "prior": round(crp_prior(g), 4), "likelihood": round(likelihood(g), 5),
              "score": round(SCORE[g], 6), "posterior = score / sum": round(POSTERIOR[g], 3)}
             for g in sorted(GROUPINGS, key=lambda g: -SCORE[g])],
            selection=None, pagination=False,
        ),
        f"The exact posterior. The sum of all {len(GROUPINGS)} scores is **{TOTAL:.6f}**; each "
        f"posterior is a score divided by it, so the last column adds to 1.",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. In a ratio, the sum cancels

    How much more likely is one grouping than another?

    $$\frac{p(z_1 \mid \text{data})}{p(z_2 \mid \text{data})}
      = \frac{\text{score}(z_1) / \textstyle\sum}{\text{score}(z_2) / \textstyle\sum}
      = \frac{\text{score}(z_1)}{\text{score}(z_2)}$$

    Two scores are enough. Pick any two groupings:
    """)
    return


@app.cell(hide_code=True)
def _(GROUPINGS, mo, show):
    _names = {show(g): g for g in GROUPINGS}
    pick_z1 = mo.ui.dropdown(options=_names, value="{ABCD}", label="grouping z1")
    pick_z2 = mo.ui.dropdown(options=_names, value="{AB}{CD}", label="grouping z2")
    mo.hstack([pick_z1, pick_z2], justify="start", gap=2)
    return pick_z1, pick_z2


@app.cell(hide_code=True)
def _(POSTERIOR, SCORE, mo, pick_z1, pick_z2):
    _z1, _z2 = pick_z1.value, pick_z2.value
    mo.md(
        "| ratio z1 / z2 | value |\n| --- | --- |\n"
        f"| from the posteriors (needs the sum) | {POSTERIOR[_z1]:.4f} / {POSTERIOR[_z2]:.4f} = **{POSTERIOR[_z1] / POSTERIOR[_z2]:.4f}** |\n"
        f"| from the two scores only | {SCORE[_z1]:.6f} / {SCORE[_z2]:.6f} = **{SCORE[_z1] / SCORE[_z2]:.4f}** |"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. One Gibbs step

    Gibbs builds on the ratio. It picks up **one person**, who can join any group of the others
    or sit alone. It scores only those few options and picks one at random, in proportion to
    its score. The sum it needs has 2–4 terms, never all of them.
    """)
    return


@app.cell
def _(SCORE, canon):
    def gibbs_options(g, person):
        """Where `person` can move from grouping g: (new grouping, score, move probability)."""
        others = [b for b in (tuple(p for p in b if p != person) for b in g) if b]
        options = [canon(others[:i] + [others[i] + (person,)] + others[i + 1 :]) for i in range(len(others))]
        options.append(canon(others + [(person,)]))  # sit alone
        scores = [SCORE[o] for o in options]
        return [(o, s, s / sum(scores)) for o, s in zip(options, scores)]

    return (gibbs_options,)


@app.cell(hide_code=True)
def _(GROUPINGS, NODES, mo, show):
    _names = {show(g): g for g in GROUPINGS}
    pick_current = mo.ui.dropdown(options=_names, value="{AB}{CD}", label="current grouping")
    pick_person = mo.ui.dropdown(options=list(NODES), value="D", label="person picked up")
    mo.hstack([pick_current, pick_person], justify="start", gap=2)
    return pick_current, pick_person


@app.cell(hide_code=True)
def _(figure, gibbs_options, mo, pick_current, pick_person, show):
    _opts = gibbs_options(pick_current.value, pick_person.value)
    figure(
        mo.ui.table(
            [{"move to": show(o), "score": round(s, 6), "move probability = score / sum": round(p, 3)}
             for o, s, p in _opts],
            selection=None, pagination=False,
        ),
        f"One step. The sum of these **{len(_opts)}** scores is {sum(s for _, s, _ in _opts):.6f}, "
        f"and that is all Gibbs needs for this step.",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. A Gibbs chain

    Repeat the step: pick a person at random, move them, write down the grouping. After many
    steps, **the share of visits to each grouping approaches its posterior probability**, so
    counting visits estimates the posterior without the full sum.

    /// details | Two numbers that are easy to mix up

    - **Move probability** (section 6): *if* the chain is at {AB}{CD} *and* picks D, the chance
      that D stays. One step, from one place.
    - **Share of visits**: over the whole run, the fraction of steps spent at {AB}{CD}. In the
      long run that is the posterior.
    ///
    """)
    return


@app.cell
def _(NODES, gibbs_options, random):
    def run_gibbs(start, steps, seed):
        """Random-scan single-site Gibbs. Returns the list of groupings visited."""
        rng = random.Random(seed)
        g, visited = start, []
        for _ in range(steps):
            opts = gibbs_options(g, rng.choice(NODES))
            g = rng.choices([o for o, _, _ in opts], weights=[s for _, s, _ in opts])[0]
            visited.append(g)
        return visited

    return (run_gibbs,)


@app.cell(hide_code=True)
def _(GROUPINGS, mo, show):
    _names = {show(g): g for g in GROUPINGS}
    pick_start = mo.ui.dropdown(options=_names, value="{AC}{BD}", label="start at")
    pick_steps = mo.ui.dropdown(options={"10": 10, "100": 100, "1,000": 1000, "10,000": 10000, "100,000": 100000},
                                value="10", label="steps")
    pick_seed = mo.ui.number(start=1, stop=1000, step=1, value=1, label="seed")
    mo.hstack([pick_start, pick_steps, pick_seed], justify="start", gap=2)
    return pick_seed, pick_start, pick_steps


@app.cell(hide_code=True)
def _(Counter, GROUPINGS, POSTERIOR, figure, mo, pick_seed, pick_start, pick_steps, run_gibbs, show):
    _visited = run_gibbs(pick_start.value, pick_steps.value, int(pick_seed.value))
    _counts = Counter(_visited)
    _first = " ".join(show(g) for g in _visited[:12]) + (" …" if len(_visited) > 12 else "")
    figure(
        mo.vstack([
            mo.md(f"**First visits:** {_first}"),
            mo.ui.table(
                [{"grouping": show(g), "visits": _counts[g], "share of visits": round(_counts[g] / len(_visited), 3),
                  "exact posterior": round(POSTERIOR[g], 3)}
                 for g in sorted(GROUPINGS, key=lambda g: -POSTERIOR[g])],
                selection=None, pagination=False,
            ),
        ]),
        f"One chain of **{len(_visited):,} steps** from {show(pick_start.value)}. Compare the share "
        f"of visits with the exact posterior, then change the number of steps or the seed.",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### How the share settles

    Five chains, each started at {AC}{BD} with a different seed, 100,000 steps each. Each line
    shows the share of visits to {AB}{CD} *so far*.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    run_long = mo.ui.run_button(label="Run five long chains (≈5 s)")
    run_long
    return (run_long,)


@app.cell(hide_code=True)
def _(ACCENT, GROUPINGS, POSTERIOR, STYLE, figure, mo, np, plt, run_gibbs, run_long, show):
    mo.stop(not run_long.value, mo.md("Press the button to run the chains."))

    _target = next(g for g in GROUPINGS if show(g) == "{AB}{CD}")
    _start = next(g for g in GROUPINGS if show(g) == "{AC}{BD}")
    _steps = 100_000
    _x = np.arange(1, _steps + 1)
    with plt.style.context(STYLE):
        _fig, _ax = plt.subplots(figsize=(4.4, 2.8), constrained_layout=True)
        with mo.status.progress_bar(total=5, title="Running chains") as _bar:
            for _seed in range(1, 6):
                _hit = np.fromiter((v == _target for v in run_gibbs(_start, _steps, _seed)), float, _steps)
                _ax.plot(_x, np.cumsum(_hit) / _x, color="0.55", lw=0.8)
                _bar.update()
        _ax.axhline(POSTERIOR[_target], color=ACCENT, ls="--", lw=1.2)
        _ax.annotate(f"exact posterior {POSTERIOR[_target]:.3f}", xy=(_steps, POSTERIOR[_target]),
                     xytext=(0, 12), textcoords="offset points", ha="right", va="bottom", color=ACCENT)
        _ax.set_xscale("log")
        _ax.set_xlabel("Gibbs steps so far")
        _ax.set_ylabel("share of visits to {AB}{CD}")
        _ax.set_ylim(0, 0.6)
        _ax.set_yticks([0, 0.2, 0.4, 0.6])
        _ax.margins(x=0.02)
    figure(_fig, "The x-axis is logarithmic, so the first 10 steps get as much room as the last "
                 "90,000. Short runs scatter; long runs meet the dashed line.")
    return


@app.cell(hide_code=True)
def _(GROUPINGS, figure, mo, np, run_gibbs, run_long, show):
    mo.stop(not run_long.value)
    _target = next(g for g in GROUPINGS if show(g) == "{AB}{CD}")
    _starts = ["{AC}{BD}", "{ABCD}", "{A}{B}{C}{D}", "{AD}{BC}", "{ABC}{D}"]  # spread-out starts
    _by_name = {show(g): g for g in GROUPINGS}
    _hits = np.array([[v == _target for v in run_gibbs(_by_name[_s], 100_000, _i + 1)]
                      for _i, _s in enumerate(_starts)], float)
    _rows = []
    for _n in [10, 100, 1_000, 10_000, 100_000]:
        _sh = _hits[:, :_n].mean(axis=1)
        _rows.append(f"| {_n:,} | {', '.join(f'{x:.3f}' for x in _sh)} | **{_sh.max() - _sh.min():.3f}** |")
    figure(
        mo.md(f"Chains 1–5 start at {', '.join(_starts)}.\n\n"
              "| steps | share of {AB}{CD} in chains 1–5 | spread (max − min) |\n| --- | --- | --- |\n" + "\n".join(_rows)),
        "Do the chains agree? This table never uses the exact posterior, so it works on networks "
        "too big to enumerate. The chains start far apart on purpose: chains that share a start can "
        "agree only because they have not left it. When chains disagree, at least one has not run "
        "long enough. Comparing the spread between chains with the spread within each chain is the "
        "idea behind Gelman and Rubin's (1992) diagnostic.",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Forgetting the start

    One short run is too noisy to show what a short run does on average. So we run thousands of
    short chains from the same start (the **start at** dropdown in section 7) and ask, at each
    step: how often is the chain still at the grouping it started in?

    /// details | Why the starting grouping is too common in short runs

    A step moves one person, so the first few groupings visited are always close to the start.
    Gibbs does not learn: every step uses the same rule and the same scores. What changes with
    more steps is only that the chain gets further from where it began. The steps before it has
    forgotten the start are called **burn-in**; how fast it forgets is called **mixing**.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(ACCENT, POSTERIOR, STYLE, figure, np, pick_start, plt, run_gibbs, show):
    _start, _runs, _steps = pick_start.value, 3000, 40
    _at_start = np.zeros(_steps)
    for _seed in range(_runs):
        _at_start += np.fromiter((v == _start for v in run_gibbs(_start, _steps, 10_000 + _seed)), float, _steps)
    _at_start /= _runs
    _t = np.arange(1, _steps + 1)
    _settled = int(np.argmax(np.abs(_at_start - POSTERIOR[_start]) < 0.01)) + 1
    with plt.style.context(STYLE):
        _fig, _ax = plt.subplots(figsize=(4.2, 2.6), constrained_layout=True)
        _ax.plot(_t, _at_start, "o-", color=ACCENT, ms=2.5, lw=1)
        _ax.axhline(POSTERIOR[_start], color="0.4", ls="--", lw=1)
        _ax.annotate(f"exact posterior {POSTERIOR[_start]:.3f}", xy=(_steps, POSTERIOR[_start]),
                     xytext=(0, 6), textcoords="offset points", ha="right", va="bottom", color="0.3")
        _ax.set_xlabel("step")
        _ax.set_ylabel(f"P(chain is at {show(_start)})")
        _ax.set_ylim(0, _at_start.max() * 1.2)
        _ax.margins(x=0.03)
    figure(_fig, f"{_runs:,} chains started at {show(_start)}. Over a 10-step run, they spend "
                 f"**{_at_start[:10].mean():.3f}** of their visits there on average, against the "
                 f"exact **{POSTERIOR[_start]:.3f}**. Within 0.01 of the posterior from step "
                 f"**{_settled}** on: after that, the start no longer matters.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// attention | Your turn

    Start the chains far apart, then check that they agree. On four people that works within a few
    hundred steps, because every grouping is at most three one-person moves from any other.

    On a big network, picture two very different groupings that both score high. Getting from one
    to the other one person at a time passes through groupings that score very low.

    **Which row of the leverage table would you change so the chain can travel between them, and
    what would the change be?**
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What we saw

    - **The unknown** is a grouping; the Bayesian answer is a probability for every grouping.
    - **Bayes' rule** needs the sum of all scores. **The Bell number** counts its terms: 15 here,
      about $5 \times 10^{13}$ for twenty people.
    - **In a ratio the sum cancels.** A Gibbs step compares only the few places one person can go.
    - **Counting visits** of a long chain recovers the exact posterior. A short chain does not.
    - **A short chain remembers its start**: whichever grouping it starts in is too common. Gibbs
      does not learn; it forgets the start, here within 10–20 steps (burn-in).
    - **Without the exact answer**, run several chains from spread-out starts and check that they
      agree: the spread between them shrinks as they run (Gelman and Rubin, 1992). Chains that
      share a start can agree only because none of them has left it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Going further

    **Try it in this notebook**

    - In section 1, set `ALPHA = 0.2` and then `ALPHA = 5`. Which grouping has the highest
      posterior each time, and why does the CRP prior decide that, not the links?
    - Add a link, say `("B", "C")`, to `LINKS`. How does the posterior table change?

    **Where this leads in the landscape**

    - *Is single-site Gibbs enough?* On real networks there is no exact table to draw the dashed
      line. [Peixoto (2020)](https://doi.org/10.1103/physreve.102.012305) reports that
      single-node moves fail to sample SBM posteriors even on small networks.
    - *Checking convergence.* [Gelman and Rubin (1992)](https://doi.org/10.1214/ss/1177011136)
      compare the spread between chains with the spread within each chain.
    - *What VI does instead.* VI fits a simpler distribution close to the posterior.
      [Blei, Kucukelbir and McAuliffe (2017)](https://doi.org/10.1080/01621459.2017.1285773)
      is the review for statisticians; Ishiguro et al. (2014,
      [arXiv 1409.4757](https://arxiv.org/abs/1409.4757)) apply collapsed VB to the IRM.
    - The vault page `01_wiki/synthesis/landscape-bayesian-posterior-inference.md` lists the
      contested results these ideas lead to.
    """)
    return


if __name__ == "__main__":
    app.run()
