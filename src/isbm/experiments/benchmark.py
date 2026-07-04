"""Benchmark the delayed-acceptance IRM sampler against the baseline collapsed
Gibbs sampler.

Runs the baseline ``CGS_IRM`` and one or more ``DA_IRM`` variants on the *same*
synthetic graph(s) and reports two things:

* Correctness parity -- ARI, inferred cluster count ``K`` and posterior mean
  pseudo log-likelihood. Because the delayed-acceptance kernel targets the exact
  same posterior, these should agree with the baseline within Monte-Carlo noise.
* Efficiency -- wall-clock time, number of exact ``gammaln`` marginal
  evaluations, effective sample size (ESS) of the pseudo-LL trace, ESS per
  second, and the delayed-acceptance stage-2 acceptance rate.

Usage::

    PYTHONPATH=src python -m isbm.experiments.benchmark
    PYTHONPATH=src python -m isbm.experiments.benchmark --N 300 --n-iter 400 --seeds 3
"""

import argparse
import time

import numpy as np

from isbm.data.synthetic import generate_irm_graph
from isbm.metrics.clustering import ari, effective_sample_size
from isbm.models import CGS_IRM, DA_IRM, SMC_IRM


def _heldout_pairs(N1, N2, frac, rng):
    n_pairs = int(frac * N1 * N2)
    i_idx = rng.integers(0, N1, size=n_pairs)
    j_idx = rng.integers(0, N2, size=n_pairs)
    return i_idx, j_idx


def _run_one(cls, kwargs, X, Z, i_test, j_test):
    model = cls(**kwargs)
    t0 = time.perf_counter()
    model.fit(X)
    wall = time.perf_counter() - t0

    post_ll = model.ll_trace_[model.burnin :]
    ess = effective_sample_size(post_ll)
    result = {
        "wall_s": wall,
        "ari": ari(Z, model.z1_),
        "K": int(model.z1_.max() + 1),
        "mean_ll": float(np.mean(post_ll)),
        "pred_ll": float(model.predict(X, i_test, j_test)),
        "ess": ess,
        "ess_per_s": ess / wall if wall > 0 else float("nan"),
        "exact_evals": int(getattr(model, "n_exact_evals_", 0)),
        "acc": float(getattr(model, "acceptance_rate_", float("nan"))),
    }
    return result


def _aggregate(runs):
    keys = runs[0].keys()
    return {k: (float(np.mean([r[k] for r in runs])), float(np.std([r[k] for r in runs]))) for k in keys}


def run_benchmark(
    N=200, n_iter=300, burnin=None, seeds=3, heldout_frac=0.1, data_seed=0, alpha=5.0
):
    if burnin is None:
        burnin = n_iter // 2

    configs = [
        ("CGS (baseline)", CGS_IRM, {}),
        ("DA plugin", DA_IRM, {"surrogate": "plugin"}),
        ("DA subsample 0.5", DA_IRM, {"surrogate": "subsample", "subsample_frac": 0.5}),
        ("DA subsample 0.25", DA_IRM, {"surrogate": "subsample", "subsample_frac": 0.25}),
        ("DA learned (linear)", DA_IRM, {"surrogate": "learned", "learned_regressor": "linear"}),
    ]

    common = dict(n_iter=n_iter, burnin=burnin)

    aggregated = {}
    for label, cls, extra in configs:
        runs = []
        for s in range(seeds):
            X, Z = generate_irm_graph(N=N, alpha=alpha, seed=data_seed + s)
            rng = np.random.default_rng(1000 + s)
            i_test, j_test = _heldout_pairs(N, N, heldout_frac, rng)
            kwargs = dict(common, seed=s, **extra)
            runs.append(_run_one(cls, kwargs, X, Z, i_test, j_test))
        aggregated[label] = _aggregate(runs)

    _print_table(aggregated, N, n_iter, seeds, alpha)
    return aggregated


def _print_table(aggregated, N, n_iter, seeds, alpha):
    baseline = aggregated["CGS (baseline)"]
    base_ll = baseline["mean_ll"][0]
    base_wall = baseline["wall_s"][0]
    base_evals = baseline["exact_evals"][0]

    print(f"\nIRM delayed-acceptance benchmark  (N={N}, n_iter={n_iter}, seeds={seeds}, alpha={alpha})")
    print("Means over seeds. speedup/eval_x are relative to the CGS baseline.")
    print("ll_gap = baseline_mean_ll - this_mean_ll  (>0 means worse fit / poorer mixing).\n")

    header = (
        f"{'sampler':<20} {'wall_s':>8} {'speedup':>8} {'ari':>7} {'K':>5} "
        f"{'mean_ll':>10} {'ll_gap':>9} {'ess':>7} {'exact_evals':>12} {'eval_x':>7} {'acc':>7}"
    )
    print(header)
    print("-" * len(header))
    for label, a in aggregated.items():
        wall_m = a["wall_s"][0]
        speedup = base_wall / wall_m if wall_m > 0 else float("nan")
        ll_gap = base_ll - a["mean_ll"][0]
        eval_x = base_evals / a["exact_evals"][0] if a["exact_evals"][0] > 0 else float("nan")
        acc = a["acc"][0]
        acc_str = f"{acc:>7.3f}" if not np.isnan(acc) else f"{'-':>7}"
        ess = a["ess"][0]
        ess_str = f"{ess:>7.1f}" if not np.isnan(ess) else f"{'nan':>7}"
        print(
            f"{label:<20} {wall_m:>8.3f} {speedup:>8.2f} {a['ari'][0]:>7.3f} "
            f"{a['K'][0]:>5.1f} {a['mean_ll'][0]:>10.1f} {ll_gap:>9.1f} {ess_str} "
            f"{a['exact_evals'][0]:>12.0f} {eval_x:>7.2f} {acc_str}"
        )
    print()
    _print_verdict(aggregated, base_ll)


def _print_verdict(aggregated, base_ll):
    print("Verdict (delayed-acceptance vs exact Gibbs):")
    tol = max(1.0, abs(base_ll) * 0.01)  # 1% of baseline LL as a parity tolerance
    for label, a in aggregated.items():
        if label.startswith("CGS"):
            continue
        ll_gap = base_ll - a["mean_ll"][0]
        speedup = aggregated["CGS (baseline)"]["wall_s"][0] / a["wall_s"][0]
        if ll_gap > tol:
            quality = f"WORSE mixing (ll_gap={ll_gap:.1f} > tol={tol:.1f})"
        else:
            quality = f"matches baseline (ll_gap={ll_gap:.1f})"
        speed = f"{speedup:.2f}x wall-clock"
        improves = (ll_gap <= tol) and (speedup > 1.0)
        tag = "IMPROVES" if improves else "does not improve"
        print(f"  - {label:<18}: {tag}. {quality}; {speed}.")
    print()


def run_smc_benchmark(
    N=150, seeds=3, n_particles=200, n_rejuv=1, alpha=5.0, data_seed=0, gibbs_iter=300
):
    """Compare collapsed Gibbs against SMC-over-data (exact and surrogate-accelerated)."""
    smc_configs = [
        ("Gibbs (CGS)", CGS_IRM, {"n_iter": gibbs_iter, "burnin": gibbs_iter // 2}),
        ("SMC exact", SMC_IRM, {"n_particles": n_particles, "proposal": "exact", "n_rejuv": n_rejuv}),
        ("SMC surrogate", SMC_IRM, {"n_particles": n_particles, "proposal": "surrogate", "n_rejuv": n_rejuv}),
    ]

    rows = {}
    for label, cls, extra in smc_configs:
        runs = []
        for s in range(seeds):
            X, Z = generate_irm_graph(N=N, alpha=alpha, seed=data_seed + s)
            model = cls(seed=s, **extra)
            t0 = time.perf_counter()
            model.fit(X)
            wall = time.perf_counter() - t0
            runs.append(
                {
                    "wall_s": wall,
                    "ari": ari(Z, model.z1_),
                    "K": int(model.z1_.max() + 1),
                    "logZ": float(getattr(model, "logZ_", float("nan"))),
                    "exact_evals": int(getattr(model, "n_exact_evals_", 0)),
                }
            )
        rows[label] = _aggregate(runs)

    print(f"\nSMC-over-data benchmark  (N={N}, seeds={seeds}, particles={n_particles}, rejuv={n_rejuv}, alpha={alpha})")
    header = f"{'sampler':<16} {'wall_s':>8} {'speedup':>8} {'ari':>7} {'K':>5} {'logZ':>10} {'exact_evals':>12} {'eval_x':>7}"
    print(header)
    print("-" * len(header))
    base_wall = rows["Gibbs (CGS)"]["wall_s"][0]
    base_evals = rows["Gibbs (CGS)"]["exact_evals"][0]
    for label, a in rows.items():
        wall_m = a["wall_s"][0]
        speedup = base_wall / wall_m if wall_m > 0 else float("nan")
        logz = a["logZ"][0]
        logz_str = f"{logz:>10.1f}" if not np.isnan(logz) else f"{'-':>10}"
        eval_x = base_evals / a["exact_evals"][0] if a["exact_evals"][0] > 0 else float("nan")
        print(
            f"{label:<16} {wall_m:>8.3f} {speedup:>8.2f} {a['ari'][0]:>7.3f} "
            f"{a['K'][0]:>5.1f} {logz_str} {a['exact_evals'][0]:>12.0f} {eval_x:>7.2f}"
        )
    print()
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--N", type=int, default=200, help="number of nodes per domain")
    p.add_argument("--n-iter", type=int, default=300, help="Gibbs sweeps")
    p.add_argument("--burnin", type=int, default=None, help="burn-in sweeps (default n_iter/2)")
    p.add_argument("--seeds", type=int, default=3, help="number of repeated seeds")
    p.add_argument("--heldout-frac", type=float, default=0.1, help="fraction of entries scored for predictive LL")
    p.add_argument("--data-seed", type=int, default=0, help="base seed for the synthetic graph")
    p.add_argument("--alpha", type=float, default=5.0, help="CRP concentration of the synthetic graph (higher => more, smaller clusters)")
    p.add_argument("--smc", action="store_true", help="also run the SMC-over-data comparison")
    p.add_argument("--n-particles", type=int, default=200, help="SMC particle count")
    p.add_argument("--n-rejuv", type=int, default=1, help="SMC rejuvenation sweeps per resample")
    args = p.parse_args()
    run_benchmark(
        N=args.N,
        n_iter=args.n_iter,
        burnin=args.burnin,
        seeds=args.seeds,
        heldout_frac=args.heldout_frac,
        data_seed=args.data_seed,
        alpha=args.alpha,
    )
    if args.smc:
        run_smc_benchmark(
            N=args.N,
            seeds=args.seeds,
            n_particles=args.n_particles,
            n_rejuv=args.n_rejuv,
            alpha=args.alpha,
            data_seed=args.data_seed,
            gibbs_iter=args.n_iter,
        )


if __name__ == "__main__":
    main()
