"""isbm: infinite stochastic block models (IRM) and their inference.

Modules:
    isbm.irm      single-domain IRM generator (generate_irm_graph)
    isbm.cgs      collapsed Gibbs sampler for the two-domain IRM (CGS_IRM)
    isbm.metrics  ARI and held-out log-likelihood (needs scikit-learn)
"""

from isbm.cgs import CGS_IRM
from isbm.irm import generate_irm_graph

__all__ = ["CGS_IRM", "generate_irm_graph"]
