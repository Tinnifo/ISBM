import random

import numpy as np


def set_global_seed(seed):
    """Seed Python's `random` and NumPy's legacy global RNG.

    Local samplers should still take their own seed; this is a safety net for
    any library code that reaches for the global RNG.
    """
    random.seed(seed)
    np.random.seed(seed)
