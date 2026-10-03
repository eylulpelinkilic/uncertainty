"""
Uncertainty computation utilities.

All functions are designed to work with pandas Series and use consistent
epsilon handling to prevent numerical issues.
"""

import numpy as np
import pandas as pd

EPS = 1e-12
N_BINS = 20


def kl_divergence(p, q, eps=EPS):
    """
    D_KL(p || q)
    """
    union_idx = p.index.union(q.index) # combine all unique values from both p and q
    p, q = p.reindex(union_idx, fill_value=0), q.reindex(union_idx, fill_value=0) # ensures both series have the same indices. Missing ones are replaced by 0
    return float(np.sum(p * np.log2((p + eps) / (q + eps))))

def get_distribution(series):
    # get the PMF of a feature
    return series.value_counts(normalize=True)

def discretise(series, n_bins=N_BINS):
    # PMF might be near continuous so get_distribution would create 1/N tiny spikes in the dist.
    ## seperate into n_bins bins
    if series.nunique() > n_bins*2: # if more than n_bins*2 unique values
            return pd.qcut(series, q=n_bins, duplicates="drop")
    return series

def shannon_entropy(p, eps=EPS):
    """H(p)  (base-2).  `p` is a pandas Series whose values sum to 1."""
    return float(-np.sum(p * np.log2(p + eps)))

def js_divergence(p, q, eps=EPS):
    """
    Jensen-Shannon divergence.
    Symmetric, bounded in [0, 1] when log base is 2.
    """
    union = p.index.union(q.index)
    p, q = p.reindex(union, fill_value=0), q.reindex(union, fill_value=0)
    m = 0.5 * (p + q)
    kl_p_m = np.sum(p * np.log2((p + eps) / (m + eps)))
    kl_q_m = np.sum(q * np.log2((q + eps) / (m + eps)))
    return 0.5 * (kl_p_m + kl_q_m)
