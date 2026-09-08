"""Paired statistics: Wilcoxon signed-rank + Holm-Bonferroni (paper Tables VII, X)."""
import numpy as np
from scipy.stats import wilcoxon


def paired_wilcoxon(x, y) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    stat, p = wilcoxon(x, y)
    return {"statistic": float(stat), "p": float(p),
            "delta_median": float(np.median(x - y))}


def holm_bonferroni(pvals) -> np.ndarray:
    """Holm-corrected p-values, returned in the original order."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.minimum.accumulate((n - np.arange(n)) * p[order])
    adj = np.minimum(adj, 1.0)
    out = np.empty(n)
    out[order] = adj
    return out
