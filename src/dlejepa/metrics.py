"""Distributional metrics for maximum-entropy validation (paper Sec. VI).

Two d_eff estimators (Remark VI.3) — do NOT compare them numerically:

* ``compute_effective_dim``                 — DIAGONAL estimator: Eq. (37)
  applied to the K per-dimension marginal variances. Fast
  sample-complexity sweeps only.
* ``compute_effective_dim_from_embeddings`` — COVARIANCE estimator: full
  K x K sample covariance with shrinkage. Used for EVERY final ablation
  comparison. Subject to the Proposition VI.1 rank floor when N < K
  (reliable iff N/K >= 5, Corollary VI.2).
"""
from typing import Optional

import numpy as np
from scipy.stats import kstest


def compute_entropy_ratio(var_per_dim, sigma2: float = 1.0,
                          K: Optional[int] = None) -> float:
    """H(p) / H(N(0, sigma^2 I_K)) from per-dimension variances."""
    if K is None:
        K = len(var_per_dim)
    var = np.maximum(np.asarray(var_per_dim, dtype=np.float64), 1e-10)
    H_p = 0.5 * np.sum(np.log(2 * np.pi * np.e) + np.log(var))
    H_target = 0.5 * K * np.log(2 * np.pi * np.e * sigma2)
    return float(H_p / H_target) if H_target > 1e-10 else 0.0


def compute_effective_dim(var_per_dim, K: Optional[int] = None) -> float:
    """DIAGONAL estimator: d_eff = (sum var)^2 / (K * sum var^2)."""
    if K is None:
        K = len(var_per_dim)
    var = np.asarray(var_per_dim, dtype=np.float64)
    if len(var) == 0 or np.all(var < 1e-10):
        return 1.0 / K
    var = np.maximum(var, 1e-10)
    sv, svs = np.sum(var), np.sum(var ** 2)
    if svs < 1e-20:
        return 1.0 / K
    return float(np.clip((sv ** 2) / (K * svs), 1.0 / K, 1.0))


def compute_effective_dim_from_embeddings(z: np.ndarray,
                                           shrinkage: Optional[float] = None) -> dict:
    """COVARIANCE estimator (the paper's final-ablation metric).

    Ledoit-Wolf-style shrinkage toward (trace/K) * I. Returns d_eff,
    H-ratio, scale ratio, and per-dimension variance statistics.
    """
    N, K = z.shape
    zc = z - z.mean(axis=0, keepdims=True)
    cov = (zc.T @ zc) / (N - 1) if N > 1 else zc.T @ zc
    if shrinkage is None:
        shrinkage = 0.01 if N > K else (0.05 if N > K // 2 else max(0.1, 1.0 - N / K))
    cov_s = shrinkage * (np.trace(cov) / K) * np.eye(K) + (1 - shrinkage) * cov
    eig = np.maximum(np.linalg.eigvalsh(cov_s), 1e-10)
    s, s2 = eig.sum(), (eig ** 2).sum()
    d_eff = (s ** 2) / (K * s2) if s2 > 1e-20 else 1.0 / K

    var_per_dim = np.var(z, axis=0)
    return {
        "d_eff": float(np.clip(d_eff, 0.0, 1.0)),
        "h_ratio": compute_entropy_ratio(var_per_dim, 1.0, K),
        "scale_ratio": float(np.mean(var_per_dim)),
        "var_mean": float(np.mean(var_per_dim)),
        "var_std": float(np.std(var_per_dim)),
        "shrinkage_used": shrinkage, "N": int(N), "K": int(K),
        "N_over_K": N / K,
    }


def ks_gaussian_stats(z: np.ndarray, max_dims: int = 50) -> dict:
    """Mean/max KS statistic of standardized marginals vs N(0,1)."""
    stats = []
    for d in range(min(z.shape[1], max_dims)):
        col = z[:, d]
        if col.std() > 1e-10:
            stats.append(kstest((col - col.mean()) / (col.std() + 1e-10), "norm")[0])
    if not stats:
        return {"ks_stat_mean": float("inf"), "ks_stat_max": float("inf")}
    return {"ks_stat_mean": float(np.mean(stats)),
            "ks_stat_max": float(np.max(stats))}
