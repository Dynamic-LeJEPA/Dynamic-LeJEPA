"""Tests for dlejepa.metrics — the Proposition VI.1 rank floor, the
Corollary VI.2 reliability threshold, and the Phase-3 correlated-collapse
blind spot.

NOTE ON THE RAW-vs-SHRUNK DISTINCTION (the source of the original CI
failure): the (N-1)/K floor of Proposition VI.1 is a property of the RAW
sample covariance (shrinkage=0). The default estimator applies aggressive
shrinkage (gamma = 1 - N/K) when N < K, which fills the K-N+1 null
eigenvalues and PARTIALLY MASKS the floor — a true N(0, I) at N=64, K=256
reads ~0.80 under the default schedule instead of ~0.20. Both behaviors
are pinned below.
"""
import numpy as np

from dlejepa.metrics import compute_effective_dim_from_embeddings


def test_prop_vi1_rank_floor_raw_covariance():
    """Prop VI.1: the RAW sample covariance of N < K samples has rank
    <= N-1, so d_eff <= (N-1)/K even for a TRUE isotropic Gaussian.

    shrinkage=0.0 is passed deliberately (see module docstring).
    """
    rng = np.random.RandomState(0)
    m = compute_effective_dim_from_embeddings(
        rng.randn(64, 256).astype(np.float32), shrinkage=0.0)
    assert m["d_eff"] < (64 - 1) / 256 + 0.05    # bound 0.246; observed ~0.20


def test_prop_vi1_floor_masked_by_default_shrinkage():
    """Pins the shipped default behavior: at N=64 < K, the schedule picks
    gamma = 1 - N/K = 0.75, filling the 193 null eigenvalues with ~0.75
    and lifting a true Gaussian's d_eff to ~0.80. The floor is only
    partially visible under the default estimator — one more reason d_eff
    must not be trusted at N/K < 5 (Cor VI.2).
    """
    rng = np.random.RandomState(0)
    m = compute_effective_dim_from_embeddings(
        rng.randn(64, 256).astype(np.float32))
    assert 0.6 < m["d_eff"] < 0.95               # observed: 0.7975
    assert m["nk_reliable"] is False


def test_cor_vi2_reliability_threshold():
    """Cor VI.2: even with the default (shrunk) estimator, a TRUE Gaussian
    at N/K < 1 reads clearly below its N/K >= 5 value."""
    rng = np.random.RandomState(0)
    low = compute_effective_dim_from_embeddings(
        rng.randn(64, 256).astype(np.float32))       # N/K = 0.25 -> ~0.80
    high = compute_effective_dim_from_embeddings(
        rng.randn(5000, 256).astype(np.float32))     # N/K ~ 20  -> ~0.95
    assert low["d_eff"] < high["d_eff"] - 0.1


def test_reliable_at_nk_ge_5():
    rng = np.random.RandomState(0)
    m = compute_effective_dim_from_embeddings(
        rng.randn(5000, 256).astype(np.float32))
    assert m["d_eff"] > 0.75
    assert m["nk_reliable"] is True


def test_correlated_collapse_detected():
    """The Phase 3 blind spot: perfect marginals, rank-1 joint."""
    g = np.random.RandomState(1).randn(4096, 1)
    s = np.random.RandomState(2).randint(0, 2, (1, 256)) * 2.0 - 1.0
    m = compute_effective_dim_from_embeddings((g * s).astype(np.float32))
    assert m["h_ratio"] > 0.9      # marginals look perfect...
    assert m["d_eff"] < 0.05       # ...but the joint is collapsed
