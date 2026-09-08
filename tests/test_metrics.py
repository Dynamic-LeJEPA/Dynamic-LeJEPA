import numpy as np
from dlejepa.metrics import compute_effective_dim_from_embeddings


def test_prop_vi1_rank_floor():
    """When N < K, even a TRUE isotropic Gaussian shows depressed d_eff
    (Proposition VI.1) — the reason Phase 1 (N/K < 5) cannot test isotropy."""
    rng = np.random.RandomState(0)
    m = compute_effective_dim_from_embeddings(
        rng.randn(64, 256).astype(np.float32))
    assert m["d_eff"] < (64 - 1) / 256 + 0.05


def test_reliable_at_nk_ge_5():
    rng = np.random.RandomState(0)
    m = compute_effective_dim_from_embeddings(
        rng.randn(5000, 256).astype(np.float32))
    assert m["d_eff"] > 0.75


def test_correlated_collapse_detected():
    """The Phase 3 blind spot: perfect marginals, rank-1 joint."""
    g = np.random.RandomState(1).randn(4096, 1)
    s = np.random.RandomState(2).randint(0, 2, (1, 256)) * 2.0 - 1.0
    m = compute_effective_dim_from_embeddings((g * s).astype(np.float32))
    assert m["h_ratio"] > 0.9      # marginals look perfect...
    assert m["d_eff"] < 0.05       # ...but the joint is collapsed
