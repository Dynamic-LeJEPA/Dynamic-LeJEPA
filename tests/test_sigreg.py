import torch
from dlejepa.sigreg import SIGRegPlus


def test_perfect_gaussian_low_loss():
    sig = SIGRegPlus(256, 256, 1.0, 1.0)
    assert sig(torch.randn(64, 256)).item() < 0.5


def test_correlated_collapse_penalized():
    """v1's blind spot: rank-1 with unit marginals must now be punished."""
    sig = SIGRegPlus(256, 256, 1.0, 1.0)
    g = torch.randn(64, 1)
    s = torch.randint(0, 2, (1, 256)) * 2.0 - 1.0
    assert sig(g * s).item() > 1.0


def test_single_dim_collapse():
    sig = SIGRegPlus(256, 256, 1.0, 1.0)
    z = torch.zeros(64, 256)
    z[:, 0] = torch.randn(64) * 10
    assert sig(z).item() > 5.0
