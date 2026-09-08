"""SIGReg regularizers.

* ``SIGReg``    — Phase 1/2 version ("v1, fixed"): marginal variance matching
  + projection sketch + sampled pairwise decorrelation, all ~1.0 scale,
  honest batch diagnostics (no frozen-EMA bug).
* ``SIGRegPlus``— Phase 3 version ("v2"): adds (i) full covariance
  off-diagonal penalty, (ii) denser correlation sampling (500 -> 2048 pairs),
  (iii) projection weight 0.1 -> 1.0. Motivation (paper Sec VII-D2): on
  MimicGen's ~10-D task manifold, v1 converges to CORRELATED COLLAPSE —
  perfect marginals (scale=1.0, H=1.0) with joint covariance rank ~6/256
  (d_eff=0.022). Theorem IV.4(a) needs the JOINT N(0, sigma^2 I), so v2
  enforces joint-level isotropy. Applied IDENTICALLY to all three ablation
  conditions.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .metrics import compute_effective_dim, compute_entropy_ratio


class SIGReg(nn.Module):
    """Phase 1/2 SIGReg (v1: no hidden multipliers, actual-batch diagnostics)."""

    def __init__(self, latent_dim: int, num_projections: int = 256,
                 sigma: float = 1.0, var_weight: float = 1.0):
        super().__init__()
        self.K, self.num_projections, self.sigma2 = latent_dim, num_projections, sigma ** 2
        self.var_weight = var_weight
        projections = F.normalize(torch.randn(num_projections, latent_dim), dim=1)
        self.register_buffer("projections", projections)
        self.log_scale = nn.Parameter(torch.tensor(0.0))
        self._batch_var, self._batch_mean, self._batch_size = None, None, 0

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, K = z.shape
        if B < 2:
            return torch.tensor(0.0, device=z.device)
        self._batch_var = z.var(dim=0).detach().clone()
        self._batch_mean = z.mean(dim=0).detach().clone()
        self._batch_size = B

        var_per_dim = z.var(dim=0)
        per_dim_var_loss = (((var_per_dim - self.sigma2) / self.sigma2) ** 2).mean()
        mean_loss = ((z.mean(dim=0) / self.sigma2 ** 0.5) ** 2).mean()
        var_mean = var_per_dim.mean().clamp(min=1e-6)
        uniformity_loss = (var_per_dim.std() / var_mean) ** 2

        projected = self.projections @ z.T
        proj_var = projected.var(dim=1, unbiased=False)
        projection_loss = (((proj_var - self.sigma2) / self.sigma2) ** 2).mean()

        zc = z - z.mean(dim=0, keepdim=True)
        std = var_per_dim.sqrt().clamp(min=1e-6)
        ii = torch.randint(0, K, (500,), device=z.device)
        jj = torch.randint(0, K, (500,), device=z.device)
        valid = ii != jj
        ii, jj = ii[valid], jj[valid]
        cov_ij = (zc[:, ii] * zc[:, jj]).mean(dim=0)
        corr = cov_ij / (std[ii] * std[jj]).clamp(min=1e-6)
        correlation_loss = (corr ** 2).mean()

        return (self.var_weight * per_dim_var_loss + 0.1 * mean_loss
                + 0.1 * projection_loss + 0.1 * uniformity_loss
                + 0.1 * correlation_loss)

    def get_scale(self) -> float:
        return self.log_scale.exp().item()

    def get_diagnostics(self) -> dict:
        if self._batch_var is None:
            return {"d_eff": 0.0, "H_ratio": 0.0, "scale_ratio": 0.0, "N": 0,
                    "source": "no_data_yet"}
        var = self._batch_var.cpu().numpy()
        return {"d_eff": compute_effective_dim(var, self.K),
                "H_ratio": compute_entropy_ratio(var, self.sigma2, self.K),
                "scale_ratio": float(np.mean(var)) / self.sigma2,
                "var_mean": float(np.mean(var)), "var_std": float(np.std(var)),
                "N": self._batch_size, "source": "actual_batch"}


class SIGRegPlus(SIGReg):
    """Phase 3 SIGReg+ (v2): marginal matching + JOINT isotropy terms."""

    def __init__(self, latent_dim: int, num_projections: int = 256,
                 sigma: float = 1.0, var_weight: float = 1.0,
                 corr_pairs: int = 2048, corr_weight: float = 1.0,
                 cov_weight: float = 1.0, proj_weight: float = 1.0):
        super().__init__(latent_dim, num_projections, sigma, var_weight)
        self.corr_pairs, self.corr_weight = corr_pairs, corr_weight
        self.cov_weight, self.proj_weight = cov_weight, proj_weight
        self._last_cov = None

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, K = z.shape
        if B < 2:
            return torch.tensor(0.0, device=z.device)
        self._batch_var = z.var(dim=0).detach().clone()
        self._batch_mean = z.mean(dim=0).detach().clone()
        self._batch_size = B

        # ---- marginals (unchanged from v1) ----
        var_per_dim = z.var(dim=0)
        per_dim_var_loss = (((var_per_dim - self.sigma2) / self.sigma2) ** 2).mean()
        mean_loss = ((z.mean(dim=0) / self.sigma2 ** 0.5) ** 2).mean()
        var_mean = var_per_dim.mean().clamp(min=1e-6)
        uniformity_loss = (var_per_dim.std() / var_mean) ** 2

        # ---- projection sketch (weight 0.1 -> 1.0) ----
        projected = self.projections @ z.T
        proj_var = projected.var(dim=1, unbiased=False)
        projection_loss = (((proj_var - self.sigma2) / self.sigma2) ** 2).mean()

        # ---- pairwise correlation (500 -> 2048 pairs, weight -> 1.0) ----
        zc = z - z.mean(dim=0, keepdim=True)
        std = var_per_dim.sqrt().clamp(min=1e-6)
        ii = torch.randint(0, K, (self.corr_pairs,), device=z.device)
        jj = torch.randint(0, K, (self.corr_pairs,), device=z.device)
        valid = ii != jj
        ii, jj = ii[valid], jj[valid]
        cov_ij = (zc[:, ii] * zc[:, jj]).mean(dim=0)
        corr = cov_ij / (std[ii] * std[jj]).clamp(min=1e-6)
        correlation_loss = (corr ** 2).mean()

        # ---- NEW: full covariance off-diagonal penalty ----
        cov = (zc.T @ zc) / B
        cov_sq = cov ** 2
        off_diag = cov_sq.sum() - torch.diagonal(cov_sq).sum()
        cov_loss = off_diag / (K * (K - 1) * self.sigma2 ** 2)
        self._last_cov = cov.detach()

        return (self.var_weight * per_dim_var_loss + 0.1 * mean_loss
                + 0.1 * uniformity_loss + self.proj_weight * projection_loss
                + self.corr_weight * correlation_loss + self.cov_weight * cov_loss)

    def get_diagnostics(self) -> dict:
        d = super().get_diagnostics()
        cov_rank = 0.0
        if self._last_cov is not None:
            eig = torch.linalg.eigvalsh(self._last_cov.cpu()).clamp(min=1e-10)
            cov_rank = float((eig.sum() ** 2) / (self.K * (eig ** 2).sum()))
        d["cov_rank"] = cov_rank  # batch-level indicator; final d_eff uses N=20,000
        return d
