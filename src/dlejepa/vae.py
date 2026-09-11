"""VAE arm — the variational encoder mechanism (paper Secs. VII-C5, VII-D3).

One module per distribution-matching mechanism, parallel to `dlejepa.sigreg`:
SIGReg / SIGReg+ drive the deterministic encoders in `dlejepa.models`; this
module drives the Gaussian (mu, logvar) head with the closed-form marginal
KL(q(z|x) || N(0, I)).

TWO OPERATING POINTS (do not conflate — see `dlejepa.configs`):
  * Phase 2 (nuScenes):  KL weight 0.01 — LEGACY WEAK-PRIOR REGIME, the KL
    term effectively dormant; only the within-arm placement ORDERING is
    interpreted (paper caveat). Evaluated on SAMPLED latents
    (`NuScenesVAELeJEPA` with vae_eval_deterministic=False).
  * Phase 3 (MimicGen):  KL weight 5.0 — CALIBRATED pre-run
    (experiments/phase3_mimicgen/calibrate_vae_kl.py). Primary
    encoder-mechanism test; eval + planning on DETERMINISTIC posterior
    means via `VAEDeterministic`.

The marginal-only KL is DELIBERATE: it instantiates the hypothesis that
correlated collapse is a property of MARGINAL-ONLY regularizers on
low-intrinsic-dimension data (paper Sec. VII-D3). `vae_plus_cov_loss`
provides the optional VAE+ stretch (SIGReg+ covariance penalty), released
for future work.

Posterior signal fraction (paper Remark "Posterior Signal Fraction"):
per dimension, sf = Var_x[mu] / (Var_x[mu] + E_x[sigma^2]). Marginal
diagnostics on SAMPLED latents can look healthy while mu carries almost no
information — collapse hiding beneath the encoder's own sampling-noise
floor. `vae_latent_stats` reports sf, mu_d_eff (covariance d_eff of the
posterior mean, same estimator as `dlejepa.metrics`), and the raw
posterior internals.

Checkpoint compatibility (state-dict key families):
  * VAEEncoder (Phase 3):          patch / cls / pos / blocks / norm / head
  * VAEEncoderNuscenes (Phase 2):  patch_embed / cls_token / pos_embed /
                                   blocks / norm / latent_proj
Both mirror the released `.pt` checkpoints (`encoder_class` field written
by the trainers dispatches the correct constructor).
"""
import numpy as np
import torch
import torch.nn as nn

from .metrics import compute_effective_dim_from_embeddings
from .models import (ActionPredictor, EgoMotionPredictor, PatchEmbed,
                     PhysicsInformedDepthDecoder, ProprioDecoder,
                     TransformerBlock)

__all__ = ["VAEEncoder", "VAEEncoderNuscenes", "VAEDeterministic",
           "vae_kl_loss", "vae_plus_cov_loss", "vae_latent_stats",
           "build_vae_models", "build_vae_models_nuscenes"]


# ============================ losses =========================================

def vae_kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Closed-form D_KL(q(z|x) || N(0, I)) per element, MEANED over batch
    and dims (~O(1) scale, matching this pipeline's loss-scale convention
    and the SIGReg slot it replaces)."""
    kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
    return kl.mean()


def vae_plus_cov_loss(z: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """OPTIONAL VAE+ term: SIGReg+'s covariance off-diagonal penalty,
    applied to sampled latents (paper Sec. VII-D3, VAE+ stretch released
    for future work). Normalized by K(K-1) sigma^4; callers use the
    default sigma=1.0 == cfg.sigma."""
    zc = z - z.mean(dim=0, keepdim=True)
    B, K = zc.shape
    cov = (zc.T @ zc) / max(B - 1, 1)
    off = (cov ** 2).sum() - torch.diagonal(cov ** 2).sum()
    return off / (K * (K - 1) * sigma ** 2)


# ============================ Phase 3: MimicGen ==============================

class VAEEncoder(nn.Module):
    """z_t = f(img_t, proprio_t) with a Gaussian (mu, logvar) head — the
    variational counterpart of `ViTEncoder` (paper Sec. VII-D3).

    Architecturally identical trunk (same patch conv / cls / pos / blocks /
    norm keys as the SIGReg arm); only the final projection differs:
    (2K outputs -> mu, logvar) with reparameterized sampling and a
    numerical-safety logvar clamp (standard VAE practice — prevents logvar
    explosion early in training before the KL has pulled it toward 0)."""

    def __init__(self, img_size=84, patch=14, dim=192, depth=6, heads=3,
                 prop_dim=32, latent=256, logvar_clamp=10.0):
        super().__init__()
        n_p = (img_size // patch) ** 2
        self.patch = nn.Conv2d(3, dim, patch, patch)
        self.cls = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.pos = nn.Parameter(torch.randn(1, n_p + 1, dim) * 0.02)
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads,
                                                      mlp_ratio=2.0)
                                     for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(nn.Linear(dim + prop_dim, 2 * latent),
                                  nn.GELU(), nn.LayerNorm(2 * latent),
                                  nn.Linear(2 * latent, 2 * latent))
        self.logvar_clamp = logvar_clamp

    def trunk(self, img: torch.Tensor, prop: torch.Tensor):
        B = img.shape[0]
        x = self.patch(img).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls.expand(B, -1, -1), x], dim=1) + self.pos
        for b in self.blocks:
            x = b(x)
        stats = self.head(torch.cat([self.norm(x[:, 0]), prop], dim=1))
        mu, logvar = stats.chunk(2, dim=-1)
        logvar = logvar.clamp(-self.logvar_clamp, self.logvar_clamp)
        return mu, logvar

    def forward(self, img: torch.Tensor, prop: torch.Tensor,
                sample: bool = True) -> torch.Tensor:
        mu, logvar = self.trunk(img, prop)
        if sample:
            return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return mu

    def stats(self, img: torch.Tensor, prop: torch.Tensor):
        """Raw (mu, logvar) heads — required by `vae_latent_stats` closures
        (the deterministic shim exposes only mu)."""
        return self.trunk(img, prop)


# ============================ Phase 2: nuScenes ==============================

class VAEEncoderNuscenes(nn.Module):
    """Variational counterpart of `MaxEntropyEncoder` (paper Sec. VII-C5).

    Identical trunk (PatchEmbed / cls_token / pos_embed / blocks / norm);
    `latent_proj` maps the CLS token to 2K outputs -> (mu, logvar).
    `forward` returns (mu, logvar) — or (mu, logvar, patch_tokens) with
    ``return_features=True`` for the `vae_encoder_physics` mode: the
    invalid placement (Cor. IV.11) applies the smoothness prior to the
    trunk's patch features, matching the SIGReg arms' convention in this
    package. Dropout-free state dict — keys match the released Phase 2
    VAE checkpoints exactly."""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=192,
                 depth=6, num_heads=6, latent_dim=256, drop_rate=0.1,
                 logvar_clamp=10.0):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans,
                                      embed_dim)
        n = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)      # no state — key-compatible
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, drop=drop_rate)
            for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.latent_proj = nn.Linear(embed_dim, 2 * latent_dim)
        self.logvar_clamp = logvar_clamp
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def _trunk_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        feat = self._trunk_features(x)
        stats = self.latent_proj(feat[:, 0])
        mu, logvar = stats.chunk(2, dim=-1)
        logvar = logvar.clamp(-self.logvar_clamp, self.logvar_clamp)
        if return_features:
            return mu, logvar, feat[:, 1:]
        return mu, logvar

    def stats(self, x: torch.Tensor):
        """Raw (mu, logvar) heads."""
        return self.forward(x)


# ============================ deterministic shim =============================

class VAEDeterministic(nn.Module):
    """encoder(img, prop) -> mu. Lets `encode_frames` /
    `evaluate_mimicgen_run` / `quick_val` / CEM / replay run UNCHANGED on
    VAE checkpoints with deterministic latents (train/eval propagates to
    the inner module). Used for the Phase 3 arm: the covariance d_eff
    estimator (Remark VI.3) then applies identically across the SIGReg and
    VAE arms, and the replay floor in `cem_planning` is not noise-inflated.

    Note: the shim exposes only mu. Aggregate-posterior diagnostics that
    need (mu, logvar) go through the RAW encoder's `.stats` (see
    `vae_latent_stats` and the closures in `dlejepa.train`)."""

    def __init__(self, vae: nn.Module):
        super().__init__()
        self.vae = vae

    def forward(self, img: torch.Tensor, prop: torch.Tensor) -> torch.Tensor:
        return self.vae.trunk(img, prop)[0]


# ============================ diagnostics ====================================

def vae_latent_stats(stats_fn, indices, device, batch: int = 256) -> dict:
    """Aggregate-posterior diagnostics for a VAE arm (paper Remark
    "Posterior Signal Fraction").

    Args:
        stats_fn: callable(index_chunk) -> (mu, logvar) torch tensors —
            a closure over the RAW encoder (not the shim) and the data
            source, e.g. `dlejepa.train._mimicgen_vae_stats_fn`.
        indices: frame indices to diagnose.
        device: torch device for the closure's forward passes.
        batch: encode batch size.

    Returns dict with:
        mu_d_eff        — covariance d_eff of the posterior mean (the
                          SAME estimator as the SIGReg arms' final
                          metrics; low = correlated collapse of the
                          SIGNAL even when sampled latents look healthy)
        mu_scale_ratio, mu_h_ratio — marginal scale / entropy ratio of mu
        signal_fraction_mean/min — per-dim Var(mu)/(Var(mu)+E[sigma^2]);
                          LOW + healthy-looking sampled latents = collapse
                          hiding under the sampling-noise floor
        posterior_std_mean, logvar_mean/std, kl_mean, noise_floor_mean,
        total_scale, N
    """
    idx = np.asarray(indices)
    mus, lvs = [], []
    with torch.no_grad():
        for i in range(0, len(idx), batch):
            c = idx[i:i + batch]
            mu, lv = stats_fn(c)
            mus.append(mu.detach().cpu())
            lvs.append(lv.detach().cpu())
    mu = torch.cat(mus).numpy().astype(np.float64)
    lv = torch.cat(lvs).numpy().astype(np.float64)

    var_mu = mu.var(axis=0)
    noise = np.exp(lv).mean(axis=0)                    # E_x[sigma^2] per dim
    sf = var_mu / np.maximum(var_mu + noise, 1e-12)
    m_mu = compute_effective_dim_from_embeddings(mu.astype(np.float32))
    return {
        "mu_d_eff": m_mu["d_eff"],
        "mu_scale_ratio": m_mu["scale_ratio"],
        "mu_h_ratio": m_mu["h_ratio"],
        "signal_fraction_mean": float(sf.mean()),
        "signal_fraction_min": float(sf.min()),
        "posterior_std_mean": float(np.exp(0.5 * lv).mean()),
        "logvar_mean": float(lv.mean()),
        "logvar_std": float(lv.std()),
        "kl_mean": float(np.mean(-0.5 * (1.0 + lv - mu ** 2 - np.exp(lv)))),
        "noise_floor_mean": float(noise.mean()),
        "total_scale": float((var_mu + noise).mean()),
        "N": int(mu.shape[0]),
    }


# ============================ factories ======================================

def build_vae_models(cfg, prop_dim: int, action_dim: int, device):
    """Phase 3 (MimicGen) VAE-arm factory: identical predictor / decoder
    to `dlejepa.train.build_phase3_models`; VAEEncoder replaces
    ViTEncoder. Returns (enc, pred, dec)."""
    enc = VAEEncoder(cfg.img_size, cfg.patch, cfg.enc_dim, cfg.enc_depth,
                     cfg.enc_heads, prop_dim, cfg.K,
                     logvar_clamp=cfg.vae_logvar_clamp).to(device)
    pred = ActionPredictor(cfg.K, action_dim, cfg.pred_hidden,
                           cfg.pred_layers).to(device)
    dec = ProprioDecoder(cfg.K, prop_dim, cfg.dec_hidden).to(device)
    return enc, pred, dec


def build_vae_models_nuscenes(cfg, device):
    """Phase 2 (nuScenes) VAE-arm factory — convenience mirror of the
    inline construction in `dlejepa.train.NuScenesVAELeJEPA`.
    Returns (enc, pred, dec)."""
    enc = VAEEncoderNuscenes(cfg.img_size, 16, 3, 192, 6, 6, cfg.K,
                             drop_rate=0.1,
                             logvar_clamp=cfg.vae_logvar_clamp).to(device)
    pred = EgoMotionPredictor(cfg.K_ego, cfg.K_static, 6, 256, 3).to(device)
    dec = PhysicsInformedDepthDecoder(cfg.K_static, 256, cfg.depth_size, 3,
                                      cfg.edge_thresh).to(device)
    return enc, pred, dec
