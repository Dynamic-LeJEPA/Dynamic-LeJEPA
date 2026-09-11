"""Per-(mode, seed) trainers for all phases.

Modes (Theorem IV.10 / Corollary IV.11 ablation):
  no_physics      : SIGReg + prediction + decoder reconstruction
  decoder_physics : + physics constraints on the DECODED output
                    (valid placement, Thm IV.10)
  encoder_physics : + the same constraints on z (or encoder features)
                    itself (invalid, Cor IV.11)

VAE arm (Ph. 2-3, paper Secs. VII-C5 / VII-D3):
  vae_no_physics / vae_decoder_physics / vae_encoder_physics :
                    the identical protocol with the closed-form marginal
                    KL(q(z|x) || N(0,I)) on a Gaussian (mu, logvar) head
                    replacing SIGReg — `NuScenesVAELeJEPA` +
                    `train_nuscenes_vae_run` (Ph 2, sampled-latent eval per
                    the published protocol) and `train_mimicgen_vae_run`
                    (Ph 3, deterministic-mu eval via VAEDeterministic).
"""
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .configs import Phase2Config, Phase3Config
from .data_mimicgen import MimicGenBundle, make_train_loader, encode_frames
from .evaluate import evaluate_mimicgen_run, evaluate_nuscenes_run
from .models import (ActionPredictor, EgoMotionPredictor, MaxEntropyEncoder,
                     PhysicsInformedDepthDecoder, ProprioDecoder, ViTEncoder,
                     encoder_physics_loss)
from .seed import seed_everything
from .sigreg import SIGReg, SIGRegPlus
from .vae import (VAEDeterministic, VAEEncoderNuscenes, build_vae_models,
                  vae_kl_loss, vae_latent_stats, vae_plus_cov_loss)

# ====================== Phase 1/2: nuScenes trainer ==========================

class NuScenesLeJEPA(nn.Module):
    """Assembled Phase 1/2 model — Theorem V.1 (C1-C4):

    C1 encoder:   z ~ N(0, sigma^2 I) via SIGReg (no other constraints)
    C2 predictor: ego-motion-decomposed dynamics (Theorem IV.7)
    C3 decoder:   physics-informed depth (Theorem IV.10, Example IV.12)
    C4 separation: independent optimizations

    ``mode`` routes the ablation (Corollary IV.11):
      no_physics      -> physics weight 0
      decoder_physics -> LiDAR + smoothness on decoded depth (VALID)
      encoder_physics -> same prior on encoder patch features (INVALID)
    """

    def __init__(self, cfg: Phase2Config):
        super().__init__()
        self.cfg = cfg
        self.encoder = MaxEntropyEncoder(cfg.img_size, 16, 3, 192, 6, 6,
                                         cfg.K, drop_rate=0.1)
        self.predictor = EgoMotionPredictor(cfg.K_ego, cfg.K_static, 6, 256, 3)
        self.decoder = PhysicsInformedDepthDecoder(cfg.K_static, 256,
                                                   cfg.depth_size, 3,
                                                   cfg.edge_thresh)
        self.sigreg = SIGReg(cfg.K, cfg.sigreg_num_projections, cfg.sigma, 1.0)
        self.pred_norm = nn.LayerNorm(cfg.K, elementwise_affine=True)

    def forward(self, img_t, img_t1, ego_motion, depth_sparse=None,
                mode="decoder_physics"):
        res = {}
        # ---- C1: encode (grab patch tokens only for the invalid ablation) ----
        if mode == "encoder_physics":
            z_t, patch_t = self.encoder(img_t, return_features=True)
            z_t1 = self.encoder(img_t1)
            res["encoder_physics_loss"] = encoder_physics_loss(patch_t)
        else:
            z_t, z_t1 = self.encoder(img_t), self.encoder(img_t1)

        scale = self.sigreg.log_scale.exp()
        z_t_s, z_t1_s = z_t * scale, z_t1 * scale
        res["z_t"], res["z_t1"] = z_t_s, z_t1_s
        sigreg_loss = self.sigreg(z_t_s) + self.sigreg(z_t1_s)
        res["sigreg_loss"] = sigreg_loss

        # ---- C2: ego-motion-decomposed prediction (Theorem IV.7) ----
        z_t_n, z_t1_n = self.pred_norm(z_t_s), self.pred_norm(z_t1_s)
        _, z_ego, z_ego_hat, z_static = self.predictor(z_t_n, ego_motion)
        z_ego_t1_n, _ = self.predictor.decompose(z_t1_n)
        pred_loss = F.mse_loss(z_ego_hat, z_ego_t1_n)
        res["pred_loss"] = pred_loss

        # ---- C3: physics-informed depth from DETACHED z_static ----
        depth_hat, physics_loss = self.decoder(z_static.detach(), img_t,
                                               depth_sparse)
        res["depth_hat"], res["physics_loss"] = depth_hat, physics_loss

        phys_w = self.cfg.physics_weight if mode == "decoder_physics" else 0.0
        total = (self.cfg.prediction_weight * pred_loss
                 + self.cfg.sigreg_weight * sigreg_loss
                 + phys_w * physics_loss)
        if mode == "encoder_physics":
            total = total + self.cfg.encoder_physics_weight \
                * res["encoder_physics_loss"]
        res["total_loss"] = total
        return res


def train_nuscenes_run(mode, seed, cfg: Phase2Config, train_loader,
                       val_loader, val_dataset, ckpt_dir=None, verbose=True):
    """One (mode, seed) run for Phases 1-2. Returns a flat metrics dict
    matching the results PKL schema (final_metrics nested)."""
    t0 = time.time()
    seed_everything(seed)
    device = cfg.device
    model = NuScenesLeJEPA(cfg).to(device)
    opt = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": cfg.lr_encoder},
        {"params": model.predictor.parameters(), "lr": cfg.lr_predictor},
        {"params": model.decoder.parameters(), "lr": cfg.lr_decoder},
        {"params": model.sigreg.parameters(), "lr": cfg.lr_encoder},
    ], weight_decay=cfg.weight_decay)

    history = []
    for ep in range(cfg.num_epochs):
        model.train()
        sums = defaultdict(float)
        nb = 0
        for b in train_loader:
            res = model(b["img_t"].to(device), b["img_t1"].to(device),
                        b["ego_motion"].to(device),
                        b["depth_sparse"].to(device), mode=mode)
            opt.zero_grad()
            res["total_loss"].backward()
            opt.step()
            for k in ("total_loss", "pred_loss", "sigreg_loss", "physics_loss"):
                sums[k] += float(res[k])
            if "encoder_physics_loss" in res:
                sums["encoder_physics"] += float(res["encoder_physics_loss"])
            nb += 1
        model.eval()
        with torch.no_grad():
            vsum, vnb = 0.0, 0
            for b in val_loader:
                res = model(b["img_t"].to(device), b["img_t1"].to(device),
                            b["ego_motion"].to(device),
                            b["depth_sparse"].to(device), mode=mode)
                vsum += float(res["total_loss"])
                vnb += 1
        h = {"epoch": ep, "mode": mode,
             **{k: v / max(nb, 1) for k, v in sums.items()},
             "val_loss": vsum / max(vnb, 1)}
        history.append(h)
        if verbose:
            print(f"  ep {ep + 1:2d}/{cfg.num_epochs} | "
                  f"loss {h['total_loss']:.4f} | pred {h['pred_loss']:.4f} | "
                  f"sig {h['sigreg_loss']:.4f} | phys {h['physics_loss']:.4f} | "
                  f"val {h['val_loss']:.4f}")

    metrics = evaluate_nuscenes_run(model, val_dataset, cfg)
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)
        ck = os.path.join(ckpt_dir, f"ckpt_{mode}_seed{seed}.pt")
        torch.save({"encoder": model.encoder.state_dict(),
                    "predictor": model.predictor.state_dict(),
                    "decoder": model.decoder.state_dict(),
                    "sigreg": model.sigreg.state_dict(),
                    "mode": mode, "seed": seed}, ck)
        metrics["ckpt"] = ck
    metrics.update({"seed": seed, "mode": mode,
                    "elapsed_minutes": (time.time() - t0) / 60,
                    "history": history})
    return metrics


# ====================== Phase 3: MimicGen trainer ============================

def build_phase3_models(bundle: MimicGenBundle, cfg: Phase3Config, device):
    enc = ViTEncoder(cfg.img_size, cfg.patch, cfg.enc_dim, cfg.enc_depth,
                     cfg.enc_heads, bundle.PROP_DIM, cfg.K).to(device)
    pred = ActionPredictor(cfg.K, bundle.ACTION_DIM, cfg.pred_hidden,
                           cfg.pred_layers).to(device)
    dec = ProprioDecoder(cfg.K, bundle.PROP_DIM, cfg.dec_hidden).to(device)
    return enc, pred, dec


@torch.no_grad()
def quick_val(encoder, predictor, bundle, cfg, device, n=2048):
    encoder.eval()
    predictor.eval()
    R = np.random.RandomState(999)
    sel = R.choice(len(bundle.VAL_IT), min(n, len(bundle.VAL_IT)), replace=False)
    it, inn = bundle.VAL_IT[sel], bundle.VAL_INN[sel]
    need = np.unique(np.concatenate([it, inn]))
    Zm = encode_frames(encoder, bundle, need, device)
    pos = {int(f): i for i, f in enumerate(need)}
    z_t = torch.from_numpy(Zm[[pos[int(i)] for i in it]]).float().to(device)
    z_nx = Zm[[pos[int(i)] for i in inn]]
    zp = predictor(z_t, torch.from_numpy(bundle.ACT[it]).float().to(device))
    zp = zp.cpu().numpy()
    encoder.train()
    predictor.train()
    return float(((zp - z_nx) ** 2).mean())


def train_mimicgen_run(mode, seed, bundle: MimicGenBundle, cfg: Phase3Config,
                       epochs=None, steps=None, ckpt_dir=None,
                       verbose=True) -> dict:
    t0 = time.time()
    device = cfg.device
    epochs = epochs or cfg.num_epochs
    steps = steps or cfg.steps_per_epoch
    seed_everything(seed)

    enc, pred, dec = build_phase3_models(bundle, cfg, device)
    sigreg = SIGRegPlus(cfg.K, cfg.sigreg_num_projections, cfg.sigma, 1.0).to(device)
    M_phys = (nn.Linear(bundle.ACTION_DIM, cfg.K, bias=False).to(device)
              if mode == "encoder_physics" else None)
    W_T = torch.from_numpy(bundle.W_CAL).float().to(device)
    EEF_T = torch.tensor(bundle.EEF_IDX, dtype=torch.long, device=device)

    phys_w = (cfg.physics_weight if mode == "decoder_physics"
              else cfg.encoder_physics_weight if mode == "encoder_physics"
              else 0.0)
    params = [{"params": enc.parameters(), "lr": cfg.lr_encoder},
              {"params": pred.parameters(), "lr": cfg.lr_predictor},
              {"params": dec.parameters(), "lr": cfg.lr_decoder}]
    if M_phys is not None:
        params.append({"params": M_phys.parameters(), "lr": cfg.lr_predictor})
    opt = torch.optim.AdamW(params, weight_decay=cfg.weight_decay)

    loader = make_train_loader(bundle, cfg, seed)
    history = []
    enc.train()
    pred.train()
    dec.train()
    for ep in range(epochs):
        sums = {"loss": 0, "pred": 0, "sig": 0, "rec": 0, "phys": 0}
        nb = 0
        for step, b in enumerate(loader):
            if step >= steps:
                break
            bi = b["img_t"].to(device)   # dataset already returns (B,3,84,84)
            bn = b["img_next"].to(device)
            bp = b["img_prev"].to(device)
            pt = b["prop_t"].to(device)
            pn = b["prop_next"].to(device)
            pp = b["prop_prev"].to(device)
            ba = b["action"].to(device)
            z_t, z_next, z_prev = enc(bi, pt), enc(bn, pn), enc(bp, pp)

            l_pred = F.mse_loss(pred(z_t, ba), z_next)            # Thm IV.4b
            l_sig = sigreg(torch.cat([z_prev, z_t, z_next], 0))   # Thm IV.4a
            q_t, q_next, q_prev = dec(z_t), dec(z_next), dec(z_prev)
            l_rec = (F.mse_loss(q_t, pt) + F.mse_loss(q_next, pn)
                     + F.mse_loss(q_prev, pp)) / 3

            l_phys = torch.tensor(0.0, device=device)
            if mode == "decoder_physics":        # VALID placement
                kin = (q_next - q_t)[:, EEF_T] - ba @ W_T
                l_phys = kin.pow(2).mean() \
                    + (q_next - 2 * q_t + q_prev).pow(2).mean()
            elif mode == "encoder_physics":      # INVALID placement
                kin = (z_next - z_t) - M_phys(ba)
                l_phys = kin.pow(2).mean() \
                    + (z_next - 2 * z_t + z_prev).pow(2).mean()

            loss = (cfg.prediction_weight * l_pred
                    + cfg.sigreg_weight * l_sig
                    + cfg.recon_weight * l_rec
                    + phys_w * l_phys)
            opt.zero_grad()
            loss.backward()
            opt.step()
            for k, v in [("loss", loss.item()), ("pred", l_pred.item()),
                         ("sig", l_sig.item()), ("rec", l_rec.item()),
                         ("phys", float(l_phys))]:
                sums[k] += v
            nb += 1

        h = {"epoch": ep, **{k: sums[k] / max(nb, 1) for k in sums}}
        if ep % 5 == 0 or ep == epochs - 1:
            h["val_pred_mse"] = quick_val(enc, pred, bundle, cfg, device)
            d = sigreg.get_diagnostics()
            h["sig_d_eff"], h["sig_scale"] = d["d_eff"], d["scale_ratio"]
        history.append(h)
        if verbose:
            print(f"  ep {ep + 1:2d}/{epochs} | loss {sums['loss'] / nb:8.4f} | "
                  f"pred {sums['pred'] / nb:.4f} | sig {sums['sig'] / nb:.4f} | "
                  f"rec {sums['rec'] / nb:.4f} | phys {sums['phys'] / nb:.4f}"
                  + (f" | val_pred {h['val_pred_mse']:.4f}"
                     if "val_pred_mse" in h else ""))

    enc.eval()
    pred.eval()
    dec.eval()
    metrics = evaluate_mimicgen_run(enc, pred, bundle, cfg, seed)
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)
        ck = os.path.join(ckpt_dir, f"ckpt_{mode}_seed{seed}.pt")
        torch.save({"encoder": enc.state_dict(),
                    "predictor": pred.state_dict(),
                    "decoder": dec.state_dict(),
                    "mode": mode, "seed": seed}, ck)
        metrics["ckpt"] = ck
    metrics.update({"seed": seed, "mode": mode,
                    "elapsed_minutes": (time.time() - t0) / 60,
                    "history": history, "kinematics_r2": bundle.KIN_R2})
    return metrics


# ============================================================================
# VAE arm trainers (Phases 2-3) — encoder-mechanism comparison
# ============================================================================
# Paper Secs. VII-C5 (nuScenes) / VII-D3 (MimicGen). Protocol mirrors the
# SIGReg arms EXACTLY (epochs, steps, splits, seeds, physics weights, mode
# routing) — the ONLY variable is how the encoder reaches its target
# distribution: the closed-form marginal KL(q(z|x) || N(0,I)) on a Gaussian
# (mu, logvar) head replaces SIGReg / SIGReg+.
#
# Refactor conventions (documented, deliberate):
#   * The marginal distribution-matching term is pooled/symmetrized over
#     all encoded frames. The notebooks applied it to a subset of frames
#     (Phase 2: frame t only); the refactored SIGReg trainers likewise
#     symmetrize SIGReg over both frames, and the arms must be
#     term-structure-identical for the controlled comparison. The
#     marginal-only CHARACTER — the hypothesis under test — is unchanged.
#   * The invalid (Cor IV.11) placement applies the SAME smoothness prior
#     to the encoder's own deterministic trunk features
#     (`encoder_physics_loss`), matching the SIGReg arms' convention in
#     this package (the notebooks used a linear depth-projection head).
#   * Phase 2 evaluates on SAMPLED latents (published protocol,
#     cfg.vae_eval_deterministic=False); Phase 3 evaluates and plans on
#     deterministic posterior means via VAEDeterministic — the protocol
#     upgrade introduced with the calibrated Phase 3 arm
#     (configs.vae_eval_deterministic=True).
#   * z_static is detached before the decoder (mirrors `NuScenesLeJEPA`;
#     the notebook did not detach — noted for provenance).
# The committed notebooks remain the exact provenance of the published
# numbers; this package is the clean reimplementation.
# ============================================================================


def _vae_placement(mode: str) -> str:
    """Map a VAE-arm mode to its physics placement (the 'vae'/'vaeplus'
    prefix is the only difference from the SIGReg arm's mode names)."""
    if mode.endswith("decoder_physics"):
        return "decoder_physics"
    if mode.endswith("encoder_physics"):
        return "encoder_physics"
    return "no_physics"


def _to_img_tensor(arr: np.ndarray) -> torch.Tensor:
    """uint8 HWC numpy -> normalized float CHW tensor (kept local so the
    trainer does not depend on data_mimicgen internals; identical to the
    notebook helper)."""
    x = torch.from_numpy(np.ascontiguousarray(arr)).permute(0, 3, 1, 2).float()
    return x.div_(255.).sub_(0.5).div_(0.5)


def _mimicgen_vae_stats_fn(vae_enc, bundle: MimicGenBundle, device):
    """Closure feeding (mu, logvar) chunks to `vae_latent_stats` from the
    bundle's frame memmap + normalized proprioception (RAW encoder — the
    shim exposes only mu, and the diagnostic needs both heads)."""

    @torch.no_grad()
    def fn(c):
        x = _to_img_tensor(bundle.IMG[c]).to(device)
        p = torch.from_numpy(bundle.PROP_N[c]).float().to(device)
        return vae_enc.stats(x, p)

    return fn


def build_phase3_vae_models(bundle: MimicGenBundle, cfg: Phase3Config,
                            device):
    """VAE-arm counterpart of `build_phase3_models`: identical predictor /
    decoder, VAEEncoder replaces ViTEncoder (paper Sec. VII-D3)."""
    return build_vae_models(cfg, bundle.PROP_DIM, bundle.ACTION_DIM, device)


def train_mimicgen_vae_run(mode, seed, bundle: MimicGenBundle,
                           cfg: Phase3Config, epochs=None, steps=None,
                           ckpt_dir=None, verbose=True) -> dict:
    """One (mode, seed) run of the Phase 3 VAE arm. Mirrors
    `train_mimicgen_run` in protocol; only the encoder mechanism and its
    distribution term differ. Returns the same metrics schema
    (final_metrics via evaluate_mimicgen_run on DETERMINISTIC latents)
    plus a `vae_stats` block (signal fraction, posterior std, KL,
    mu_d_eff — paper Remark 'Posterior Signal Fraction')."""
    allowed = list(cfg.vae_modes)
    if cfg.vae_plus_cov_weight > 0:            # optional VAE+ stretch modes
        allowed += ["vaeplus_decoder_physics", "vaeplus_encoder_physics"]
    assert mode in allowed, f"unknown VAE mode: {mode}"
    if not getattr(cfg, "vae_kl_calibrated", True):
        print("[WARN] running the VAE arm WITHOUT the pre-run KL-weight "
              "calibration sweep — confirm cfg.vae_kl_weight is intentional "
              "(experiments/phase3_mimicgen/calibrate_vae_kl.py).")
    t0 = time.time()
    device = cfg.device
    place = _vae_placement(mode)
    epochs = epochs or cfg.num_epochs
    steps = steps or cfg.steps_per_epoch
    seed_everything(seed)

    enc, pred, dec = build_phase3_vae_models(bundle, cfg, device)
    M_phys = (nn.Linear(bundle.ACTION_DIM, cfg.K, bias=False).to(device)
              if place == "encoder_physics" else None)
    W_T = torch.from_numpy(bundle.W_CAL).float().to(device)
    EEF_T = torch.tensor(bundle.EEF_IDX, dtype=torch.long, device=device)

    phys_w = (cfg.physics_weight if place == "decoder_physics"
              else cfg.encoder_physics_weight if place == "encoder_physics"
              else 0.0)
    params = [{"params": enc.parameters(), "lr": cfg.lr_encoder},
              {"params": pred.parameters(), "lr": cfg.lr_predictor},
              {"params": dec.parameters(), "lr": cfg.lr_decoder}]
    if M_phys is not None:
        params.append({"params": M_phys.parameters(), "lr": cfg.lr_predictor})
    opt = torch.optim.AdamW(params, weight_decay=cfg.weight_decay)

    stats_fn = _mimicgen_vae_stats_fn(enc, bundle, device)
    loader = make_train_loader(bundle, cfg, seed)
    history = []
    enc.train()
    pred.train()
    dec.train()
    for ep in range(epochs):
        sums = {"loss": 0, "pred": 0, "kl": 0, "rec": 0, "phys": 0}
        nb = 0
        for step, b in enumerate(loader):
            if step >= steps:
                break
            bi = b["img_t"].to(device)   # dataset already returns (B,3,84,84)
            bn = b["img_next"].to(device)
            bp = b["img_prev"].to(device)
            pt = b["prop_t"].to(device)
            pn = b["prop_next"].to(device)
            pp = b["prop_prev"].to(device)
            ba = b["action"].to(device)

            mu_t, lv_t = enc.stats(bi, pt)
            mu_n, lv_n = enc.stats(bn, pn)
            mu_p, lv_p = enc.stats(bp, pp)
            # reparameterized sampling for the prediction / physics paths
            z_t = mu_t + torch.exp(0.5 * lv_t) * torch.randn_like(mu_t)
            z_next = mu_n + torch.exp(0.5 * lv_n) * torch.randn_like(mu_n)
            z_prev = mu_p + torch.exp(0.5 * lv_p) * torch.randn_like(mu_p)

            # distribution-matching slot: marginal-only KL on the posterior
            # heads, pooled over (prev, t, next) — the SIGReg slot's exact
            # counterpart (Thm IV.4a slot)
            l_kl = vae_kl_loss(torch.cat([mu_p, mu_t, mu_n], 0),
                               torch.cat([lv_p, lv_t, lv_n], 0))
            l_pred = F.mse_loss(pred(z_t, ba), z_next)            # Thm IV.4b
            q_t, q_next, q_prev = dec(z_t), dec(z_next), dec(z_prev)
            l_rec = (F.mse_loss(q_t, pt) + F.mse_loss(q_next, pn)
                     + F.mse_loss(q_prev, pp)) / 3

            l_phys = torch.tensor(0.0, device=device)
            if place == "decoder_physics":       # VALID placement (Thm IV.10)
                kin = (q_next - q_t)[:, EEF_T] - ba @ W_T
                l_phys = kin.pow(2).mean() \
                    + (q_next - 2 * q_t + q_prev).pow(2).mean()
            elif place == "encoder_physics":     # INVALID placement (Cor IV.11)
                kin = (z_next - z_t) - M_phys(ba)
                l_phys = kin.pow(2).mean() \
                    + (z_next - 2 * z_t + z_prev).pow(2).mean()

            loss = (cfg.prediction_weight * l_pred
                    + cfg.vae_kl_weight * l_kl
                    + cfg.recon_weight * l_rec
                    + phys_w * l_phys)
            if cfg.vae_plus_cov_weight > 0:      # optional VAE+ stretch
                loss = loss + cfg.vae_plus_cov_weight * vae_plus_cov_loss(
                    torch.cat([z_prev, z_t, z_next], 0))

            opt.zero_grad()
            loss.backward()
            opt.step()
            for k, v in [("loss", loss.item()), ("pred", l_pred.item()),
                         ("kl", l_kl.item()), ("rec", l_rec.item()),
                         ("phys", float(l_phys))]:
                sums[k] += v
            nb += 1

        h = {"epoch": ep, **{k: sums[k] / max(nb, 1) for k in sums}}
        if ep % 5 == 0 or ep == epochs - 1:
            # deterministic-mu validation: the shim makes quick_val run
            # unchanged and keeps the replay floor noise-free later
            h["val_pred_mse"] = quick_val(VAEDeterministic(enc), pred,
                                          bundle, cfg, device)
            # aggregate-posterior diagnostics on a fixed 3k-frame sample
            R_h = np.random.RandomState(2000 + seed)
            n_h = min(3000, len(bundle.VAL_FRAMES))
            hst = vae_latent_stats(
                stats_fn, R_h.choice(bundle.VAL_FRAMES, n_h, replace=False),
                device)
            h["mu_d_eff"] = hst["mu_d_eff"]
            h["signal_fraction"] = hst["signal_fraction_mean"]
        history.append(h)
        if verbose:
            print(f"  ep {ep + 1:2d}/{epochs} | loss {sums['loss'] / nb:8.4f} | "
                  f"pred {sums['pred'] / nb:.4f} | kl {sums['kl'] / nb:.4f} | "
                  f"rec {sums['rec'] / nb:.4f} | phys {sums['phys'] / nb:.4f}"
                  + (f" | val_pred {h['val_pred_mse']:.4f}"
                     if "val_pred_mse" in h else ""))

    enc.eval()
    pred.eval()
    dec.eval()
    # evaluation through the deterministic-mu shim: evaluate_mimicgen_run
    # (and the CEM / replay paths in cem_planning) run UNCHANGED, and the
    # covariance d_eff estimator (Remark VI.3) applies identically across
    # the SIGReg and VAE arms
    shim = VAEDeterministic(enc)
    metrics = evaluate_mimicgen_run(shim, pred, bundle, cfg, seed)
    # same first frame draw evaluate_mimicgen_run uses (RandomState 2000+seed)
    R_ev = np.random.RandomState(2000 + seed)
    n_ev = min(cfg.eval_frames, len(bundle.VAL_FRAMES))
    metrics["vae_stats"] = vae_latent_stats(
        stats_fn, R_ev.choice(bundle.VAL_FRAMES, n_ev, replace=False), device)
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)
        ck = os.path.join(ckpt_dir, f"ckpt_{mode}_seed{seed}.pt")
        torch.save({"encoder": enc.state_dict(),
                    "predictor": pred.state_dict(),
                    "decoder": dec.state_dict(),
                    "mode": mode, "seed": seed,
                    "encoder_class": "VAEEncoder",
                    "vae_kl_weight": cfg.vae_kl_weight}, ck)
        metrics["ckpt"] = ck
    metrics.update({"seed": seed, "mode": mode,
                    "elapsed_minutes": (time.time() - t0) / 60,
                    "history": history, "kinematics_r2": bundle.KIN_R2})
    return metrics


class NuScenesVAELeJEPA(nn.Module):
    """VAE-arm counterpart of `NuScenesLeJEPA` (paper Sec. VII-C5).

    Identical trunk (PatchEmbed / TransformerBlock stack), predictor and
    decoder; the distribution-matching term is the closed-form marginal
    KL(q(z|x) || N(0,I)) on a (mu, logvar) head, replacing SIGReg. The
    forward returns the SAME result-dict schema as `NuScenesLeJEPA`
    (z_t / z_t1 / pred_loss / physics_loss / total_loss, plus kl_loss and
    encoder_physics_loss where applicable), so `evaluate_nuscenes_run`
    runs unchanged on this wrapper.

    Latents are SAMPLED (reparameterization) in both training and
    evaluation by default — the published Phase 2 protocol
    (cfg.vae_eval_deterministic=False); set the flag True for the
    Phase-3-style deterministic-mu protocol.
    """

    def __init__(self, cfg: Phase2Config):
        super().__init__()
        self.cfg = cfg
        self.encoder = VAEEncoderNuscenes(cfg.img_size, 16, 3, 192, 6, 6,
                                          cfg.K, drop_rate=0.1,
                                          logvar_clamp=cfg.vae_logvar_clamp)
        self.predictor = EgoMotionPredictor(cfg.K_ego, cfg.K_static, 6, 256, 3)
        self.decoder = PhysicsInformedDepthDecoder(cfg.K_static, 256,
                                                   cfg.depth_size, 3,
                                                   cfg.edge_thresh)
        self.deterministic = bool(getattr(cfg, "vae_eval_deterministic", False))

    def forward(self, img_t, img_t1, ego_motion, depth_sparse=None,
                mode="vae_decoder_physics"):
        res = {}
        # ---- encode (grab trunk patch tokens only for the invalid ablation) --
        if mode == "vae_encoder_physics":
            mu_t, lv_t, patch_t = self.encoder(img_t, return_features=True)
            mu_t1, lv_t1 = self.encoder(img_t1)
            res["encoder_physics_loss"] = encoder_physics_loss(patch_t)
        else:
            mu_t, lv_t = self.encoder(img_t)
            mu_t1, lv_t1 = self.encoder(img_t1)

        if self.deterministic:
            z_t, z_t1 = mu_t, mu_t1
        else:
            z_t = mu_t + torch.exp(0.5 * lv_t) * torch.randn_like(mu_t)
            z_t1 = mu_t1 + torch.exp(0.5 * lv_t1) * torch.randn_like(mu_t1)
        res["z_t"], res["z_t1"] = z_t, z_t1

        # marginal-only KL, symmetrized over both frames (see header note)
        res["kl_loss"] = vae_kl_loss(mu_t, lv_t) + vae_kl_loss(mu_t1, lv_t1)

        # ---- C2: prediction on the sampled latents (no pred_norm /
        # log_scale — those exist to interface with SIGReg, absent here) ----
        _, z_ego, z_ego_hat, z_static = self.predictor(z_t, ego_motion)
        z_ego_t1, _ = self.predictor.decompose(z_t1)
        pred_loss = F.mse_loss(z_ego_hat, z_ego_t1)
        res["pred_loss"] = pred_loss

        # ---- C3: physics-informed depth from DETACHED z_static ----
        depth_hat, physics_loss = self.decoder(z_static.detach(), img_t,
                                               depth_sparse)
        res["depth_hat"], res["physics_loss"] = depth_hat, physics_loss

        phys_w = (self.cfg.physics_weight
                  if mode == "vae_decoder_physics" else 0.0)
        total = (self.cfg.prediction_weight * pred_loss
                 + self.cfg.vae_kl_weight * res["kl_loss"]
                 + phys_w * physics_loss)
        if mode == "vae_encoder_physics":
            total = total + self.cfg.encoder_physics_weight \
                * res["encoder_physics_loss"]
        res["total_loss"] = total
        return res


def train_nuscenes_vae_run(mode, seed, cfg: Phase2Config, train_loader,
                           val_loader, val_dataset, ckpt_dir=None,
                           verbose=True):
    """One (mode, seed) run of the Phase 2 VAE arm. Mirrors
    `train_nuscenes_run` exactly (epochs, optimizer groups minus the
    SIGReg module, logging); returns the same flat metrics schema so the
    Phase 2 results stay comparable across arms.

    NOTE (runtime): this arm runs in the LEGACY WEAK-PRIOR regime
    (cfg.vae_kl_weight = 0.01, effectively dormant) — only the within-arm
    placement ORDERING is interpreted from it (paper Sec. VII-C5 caveats);
    the calibrated encoder-mechanism test is the Phase 3 arm."""
    assert mode in cfg.vae_modes, f"unknown VAE mode: {mode}"
    t0 = time.time()
    seed_everything(seed)
    device = cfg.device
    model = NuScenesVAELeJEPA(cfg).to(device)
    opt = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": cfg.lr_encoder},
        {"params": model.predictor.parameters(), "lr": cfg.lr_predictor},
        {"params": model.decoder.parameters(), "lr": cfg.lr_decoder},
    ], weight_decay=cfg.weight_decay)

    history = []
    for ep in range(cfg.num_epochs):
        model.train()
        sums = defaultdict(float)
        nb = 0
        for b in train_loader:
            res = model(b["img_t"].to(device), b["img_t1"].to(device),
                        b["ego_motion"].to(device),
                        b["depth_sparse"].to(device), mode=mode)
            opt.zero_grad()
            res["total_loss"].backward()
            opt.step()
            for k in ("total_loss", "pred_loss", "kl_loss", "physics_loss"):
                sums[k] += float(res[k])
            if "encoder_physics_loss" in res:
                sums["encoder_physics"] += float(res["encoder_physics_loss"])
            nb += 1
        model.eval()
        with torch.no_grad():
            vsum, vnb = 0.0, 0
            for b in val_loader:
                res = model(b["img_t"].to(device), b["img_t1"].to(device),
                            b["ego_motion"].to(device),
                            b["depth_sparse"].to(device), mode=mode)
                vsum += float(res["total_loss"])
                vnb += 1
        h = {"epoch": ep, "mode": mode,
             **{k: v / max(nb, 1) for k, v in sums.items()},
             "val_loss": vsum / max(vnb, 1)}
        history.append(h)
        if verbose:
            print(f"  ep {ep + 1:2d}/{cfg.num_epochs} | "
                  f"loss {h['total_loss']:.4f} | pred {h['pred_loss']:.4f} | "
                  f"kl {h['kl_loss']:.4f} | phys {h['physics_loss']:.4f} | "
                  f"val {h['val_loss']:.4f}")

    metrics = evaluate_nuscenes_run(model, val_dataset, cfg)
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)
        ck = os.path.join(ckpt_dir, f"ckpt_{mode}_seed{seed}.pt")
        torch.save({"encoder": model.encoder.state_dict(),
                    "predictor": model.predictor.state_dict(),
                    "decoder": model.decoder.state_dict(),
                    "mode": mode, "seed": seed,
                    "encoder_class": "VAEEncoderNuscenes",
                    "vae_kl_weight": cfg.vae_kl_weight}, ck)
        metrics["ckpt"] = ck
    metrics.update({"seed": seed, "mode": mode,
                    "elapsed_minutes": (time.time() - t0) / 60,
                    "history": history})
    return metrics
