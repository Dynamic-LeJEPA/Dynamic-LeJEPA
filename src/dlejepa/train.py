"""Per-(mode, seed) trainers for all phases.

Modes (Theorem IV.10 / Corollary IV.11 ablation):
  no_physics      : SIGReg + prediction + decoder reconstruction
  decoder_physics : + physics constraints on the DECODED output
                    (valid placement, Thm IV.10)
  encoder_physics : + the same constraints on z (or encoder features)
                    itself (invalid, Cor IV.11)
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
