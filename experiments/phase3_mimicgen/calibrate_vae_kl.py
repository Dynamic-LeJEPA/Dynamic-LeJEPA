#!/usr/bin/env python
"""VAE_KL_WEIGHT calibration sweep for the Phase 3 arm (paper Sec. VII-D3).

WHY THIS SCRIPT EXISTS (pre-registration evidence): the published Phase 3
VAE arm operates at KL weight 5.0, "selected from a pre-run calibration
sweep". This script IS that sweep, committed so the claim is verifiable:
run it before any VAE ablation and read the trade-off table it prints.

The problem it solves (Phase 2 lesson, paper Sec. VII-C5 caveat): at KL
weight 0.01 the KL term contributed ~0.003 of a ~136 total loss —
effectively DORMANT, so the encoder never felt its distribution-matching
term and the marginals were never enforced (H_ratio 0.26). The
distribution-matching slot here belongs to SIGReg+ at SIGREG_WEIGHT = 20
on a raw loss ~1. The KL weight must put the weighted KL term in a
comparable regime — WITHOUT swamping the physics term (intentionally
strong at 10 for the ablation contrast).

Sweep protocol (mirrors the notebook calibration): for each candidate
weight, a fresh model is trained for CALIB_STEPS steps on the fixed seed-0
loader; RAW (unweighted) loss magnitudes are tracked, and the weighted
contribution of each term is printed so the trade-off is explicit.

READ-BEFORE-PICKING RULES (printed at the end):
  - kl * w should be COMPARABLE to pred + rec (the encoder must actually
    feel its distribution term), and must NOT swamp phys * 10.
  - Reference: the SIGReg slot contributes SIGREG_WEIGHT * ~1 = 20.
PUBLISHED OPERATING POINT: KL weight 5.0 -> kl*w ~ 2.1 vs pred+rec ~ 2.8
and phys*10 ~ 7.6 (see the sweep table) — comparable to pred+rec, well
below the physics term.

Runtime: ~2 min on a T4 (5 candidates x 20 steps). Changing the published
weight invalidates the paper's numbers — re-run the sweep AND the ablation
first (`run_ablation_vae.py` asserts the operating point).
"""
import argparse
import json
import os
from collections import defaultdict

import torch

from dlejepa.configs import Phase3Config
from dlejepa.data_mimicgen import load_mimicgen, make_train_loader
from dlejepa.seed import seed_everything
from dlejepa.train import build_phase3_vae_models
from dlejepa.vae import vae_kl_loss

import torch.nn.functional as F
import torch.nn as nn

CANDIDATES = [0.5, 1.0, 2.0, 5.0, 10.0]
CALIB_STEPS = 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--cache", default="cache/phase3")
    ap.add_argument("--steps", type=int, default=CALIB_STEPS)
    ap.add_argument("--out", default="results/phase3/vae_kl_calibration.json")
    args = ap.parse_args()

    cfg = Phase3Config()
    bundle = load_mimicgen(args.hdf5, cfg, args.cache)
    device = cfg.device

    # physics operating-point tensors (decoder-physics placement is the
    # calibration reference, mirroring the notebook)
    W_T = torch.from_numpy(bundle.W_CAL).float().to(device)
    EEF_T = torch.tensor(bundle.EEF_IDX, dtype=torch.long, device=device)

    print("=" * 72)
    print(f"VAE_KL_WEIGHT CALIBRATION — Phase 3 loss scales "
          f"({len(CANDIDATES)} candidates x {args.steps} steps)")
    print(f"Reference slots: SIGReg arm contributes "
          f"SIGREG_WEIGHT * ~1 = {cfg.sigreg_weight:.0f}; "
          f"physics weight = {cfg.physics_weight:.0f}")
    print("=" * 72)

    sweep = []
    for kw in CANDIDATES:
        seed_everything(0)                       # fixed seed -> comparable
        enc, pred, dec = build_phase3_vae_models(bundle, cfg, device)
        opt = torch.optim.AdamW(
            list(enc.parameters()) + list(pred.parameters()) +
            list(dec.parameters()),
            lr=cfg.lr_encoder, weight_decay=cfg.weight_decay)
        loader = make_train_loader(bundle, cfg, 0)
        it = iter(loader)
        raw = defaultdict(list)
        for _ in range(args.steps):
            try:
                b = next(it)
            except StopIteration:
                it = iter(loader)
                b = next(it)
            bi, bn, bp = (b["img_t"].to(device), b["img_next"].to(device),
                          b["img_prev"].to(device))
            pt, pn, pp = (b["prop_t"].to(device), b["prop_next"].to(device),
                          b["prop_prev"].to(device))
            ba = b["action"].to(device)

            mu_t, lv_t = enc.stats(bi, pt)
            mu_n, lv_n = enc.stats(bn, pn)
            mu_p, lv_p = enc.stats(bp, pp)
            z_t = mu_t + torch.exp(0.5 * lv_t) * torch.randn_like(mu_t)
            z_n = mu_n + torch.exp(0.5 * lv_n) * torch.randn_like(mu_n)
            z_p = mu_p + torch.exp(0.5 * lv_p) * torch.randn_like(mu_p)

            l_kl = vae_kl_loss(torch.cat([mu_p, mu_t, mu_n], 0),
                               torch.cat([lv_p, lv_t, lv_n], 0))
            l_pred = F.mse_loss(pred(z_t, ba), z_n)
            q_t, q_n, q_p = dec(z_t), dec(z_n), dec(z_p)
            l_rec = (F.mse_loss(q_t, pt) + F.mse_loss(q_n, pn)
                     + F.mse_loss(q_p, pp)) / 3
            kin = (q_n - q_t)[:, EEF_T] - ba @ W_T     # decoder-physics
            l_phys = (kin.pow(2).mean()
                      + (q_n - 2 * q_t + q_p).pow(2).mean())  # operating pt

            loss = (cfg.prediction_weight * l_pred + kw * l_kl
                    + cfg.recon_weight * l_rec
                    + cfg.physics_weight * l_phys)
            opt.zero_grad()
            loss.backward()
            opt.step()
            for k, v in [("kl", l_kl.item()), ("pred", l_pred.item()),
                         ("rec", l_rec.item()), ("phys", l_phys.item()),
                         ("total", loss.item())]:
                raw[k].append(v)

        row = {"kl_weight": kw,
               "raw_kl": float(torch.tensor(raw["kl"]).mean()),
               "raw_pred": float(torch.tensor(raw["pred"]).mean()),
               "raw_rec": float(torch.tensor(raw["rec"]).mean()),
               "raw_phys": float(torch.tensor(raw["phys"]).mean()),
               "raw_total": float(torch.tensor(raw["total"]).mean())}
        row["weighted_kl"] = kw * row["raw_kl"]
        row["weighted_pred_rec"] = row["raw_pred"] + row["raw_rec"]
        row["weighted_phys"] = cfg.physics_weight * row["raw_phys"]
        sweep.append(row)

        print(f"\nKL weight = {kw}:")
        print(f"  raw   kl={row['raw_kl']:.4f}  pred={row['raw_pred']:.4f}"
              f"  rec={row['raw_rec']:.4f}  phys={row['raw_phys']:.4f}")
        print(f"  weighted kl*w={row['weighted_kl']:.3f} | "
              f"pred+rec={row['weighted_pred_rec']:.3f} | "
              f"phys*{cfg.physics_weight:.0f}={row['weighted_phys']:.3f} | "
              f"total={row['raw_total']:.3f}")

        del enc, pred, dec, opt
        torch.cuda.empty_cache()

    # ---- read-before-picking summary --------------------------------------
    print("\n" + "=" * 72)
    print("READ BEFORE PICKING:")
    print("  - kl*w should be COMPARABLE to pred+rec (the encoder must")
    print("    actually feel its distribution term) and NOT swamp")
    print(f"    phys*{cfg.physics_weight:.0f} (physics is intentionally strong")
    print("    for the ablation contrast).")
    print(f"  - Reference: the SIGReg slot contributes "
          f"SIGREG_WEIGHT*~1 = {cfg.sigreg_weight:.0f}.")
    print("=" * 72)
    print(f"\nPublished operating point: cfg.vae_kl_weight = 5.0 "
          f"(kl*w ~ 2.1 vs pred+rec ~ 2.8, phys*10 ~ 7.6).")
    print(f"Confirm/set it in dlejepa/configs.py (Phase3Config."
          f"vae_kl_weight) before running run_ablation_vae.py.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"candidates": CANDIDATES, "steps": args.steps,
                   "sigreg_weight_reference": cfg.sigreg_weight,
                   "published_operating_point": 5.0, "sweep": sweep}, f,
                  indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
