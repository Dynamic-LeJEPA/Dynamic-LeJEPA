#!/usr/bin/env python
"""Latent CEM planning + model-exploitation diagnostic (paper Sec VII-D8).

Loads frozen Phase 3 checkpoints, runs the planner + replay/random baselines,
and reports the action-space exploitation diagnostic per model.

Arms (paper Secs. VII-D3 / VII-D8):
  --arm sigreg  : `no_physics` / `decoder_physics` / `encoder_physics`
                  (deterministic ViT encoder, as-is)
  --arm vae     : `vae_no_physics` / `vae_decoder_physics` /
                  `vae_encoder_physics` (VAEEncoder wrapped in the
                  VAEDeterministic shim — DETERMINISTIC posterior-mean
                  latents for CEM scoring AND baselines, so the replay
                  floor is not noise-inflated and the protocol is identical
                  across arms)

Matched anchors (paper protocol): `CEMConfig(seed=model_seed)` makes the
anchor set a pure function of the seed, so the SIGReg and VAE arms share
anchor sets for the same model seed {0, 1, 2}. The old script's
`3100 + seed` offsets broke that matching; the published protocol keys
anchors to the MODEL seed.

Checkpoint compatibility: released VAE checkpoints carry
`encoder_class: "VAEEncoder"`; the loader dispatches on that field (with a
`vae_checkpoint_kl_weight` sanity check against `cfg.vae_kl_weight`) so the
correct encoder is constructed before the shim is applied.

Reported metrics (paper Tables XI-XII): FINAL-step L2 EEF error for
CEM / replay / random, plus the per-step action-distance diagnostic
normalized by the per-step action-range L2 norm (||a_max - a_min||_2 =
6.298 for the 14-DOF bimanual action space). `classify_planning_condition`
flags each condition as 'degenerate' (broken online predictor — excluded
from inference; the Phase 3 `vae_decoder_physics` case), 'below_floor
(exploitation)', or 'at/above floor'. Planning inference is n=3 seeds per
condition (UNDERPOWERED — minimum two-sided Wilcoxon p = 0.25); the
representation-level tests (n=10, Holm) carry the statistical inference.
"""
import argparse
import json
import os

import torch

from dlejepa.cem_planning import (CEMConfig, classify_planning_condition,
                                  evaluate_cem_planning)
from dlejepa.configs import Phase3Config
from dlejepa.data_mimicgen import load_mimicgen
from dlejepa.train import build_phase3_models, build_phase3_vae_models
from dlejepa.vae import VAEDeterministic

# ---- arm definitions: (modes, builder, needs_shim) -------------------------
ARMS = {
    "sigreg": {
        "modes": ["no_physics", "decoder_physics", "encoder_physics"],
        "builder": build_phase3_models,
        "shim": False,
    },
    "vae": {
        "modes": ["vae_no_physics", "vae_decoder_physics",
                  "vae_encoder_physics"],
        "builder": build_phase3_vae_models,
        "shim": True,
    },
}


def load_model(bundle, cfg, ck_path, device):
    """Load one frozen checkpoint, dispatching on `encoder_class` so the
    correct encoder is constructed and (for the VAE arm) wrapped in the
    deterministic-mu shim. Returns (encoder, predictor, decoder, arm)."""
    sd = torch.load(ck_path, map_location=device, weights_only=False)
    arm = "vae" if sd.get("encoder_class") == "VAEEncoder" else "sigreg"

    if arm == "vae":
        enc, pred, dec = ARMS["vae"]["builder"](bundle, cfg, device)
        shim = VAEDeterministic(enc)
        shim.eval()
        enc = shim
        saved_w = sd.get("vae_checkpoint_kl_weight", sd.get("vae_kl_weight"))
        if saved_w is not None and abs(saved_w - cfg.vae_kl_weight) > 1e-9:
            print(f"[warn] {os.path.basename(ck_path)}: checkpoint KL weight "
                  f"{saved_w} != cfg.vae_kl_weight {cfg.vae_kl_weight}")
    else:
        enc, pred, dec = ARMS["sigreg"]["builder"](bundle, cfg, device)
    enc.load_state_dict(sd["encoder"]) if not isinstance(
        enc, VAEDeterministic) else enc.vae.load_state_dict(sd["encoder"])
    pred.load_state_dict(sd["predictor"])
    dec.load_state_dict(sd["decoder"])
    enc.eval(); pred.eval(); dec.eval()
    return enc, pred, dec, arm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--arm", choices=["sigreg", "vae", "both"], default="both")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--anchors", type=int, default=200)
    ap.add_argument("--cache", default="cache/phase3")
    ap.add_argument("--out", default="results/phase3_cem.json")
    args = ap.parse_args()

    cfg = Phase3Config()
    bundle = load_mimicgen(args.hdf5, cfg, args.cache)

    arms = list(ARMS) if args.arm == "both" else [args.arm]
    all_results = {}
    for arm in arms:
        for mode in ARMS[arm]["modes"]:
            for seed in range(args.seeds):
                ck = os.path.join(args.ckpt_dir, f"ckpt_{mode}_seed{seed}.pt")
                if not os.path.exists(ck):
                    print(f"[skip] missing {ck}")
                    continue
                enc, pred, dec, _ = load_model(bundle, cfg, ck, cfg.device)
                res = evaluate_cem_planning(
                    bundle, enc, pred, dec,
                    CEMConfig(n_anchors=args.anchors, seed=seed),
                    cfg.device)
                res["classification"] = classify_planning_condition(
                    res["summary"])
                all_results[f"{arm}/{mode}/seed{seed}"] = res
                s = res["summary"]
                print(f"\n== {arm}/{mode} seed {seed} ==")
                for k in ("cem_error", "replay_error", "random_error",
                          "act_dist_l2", "act_dist_frac"):
                    m, sd_ = s[k]
                    print(f"  {k:16s}: {m:.4f} ± {sd_:.4f}")
                print(f"  classification  : {res['classification']}")
                print("  >> cem_error below replay_error with a large "
                      "act_dist_frac = model exploitation (paper "
                      "Sec VII-D8); 'degenerate' conditions are excluded "
                      "from inference.")

    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
