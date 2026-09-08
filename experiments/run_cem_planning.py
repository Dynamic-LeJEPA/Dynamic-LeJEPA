#!/usr/bin/env python
"""Latent CEM planning + model-exploitation diagnostic (paper Sec VII-D8).

Loads frozen Phase 3 checkpoints, runs the planner + replay/random baselines,
and reports the action-space exploitation diagnostic per model.
"""
import argparse
import json
import os

import torch

from dlejepa.cem_planning import CEMConfig, evaluate_cem_planning
from dlejepa.configs import Phase3Config
from dlejepa.data_mimicgen import load_mimicgen
from dlejepa.train import build_phase3_models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--anchors", type=int, default=200)
    ap.add_argument("--cache", default="cache/phase3")
    ap.add_argument("--out", default="results/phase3_cem.json")
    args = ap.parse_args()

    cfg = Phase3Config()
    bundle = load_mimicgen(args.hdf5, cfg, args.cache)
    all_results = {}
    for mode in ["no_physics", "decoder_physics", "encoder_physics"]:
        for seed in range(args.seeds):
            ck = os.path.join(args.ckpt_dir, f"ckpt_{mode}_seed{seed}.pt")
            if not os.path.exists(ck):
                print(f"[skip] missing {ck}")
                continue
            enc, pred, dec = build_phase3_models(bundle, cfg, cfg.device)
            sd = torch.load(ck, map_location=cfg.device)
            enc.load_state_dict(sd["encoder"])
            pred.load_state_dict(sd["predictor"])
            dec.load_state_dict(sd["decoder"])
            res = evaluate_cem_planning(
                bundle, enc, pred, dec,
                CEMConfig(n_anchors=args.anchors, seed=3100 + seed),
                cfg.device)
            all_results[f"{mode}/seed{seed}"] = res
            print(f"\n== {mode} seed {seed} ==")
            for k, (m, s) in res.items():
                print(f"  {k:16s}: {m:.4f} ± {s:.4f}")
            print("  >> if cem_cost < replay_cost while act_dist_frac is "
                  "large, the planner is exploiting the learned decoder "
                  "(paper Sec VII-D8)")
    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
