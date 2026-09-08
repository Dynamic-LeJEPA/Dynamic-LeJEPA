#!/usr/bin/env python
"""Phase 2 ablation: nuScenes Trainval Part 1 (paper Sec VII-C).

3 modes x 10 seeds x 30 epochs; paired Wilcoxon across shared seeds on the
covariance d_eff; scale-vs-structure decomposition (Sec VII-C3).

Requirements:
  --data-root : v1.0-trainval01_blobs root containing samples/ + sweeps/
  --meta-root : v1.0-trainval metadata dir containing ego_pose.json
                (strongly recommended: bypasses ~3.5 h of ICP per pass)

Auto-resumes: completed (mode, seed) pairs are skipped.
"""
import argparse
import json
import os
import pickle
from collections import defaultdict

import numpy as np
from torch.utils.data import DataLoader

from dlejepa.configs import Phase2Config
from dlejepa.data_nuscenes import (NuScenesPairsDataset, NuScenesParser,
                                   load_ego_poses_fast, precompute_caches)
from dlejepa.stats import holm_bonferroni, paired_wilcoxon
from dlejepa.train import train_nuscenes_run

MODES = ["no_physics", "decoder_physics", "encoder_physics"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True,
                    help="v1.0-trainval01_blobs root (samples/ + sweeps/)")
    ap.add_argument("--meta-root", default=None,
                    help="v1.0-trainval metadata with ego_pose.json")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--cache", default="cache/phase2")
    ap.add_argument("--ckpt", default="checkpoints/phase2")
    ap.add_argument("--out", default="results/phase2")
    args = ap.parse_args()

    cfg = Phase2Config(data_root=args.data_root,
                       meta_root=args.meta_root or "",
                       num_epochs=args.epochs, num_seeds=args.seeds)
    pose_dict = load_ego_poses_fast(cfg.meta_root) if cfg.meta_root else None

    # ---------- data ----------
    sequences = NuScenesParser.build_sequences(
        camera=cfg.primary_camera, lidar=cfg.lidar_sensor,
        data_root=cfg.data_root, use_sweeps=cfg.use_sweeps)
    assert sequences, "no sequences built -- check --data-root"
    print(f"N/K (all pairs): {len(sequences) / cfg.K:.1f} "
          f"(>= 5 required per Corollary VI.2)")
    train_seqs, val_seqs = NuScenesParser.split_by_scene(
        sequences, cfg.train_scene_ratio, cfg.split_seed)

    ego_cache, depth_cache = precompute_caches(sequences, args.cache,
                                               pose_dict=pose_dict)
    train_ds = NuScenesPairsDataset(train_seqs, cfg.img_size,
                                    ego_cache, depth_cache)
    val_ds = NuScenesPairsDataset(val_seqs, cfg.img_size,
                                  ego_cache, depth_cache)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=True, num_workers=0, pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size,
                            shuffle=False, num_workers=0, pin_memory=True)

    # ---------- resume ----------
    os.makedirs(args.out, exist_ok=True)
    pkl = os.path.join(args.out, "ablation_results.pkl")
    results = defaultdict(list)
    if os.path.exists(pkl):
        with open(pkl, "rb") as f:
            for mode, runs in pickle.load(f).items():
                results[mode] += runs
        print(f"resumed: {sum(len(v) for v in results.values())} existing runs")

    # ---------- ablation loop ----------
    for mode in MODES:
        done = {r["seed"] for r in results[mode]}
        for seed in range(args.seeds):
            if seed in done:
                print(f"[skip] {mode} seed {seed}")
                continue
            print(f"[run ] {mode} seed {seed}")
            r = train_nuscenes_run(mode, seed, cfg, train_loader,
                                   val_loader, val_ds, ckpt_dir=args.ckpt)
            results[mode].append(r)
            with open(pkl, "wb") as f:
                pickle.dump(dict(results), f,
                            protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(args.out, "ablation_results.json"),
                      "w") as f:
                json.dump({m: [{"seed": r["seed"], **r["final_metrics"]}
                               for r in runs}
                           for m, runs in results.items()}, f, indent=2)

    # ---------- paired Wilcoxon + Holm on covariance d_eff ----------
    d = {m: [r["final_metrics"]["d_eff"]
             for r in sorted(results[m], key=lambda x: x["seed"])]
         for m in MODES}
    comps = [("dec. vs. none", d["decoder_physics"], d["no_physics"]),
             ("enc. vs. none", d["encoder_physics"], d["no_physics"]),
             ("enc. vs. dec.", d["encoder_physics"], d["decoder_physics"])]
    pvals = [paired_wilcoxon(a, b)["p"] for _, a, b in comps]
    holm = holm_bonferroni(pvals)
    stats = {"means": {m: float(np.mean(v)) for m, v in d.items()},
             "stds": {m: float(np.std(v)) for m, v in d.items()},
             "wilcoxon": {}}
    print("\n==== Phase 2 d_eff (mean ± std, covariance estimator) ====")
    for m in MODES:
        print(f"  {m:17s}: {np.mean(d[m]):.4f} ± {np.std(d[m]):.4f}")
    for (name, _, _), p, ph in zip(comps, pvals, holm):
        print(f"  {name:14s}: p={p:.4f}  p_holm={ph:.4f}")
        stats["wilcoxon"][name] = {"p": p, "p_holm": float(ph)}

    # ---------- scale-vs-structure decomposition (Sec VII-C3) ----------
    sc = {m: float(np.mean([r["final_metrics"]["scale_ratio"]
                            for r in results[m]])) for m in MODES}
    ds = {
        "delta_scale_decoder": sc["no_physics"] - sc["decoder_physics"],
        "delta_scale_encoder": sc["no_physics"] - sc["encoder_physics"],
        "delta_d_eff_decoder": float(np.mean(d["no_physics"])
                                     - np.mean(d["decoder_physics"])),
        "delta_d_eff_encoder": float(np.mean(d["no_physics"])
                                     - np.mean(d["encoder_physics"])),
    }
    ds["delta_structure"] = (ds["delta_d_eff_encoder"]
                             - ds["delta_d_eff_decoder"])
    stats["scale_vs_structure"] = ds
    print(f"  delta_structure = {ds['delta_structure']:.4f} "
          f"(encoder gap / decoder gap = "
          f"{ds['delta_d_eff_encoder'] / max(ds['delta_d_eff_decoder'], 1e-9):.1f}x)")
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
