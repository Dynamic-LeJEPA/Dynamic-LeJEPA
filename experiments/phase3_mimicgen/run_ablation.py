#!/usr/bin/env python
"""Phase 3 ablation on MimicGen two_arm_threading (paper Sec VII-D).

3 modes x 10 seeds. 15 epochs = primary cohort (Table IX/X);
30 epochs = budget replication (Sec VII-D6, Fig 11).
Auto-resumes: completed (mode, seed) pairs are skipped.
"""
import argparse
import json
import os
import pickle
from collections import defaultdict

import numpy as np

from dlejepa.configs import Phase3Config
from dlejepa.data_mimicgen import load_mimicgen
from dlejepa.stats import holm_bonferroni, paired_wilcoxon
from dlejepa.train import train_mimicgen_run

MODES = ["no_physics", "decoder_physics", "encoder_physics"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", required=True,
                    help="path to two_arm_threading.hdf5 (or a dir to scan)")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--cache", default="cache/phase3")
    ap.add_argument("--ckpt", default="checkpoints/phase3")
    ap.add_argument("--out", default="results/phase3")
    args = ap.parse_args()

    cfg = Phase3Config(num_epochs=args.epochs, num_seeds=args.seeds)
    bundle = load_mimicgen(args.hdf5, cfg, args.cache)
    os.makedirs(args.out, exist_ok=True)
    pkl = os.path.join(args.out, "ablation_results.pkl")

    results = defaultdict(list)
    if os.path.exists(pkl):
        with open(pkl, "rb") as f:
            for mode, runs in pickle.load(f).items():
                results[mode] += runs
        print(f"resumed: {sum(len(v) for v in results.values())} existing runs")

    for mode in MODES:
        done = {r["seed"] for r in results[mode]}
        for seed in range(args.seeds):
            if seed in done:
                print(f"[skip] {mode} seed {seed}")
                continue
            print(f"[run ] {mode} seed {seed}")
            r = train_mimicgen_run(mode, seed, bundle, cfg,
                                   ckpt_dir=args.ckpt)
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
    stats = {"epochs": args.epochs,
             "means": {m: float(np.mean(v)) for m, v in d.items()},
             "stds": {m: float(np.std(v)) for m, v in d.items()},
             "wilcoxon": {}}
    print("\n==== Phase 3 d_eff (mean ± std, covariance estimator) ====")
    for m in MODES:
        print(f"  {m:17s}: {np.mean(d[m]):.4f} ± {np.std(d[m]):.4f}")
    for (name, _, _), p, ph in zip(comps, pvals, holm):
        print(f"  {name:14s}: p={p:.4f}  p_holm={ph:.4f}")
        stats["wilcoxon"][name] = {"p": p, "p_holm": float(ph)}
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
