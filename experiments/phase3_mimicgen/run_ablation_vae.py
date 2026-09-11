#!/usr/bin/env python
"""Phase 3 VAE arm (paper Sec. VII-D3) — 3 conditions x 10 paired seeds.

The encoder-mechanism comparison: identical trunk / predictor / decoder /
physics-placement logic / data / protocol (15 epochs x 150 steps, 10 paired
seeds, budget replication at 30 via --epochs) as the SIGReg arm
(`run_ablation.py`) — ONLY the encoder's distribution mechanism changes:
the closed-form marginal KL(q(z|x) || N(0,I)) on a Gaussian (mu, logvar)
head replaces SIGReg+.

CALIBRATED REGIME (KL weight 5.0, pre-run — see `calibrate_vae_kl.py`):
unlike the Phase 2 arm (weak-prior, ordering-only interpretation), this
arm is the paper's PRIMARY encoder-mechanism test and includes the
physics-free `vae_no_physics` reference condition, enabling the clean
scale-vs-structure claim within the arm.

The marginal-only KL is DELIBERATE: it instantiates the hypothesis that
correlated collapse is a property of MARGINAL-ONLY regularizers on
low-intrinsic-dimension data — not a SIGReg idiosyncrasy (paper Sec.
VII-D3). Evaluation and planning use DETERMINISTIC posterior means
(VAEDeterministic shim), so the covariance d_eff estimator (Remark VI.3)
applies identically across arms and the replay floor is not noise-inflated.

Resume-aware: completed (mode, seed) pairs auto-skip; results are saved
after EVERY run and written to a SEPARATE file from the SIGReg arm's, so
the four-arm scorecard (`run_model_comparison.py`) stays recomputable from
independent arm files.

Placement-rule check printed at the end: decoder > encoder on covariance
d_eff (paired Wilcoxon; published result p = 0.0020, Holm 0.0059,
Delta = +0.022, d_z = +5.05).
"""
import argparse
import json
import os
import pickle
import time
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon

from dlejepa.configs import Phase3Config
from dlejepa.data_mimicgen import load_mimicgen
from dlejepa.train import train_mimicgen_vae_run
from dlejepa.vae import build_vae_models  # noqa: F401  (import parity check)


def load_results(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        return {m: list(runs) for m, runs in data.items()}
    return {}


def save_results(results, pkl_path, json_path):
    os.makedirs(os.path.dirname(pkl_path) or ".", exist_ok=True)
    with open(pkl_path, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    # readable provenance summary (results/ committed as JSON per repo policy)
    js = {}
    for mode, runs in results.items():
        js[mode] = [{
            "seed": r["seed"],
            "d_eff": r["final_metrics"]["d_eff"],
            "H_ratio": r["final_metrics"]["H_ratio"],
            "scale_ratio": r["final_metrics"]["scale_ratio"],
            "mu_d_eff": r["vae_stats"]["mu_d_eff"],
            "signal_fraction": r["vae_stats"]["signal_fraction_mean"],
            "posterior_std": r["vae_stats"]["posterior_std_mean"],
            "kl": r["vae_stats"]["kl_mean"],
            "pred_r2": r.get("prediction", {}).get("r2_neural"),
            "probe_action_r2": r.get("probes", {}).get("action_r2_active"),
            "probe_proprio_r2": r.get("probes", {}).get("proprio_r2"),
        } for r in runs]
    with open(json_path, "w") as f:
        json.dump(js, f, indent=2)


def leaf(run, path):
    v = run
    for k in path.split("."):
        v = v[k]
    return float(v)


def vals(results, mode, path):
    return np.array([leaf(r, path) for r in
                     sorted(results.get(mode, []), key=lambda r: r["seed"])],
                    dtype=float)


def mv(results, mode, path):
    v = vals(results, mode, path)
    return (float(v.mean()), float(v.std())) if len(v) else (float("nan"),
                                                             float("nan"))


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj, prev = np.empty(m), 0.0
    for rank, i in enumerate(order):
        prev = max(prev, (m - rank) * p[i])
        adj[i] = min(prev, 1.0)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--cache", default="cache/phase3")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=None,
                    help="default = cfg.num_epochs (15); set 30 for the "
                         "budget replication")
    ap.add_argument("--ckpt-dir", default="checkpoints/phase3_vae")
    ap.add_argument("--out-pkl", default="results/phase3/phase3_ablation_"
                                         "results_vae.pkl")
    ap.add_argument("--out-json", default="results/phase3/vae_summary_"
                                          "15epo.json")
    args = ap.parse_args()

    cfg = Phase3Config()
    if args.epochs is not None:
        cfg.num_epochs = args.epochs
        args.out_json = args.out_json.replace("15epo", f"{args.epochs}epo")

    # ---- published-regime guard (calibrated Phase 3 arm) ------------------
    assert abs(cfg.vae_kl_weight - 5.0) < 1e-9, (
        f"vae_kl_weight = {cfg.vae_kl_weight}, but the published Phase 3 "
        f"operating point is 5.0 (calibrated). Re-run "
        f"calibrate_vae_kl.py and the ablation before changing it.")
    if not cfg.vae_kl_calibrated:
        print("[WARN] cfg.vae_kl_calibrated is False — confirm the KL "
              "weight is intentional before the full ablation.")

    modes = list(cfg.vae_modes)                      # 3 conditions (incl.
    if cfg.vae_plus_cov_weight > 0:                  # vae_no_physics)
        modes += ["vaeplus_decoder_physics", "vaeplus_encoder_physics"]
    print("=" * 72)
    print(f"PHASE 3 VAE ABLATION — {cfg.primary_task} | "
          f"{len(modes)} modes x {args.seeds} seeds | "
          f"{cfg.num_epochs} epochs x {cfg.steps_per_epoch} steps | "
          f"KL weight {cfg.vae_kl_weight}")
    print(f"  eval protocol: deterministic posterior means "
          f"(vae_eval_deterministic = {cfg.vae_eval_deterministic})")
    print(f"  results file : {args.out_pkl}  (separate from the SIGReg arm)")
    print("=" * 72)

    bundle = load_mimicgen(args.hdf5, cfg, args.cache)

    results = load_results(args.out_pkl)
    completed = sum(len(v) for v in results.values())
    total = len(modes) * args.seeds
    print(f"Progress: {completed}/{total} done, "
          f"{total - completed} remaining")

    t0 = time.time()
    newly = 0
    for mode in modes:
        done = {r["seed"] for r in results.get(mode, [])}
        for seed in range(args.seeds):
            if seed in done:
                print(f"  [skip] {mode} | seed {seed}")
                continue
            print(f"\n{'-' * 72}\nTRAINING (VAE): {mode} | seed {seed}\n"
                  f"{'-' * 72}")
            res = train_mimicgen_vae_run(mode, seed, bundle, cfg,
                                         ckpt_dir=args.ckpt_dir)
            results.setdefault(mode, []).append(res)
            newly += 1
            save_results(results, args.out_pkl, args.out_json)
            fm = res["final_metrics"]
            vs = res["vae_stats"]
            print(f"  [{mode} | seed {seed}] "
                  f"{res['elapsed_minutes']:.1f} min | "
                  f"d_eff={fm['d_eff']:.4f} H={fm['H_ratio']:.4f} "
                  f"scale={fm['scale_ratio']:.4f} | "
                  f"mu_d_eff={vs['mu_d_eff']:.4f} "
                  f"sig_frac={vs['signal_fraction_mean']:.4f} | "
                  f"predR2={res['prediction']['r2_neural']:.3f}")

    print(f"\nDone: {sum(len(v) for v in results.values())} runs total "
          f"({newly} this session, "
          f"{(time.time() - t0) / 60:.0f} min). Saved -> {args.out_pkl}")

    # =====================================================================
    # Aggregation + placement-rule check (paper Tables + Sec. VII-D3)
    # =====================================================================
    dist = [("d_eff", "final_metrics.d_eff"),
            ("H_ratio", "final_metrics.H_ratio"),
            ("scale_ratio", "final_metrics.scale_ratio"),
            ("mu_d_eff", "vae_stats.mu_d_eff"),
            ("signal_fraction", "vae_stats.signal_fraction_mean"),
            ("KL", "vae_stats.kl_mean"),
            ("pred R2", "prediction.r2_neural"),
            ("probe R2 (action)", "probes.action_r2_active"),
            ("probe R2 (proprio)", "probes.proprio_r2")]
    present = [m for m in modes if len(results.get(m, []))]
    print("\n" + "=" * 96)
    print(f"TABLE: PHASE 3 VAE ABLATION — {cfg.primary_task} "
          f"(mean +/- std; covariance d_eff on DETERMINISTIC posterior "
          f"means; N/K = {cfg.eval_frames // cfg.K})")
    print("=" * 96)
    print(f"{'Metric':<22}" + "".join(f"{m:>20}" for m in present))
    print("-" * 96)
    for label, path in dist:
        cells = []
        for m in present:
            mu_, sd_ = mv(results, m, path)
            cells.append(f"{mu_:>9.4f}+/-{sd_:<8.4f}")
        print(f"{label:<22}" + "".join(cells))
    print("=" * 96)
    print("NOTE: sampled-latent marginals can look healthy while the "
          "signal is dead — read signal_fraction / mu_d_eff "
          "(paper Remark 'Posterior Signal Fraction').")

    # ---- paired Wilcoxon + Holm within the arm ----------------------------
    pair_map = ([("vae_no_physics", "vae_decoder_physics"),
                 ("vae_no_physics", "vae_encoder_physics")]
                if "vae_no_physics" in present else []) + \
               [("vae_encoder_physics", "vae_decoder_physics")]
    print("\n" + "=" * 76)
    print("PAIRED WILCOXON (Holm-corrected) — within the VAE family")
    print("=" * 76)
    rows = []
    for a, b in pair_map:
        if a not in present or b not in present:
            continue
        x, y = vals(results, a, "final_metrics.d_eff"), \
               vals(results, b, "final_metrics.d_eff")
        n = min(len(x), len(y))
        if n < 5:
            print(f"  {a} vs {b}: paired n={n} — insufficient (<5)")
            continue
        _, p = wilcoxon(x[:n], y[:n])
        rows.append((f"{a} vs {b}", n, p, float(np.mean(x[:n] - y[:n]))))
    padj = holm([r[2] for r in rows]) if rows else []
    for (name, n, p, md), ph in zip(rows, padj):
        print(f"  {name:<52} n={n} p={p:.4f} p_Holm={ph:.4f} "
              f"Delta={md:+.4f} "
              f"{'[significant]' if ph < 0.05 else '[n.s.]'}")
    print("=" * 76)

    # ---- scale-vs-structure within the arm (reference = vae_no_physics) ---
    if "vae_no_physics" in present:
        sc0 = mv(results, "vae_no_physics", "final_metrics.scale_ratio")[0]
        de0 = mv(results, "vae_no_physics", "final_metrics.d_eff")[0]
        print("\n" + "=" * 76)
        print("SCALE-vs-STRUCTURE — VAE family (reference = vae_no_physics)")
        print("=" * 76)
        d_gaps = {}
        for m in ("vae_decoder_physics", "vae_encoder_physics"):
            if m not in present:
                continue
            dsc = sc0 - mv(results, m, "final_metrics.scale_ratio")[0]
            dde = de0 - mv(results, m, "final_metrics.d_eff")[0]
            d_gaps[m] = dde
            print(f"  Delta scale({m}) = {dsc:+.4f}   "
                  f"Delta d_eff({m}) = {dde:+.4f}")
        if len(d_gaps) == 2:
            ratio = d_gaps["vae_encoder_physics"] / \
                    max(abs(d_gaps["vae_decoder_physics"]), 1e-9)
            print(f"  Delta structure (enc - dec) = "
                  f"{d_gaps['vae_encoder_physics'] - d_gaps['vae_decoder_physics']:+.4f}"
                  f" | enc/dec gap ratio = {ratio:.2f}x")
            print("  -> placement-rule signature WITHIN the VAE family "
                  "(the replicated quantity is the ORDERING, not the sign "
                  "of the absolute physics effect — paper Sec. VII-D3).")
        print("=" * 76)
    else:
        print("\n[info] vae_no_physics absent — scale-vs-structure "
              "reference falls back to vae_decoder_physics in "
              "run_model_comparison.py (ENTANGLES decoder gradients; "
              "run the physics-free arm for the clean claim).")

    # ---- placement-rule verdict (published: dec > enc, p = 0.0020) --------
    if "vae_decoder_physics" in present and "vae_encoder_physics" in present:
        x = vals(results, "vae_decoder_physics", "final_metrics.d_eff")
        y = vals(results, "vae_encoder_physics", "final_metrics.d_eff")
        n = min(len(x), len(y))
        _, p = wilcoxon(x[:n], y[:n])
        gap = float(np.mean(x[:n] - y[:n]))
        dz = gap / max(float(np.std(x[:n] - y[:n], ddof=1)), 1e-12)
        print("\n" + "=" * 76)
        print(f"PLACEMENT RULE (VAE family): decoder - encoder = {gap:+.4f} "
              f"| d_z = {dz:+.2f} | paired Wilcoxon p = {p:.4f}")
        print(f"  published: +0.0220, d_z = +5.05, p = 0.0020 "
              f"(Holm 0.0059) — {'MATCH' if gap > 0 and p <= 0.01 else 'CHECK'}")
        print("=" * 76)


if __name__ == "__main__":
    main()
