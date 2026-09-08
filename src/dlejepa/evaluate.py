"""Evaluation battery.

Phase 3 (MimicGen): Theorem IV.4 (a)-(d) — embedding distribution,
action-conditioned prediction (+ linear & kNN oracles), task probes,
multi-step rollout.
Phase 1/2 (nuScenes): embedding distribution metrics on held-out pairs
(covariance d_eff, H-ratio, scale ratio, KS marginals).
"""
import numpy as np
import torch

from .configs import Phase3Config
from .data_mimicgen import MimicGenBundle, encode_frames
from .metrics import compute_effective_dim_from_embeddings, ks_gaussian_stats


def ridge_fit(X, Y, lam=1.0):
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ Y)


def r2_score(Y, Yh):
    ss = ((Y - Yh) ** 2).sum(0)
    st = ((Y - Y.mean(0)) ** 2).sum(0)
    return float(np.mean(1 - ss / np.maximum(st, 1e-12)))


# ====================== Phase 3: MimicGen ====================================

def multistep_rollout(encoder, predictor, bundle, cfg, R):
    """Multi-step rollout error vs horizon (Theorem IV.9 extension)."""
    device = cfg.device
    Hmax = max(cfg.rollout_h)
    cand = []
    for d in bundle.VAL_DEMOS:
        s, e = bundle.DEMO_START[d], bundle.DEMO_START[d + 1]
        if e - s > Hmax + 3:
            cand += list(range(s + 1, e - 1 - Hmax))
    n = min(cfg.rollout_starts, len(cand))
    starts = np.array(R.choice(cand, n, replace=False))
    idx_mat = starts[:, None] + np.arange(Hmax + 1)[None, :]
    Zall = encode_frames(encoder, bundle, idx_mat.reshape(-1), device)
    Zall = Zall.reshape(n, Hmax + 1, cfg.K)
    cur = torch.from_numpy(Zall[:, 0]).float().to(device)
    errs = {}
    for h in range(1, Hmax + 1):
        a = torch.from_numpy(bundle.ACT[idx_mat[:, h - 1]]).float().to(device)
        cur = predictor(cur, a)
        if h in cfg.rollout_h:
            errs[str(h)] = float(
                ((cur.detach().cpu().numpy() - Zall[:, h]) ** 2).mean())
    return errs


def evaluate_mimicgen_run(encoder, predictor, bundle: MimicGenBundle,
                          cfg: Phase3Config, seed: int) -> dict:
    R = np.random.RandomState(2000 + seed)
    device = cfg.device
    out = {}

    # ---------- (1) Embedding distribution: Theorem IV.4a ----------
    eval_idx = R.choice(bundle.VAL_FRAMES,
                        min(cfg.eval_frames, len(bundle.VAL_FRAMES)),
                        replace=False)
    Z = encode_frames(encoder, bundle, eval_idx, device)
    m = compute_effective_dim_from_embeddings(Z)
    out["final_metrics"] = {"d_eff": m["d_eff"], "H_ratio": m["h_ratio"],
                            "scale_ratio": m["scale_ratio"],
                            "var_std": m["var_std"], "var_mean": m["var_mean"],
                            "N": int(m["N"]),
                            "source": "val_demos_covariance"}
    cm = compute_effective_dim_from_embeddings(
        R.randn(*Z.shape).astype(np.float32))       # true-Gaussian control
    out["control_metrics"] = {"d_eff": cm["d_eff"],
                              "scale_ratio": cm["scale_ratio"],
                              "h_ratio": cm["h_ratio"]}

    # ---------- (2) Action-conditioned prediction: Theorem IV.4b ----------
    nv = min(cfg.val_triples_eval, len(bundle.VAL_IT))
    vsel = R.choice(len(bundle.VAL_IT), nv, replace=False)
    it, inn = bundle.VAL_IT[vsel], bundle.VAL_INN[vsel]
    need = np.unique(np.concatenate([it, inn]))
    Zm = encode_frames(encoder, bundle, need, device)
    pos = {int(f): i for i, f in enumerate(need)}
    z_t = Zm[[pos[int(i)] for i in it]]
    z_nx = Zm[[pos[int(i)] for i in inn]]
    a_t = bundle.ACT[it]
    with torch.no_grad():
        zp = predictor(torch.from_numpy(z_t).float().to(device),
                       torch.from_numpy(a_t).float().to(device)).cpu().numpy()
    nt = min(cfg.linear_train_triples, len(bundle.TRAIN_IT))
    tsel = R.choice(len(bundle.TRAIN_IT), nt, replace=False)
    tit, tinn = bundle.TRAIN_IT[tsel], bundle.TRAIN_INN[tsel]
    tneed = np.unique(np.concatenate([tit, tinn]))
    Ztr = encode_frames(encoder, bundle, tneed, device)
    tpos = {int(f): i for i, f in enumerate(tneed)}
    zt_tr = Ztr[[tpos[int(i)] for i in tit]]
    zn_tr = Ztr[[tpos[int(i)] for i in tinn]]
    Xtr = np.concatenate([zt_tr, bundle.ACT[tit],
                          np.ones((len(zt_tr), 1))], 1)
    Wl = ridge_fit(Xtr, zn_tr, 1.0)       # Corollary IV.5: z_hat = Az + Ba
    Xv = np.concatenate([z_t, a_t, np.ones((len(z_t), 1))], 1)
    mse = {"mean": float(((zn_tr.mean(0) - z_nx) ** 2).mean()),
           "persistence": float(((z_t - z_nx) ** 2).mean()),
           "neural": float(((zp - z_nx) ** 2).mean()),
           "linear": float(((Xv @ Wl - z_nx) ** 2).mean())}
    var_next = float(z_nx.var(0).mean())
    pred_metrics = {f"mse_{k}": v for k, v in mse.items()}
    pred_metrics["r2_neural"] = 1 - mse["neural"] / var_next
    pred_metrics["r2_linear"] = 1 - mse["linear"] / var_next
    pred_metrics["action_advantage"] = mse["persistence"] - mse["neural"]
    pred_metrics["var_next"] = var_next
    out["prediction"] = pred_metrics

    # ---------- (3) Task-relevant probes: Theorem IV.4c ----------
    pidx = R.choice(bundle.TRAIN_FRAMES,
                    min(cfg.probe_train_n, len(bundle.TRAIN_FRAMES)),
                    replace=False)
    Zp = encode_frames(encoder, bundle, pidx, device)
    Xp = np.concatenate([Zp, np.ones((len(Zp), 1))], 1)
    Wa = ridge_fit(Xp, bundle.ACT_N[pidx], 10.0)
    Wq = ridge_fit(Xp, bundle.PROP_N[pidx], 10.0)
    Xv2 = np.concatenate([Z, np.ones((len(Z), 1))], 1)
    out["probes"] = {"action_r2": r2_score(bundle.ACT_N[eval_idx], Xv2 @ Wa),
                     "action_r2_active": r2_score(
                         bundle.ACT_N[eval_idx][:, bundle.ACTIVE_ACT_DIMS],
                         (Xv2 @ Wa)[:, bundle.ACTIVE_ACT_DIMS]),
                     "proprio_r2": r2_score(bundle.PROP_N[eval_idx], Xv2 @ Wq),
                     "n_active_action_dims": int(bundle.ACTIVE_ACT_DIMS.sum())}

    # ---------- (4) Multi-step rollout ----------
    out["rollout"] = multistep_rollout(encoder, predictor, bundle, cfg, R)
    return out


# ====================== Phase 1/2: nuScenes ==================================

@torch.no_grad()
def encode_nuscenes_embeddings(model, dataset, device, max_n=None, batch=64):
    """Encode img_t of each held-out pair; embeddings carry the SIGReg
    learned scale (the distribution the maximum-entropy claim applies to)."""
    scale = model.sigreg.get_scale()
    n = len(dataset) if max_n is None else min(max_n, len(dataset))
    zs = []
    for i in range(0, n, batch):
        items = [dataset[int(j)] for j in range(i, min(i + batch, n))]
        img = torch.stack([it["img_t"] for it in items]).to(device)
        zs.append((model.encoder(img) * scale).cpu())
    return torch.cat(zs).numpy()


def evaluate_nuscenes_run(model, dataset, cfg, max_n=None):
    """Final ablation metrics for Phases 1-2 (covariance d_eff, H-ratio,
    scale ratio, KS marginal statistics) on held-out scene-level pairs."""
    Z = encode_nuscenes_embeddings(model, dataset, cfg.device, max_n=max_n)
    m = compute_effective_dim_from_embeddings(Z)
    ks = ks_gaussian_stats(Z)
    return {"final_metrics": {
        "d_eff": m["d_eff"], "H_ratio": m["h_ratio"],
        "scale_ratio": m["scale_ratio"], "var_std": m["var_std"],
        "var_mean": m["var_mean"], "N": int(m["N"]),
        "ks_stat_mean": ks["ks_stat_mean"],
        "source": "val_pairs_covariance"}}
