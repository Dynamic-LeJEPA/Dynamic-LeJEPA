"""Latent CEM planning + model-exploitation diagnostic (paper Sec VII-D8).

Protocol: anchor t0 in a held-out demo; goal = GROUND-TRUTH recorded EEF
pose at t0+H (never a model-generated proxy). CEM searches H-step action
sequences, rolling candidates through the frozen predictor and scoring them
by decoded-EEF distance to the goal at every step. Baselines: real-action
replay (the ground-truth solution / expected lower bound) and random rollout.

Diagnostic: if CEM's model-internal cost beats replay while its actions are
far from the recorded ones, the planner is EXPLOITING the learned decoder —
uniformly across ablation conditions AND encoder mechanisms in the paper
(~22-26% of the action range; Tables XI-XII).

VAE arm (paper Sec VII-D3): pass the `VAEDeterministic` shim as ``encoder``.
All latents (CEM scoring AND replay/random baselines) are then deterministic
posterior means mu — the replay floor is not inflated by sampling noise, and
the identical protocol applies across the SIGReg and VAE arms. The
`vae_decoder_physics` condition is DEGENERATE by construction of its broken
online predictor (paper Table XI caption): CEM = replay = random exactly;
`classify_planning_condition` flags it for exclusion from inference.

Matched anchors across arms (paper protocol): call
`evaluate_cem_planning` once per model with `CEMConfig(seed=model_seed)`
— anchor sets then coincide across the SIGReg and VAE arms for the same
model seed {0, 1, 2}.

Metrics vs. the original notebooks (deliberate, documented):
  * Search objective: mean-per-step squared decoded-EEF distance (the
    notebooks used the per-step SUM — a monotone rescale, so elite
    selection and the returned solution are IDENTICAL).
  * Reported errors (Table XI): FINAL-step L2 EEF error after rolling the
    chosen actions — `_final_step_error`, for CEM, replay, and random
    alike.
  * Exploitation diagnostic (Table XII): per-step L2 distance between the
    CEM solution and the recorded trajectory, meaned over steps,
    normalized by the per-step action-range L2 norm
    (||a_max - a_min||_2 = 6.298 for the 14-DOF bimanual action space).
"""
from dataclasses import dataclass

import numpy as np
import torch

from .data_mimicgen import encode_frames


@dataclass
class CEMConfig:
    horizon: int = 8
    population: int = 256
    iterations: int = 4
    n_elite: int = 32
    n_anchors: int = 200
    seed: int = 0


@torch.no_grad()
def _batched_cost(predictor, decoder, z0, actions, eef_idx, goal):
    """actions: (P, H, A). Search objective = mean over steps of squared
    decoded-EEF distance to the goal.

    NOTE: the notebooks optimized the per-step SUM; dividing by H is a
    monotone rescale, so elite selection and the returned solution are
    identical. This is the SEARCH objective only — reported metrics come
    from `_final_step_error`."""
    P, H, _ = actions.shape
    z = z0.unsqueeze(0).expand(P, -1)
    total = torch.zeros(P, device=z.device)
    for h in range(H):
        z = predictor(z, actions[:, h, :])
        eef = decoder(z)[:, eef_idx]
        total += ((eef - goal) ** 2).sum(dim=1)
    return total / H


@torch.no_grad()
def _final_step_error(predictor, decoder, z0, actions, eef_idx, goal, device):
    """Reported metric (paper Table XI): FINAL-step L2 EEF error after
    rolling `actions` (P, H, A) through the frozen predictor. Returns a
    length-P tensor of L2 distances."""
    z = z0.to(device).unsqueeze(0).expand(actions.shape[0], -1)
    for h in range(actions.shape[1]):
        z = predictor(z, actions[:, h, :])
    q_final = decoder(z)[:, eef_idx]
    return ((q_final - goal.to(device)) ** 2).sum(dim=1).sqrt()


@torch.no_grad()
def cem_plan(predictor, decoder, z0, goal, eef_idx, act_lo, act_hi,
             cfg: CEMConfig, device):
    z0 = z0.to(device)
    goal = goal.to(device)
    lo = torch.as_tensor(act_lo, dtype=torch.float32, device=device)
    hi = torch.as_tensor(act_hi, dtype=torch.float32, device=device)
    A = lo.shape[0]
    mean = (lo + hi) / 2
    std = ((hi - lo) / 4).clamp(min=1e-3)
    g = torch.Generator(device="cpu").manual_seed(cfg.seed)
    for _ in range(cfg.iterations):
        eps = torch.randn(cfg.population, cfg.horizon, A, generator=g).to(device)
        samples = (mean.unsqueeze(0) + std.unsqueeze(0) * eps).clamp(
            lo.unsqueeze(0).unsqueeze(0), hi.unsqueeze(0).unsqueeze(0))
        costs = _batched_cost(predictor, decoder, z0, samples, eef_idx, goal)
        elites = samples[costs.topk(cfg.n_elite, largest=False).indices]
        mean = elites.mean(dim=0)
        std = elites.std(dim=0).clamp(min=1e-3)
    finals = (mean.unsqueeze(0).expand(cfg.population, -1, -1)
              + std.unsqueeze(0) * torch.randn(
                  cfg.population, cfg.horizon, A, generator=g).to(device))
    finals = finals.clamp(lo.unsqueeze(0).unsqueeze(0),
                          hi.unsqueeze(0).unsqueeze(0))
    costs = _batched_cost(predictor, decoder, z0, finals, eef_idx, goal)
    best = finals[costs.argmin()]
    return best, float(costs.min())


def classify_planning_condition(summary: dict, rel_tol: float = 1e-3) -> str:
    """Paper Table XI flags, from a `summary` dict of (mean, std) pairs.

    Returns one of:
      'degenerate'            — CEM = replay = random within tolerance
                                (broken online predictor; e.g. the Phase 3
                                vae_decoder_physics condition). Excluded
                                from planning-based inference.
      'below_floor (exploitation)' — CEM beats the real-action replay floor
                                — impossible for genuine planning; the
                                model-exploitation signature.
      'at/above floor'        — no exploitation signature detected."""
    cem = summary["cem_error"][0]
    rep = summary["replay_error"][0]
    rnd = summary["random_error"][0]
    scale = max(1.0, abs(rep))
    if (abs(cem - rep) < rel_tol * scale
            and abs(rnd - rep) < rel_tol * scale):
        return "degenerate"
    return "below_floor (exploitation)" if cem < rep else "at/above floor"


@torch.no_grad()
def evaluate_cem_planning(bundle, encoder, predictor, decoder, cfg: CEMConfig,
                          device, n_anchors=None):
    """Per-model aggregate: CEM vs replay vs random FINAL-step L2 EEF error
    (paper Table XI) + the action-space exploitation diagnostic (Table XII).

    Arm-agnostic: pass the `VAEDeterministic` shim as ``encoder`` for the
    VAE arm — deterministic-mu latents throughout, matched protocol across
    arms. Anchor sets are a pure function of `cfg.seed`; call once per
    model with `CEMConfig(seed=model_seed)` to obtain the paper's matched
    cross-arm anchor sets.

    Returns {"summary": {key: (mean, std)}, "per_anchor": {key: [float]}}
    with keys: cem_error, replay_error, random_error, act_dist_l2,
    act_dist_frac. `summary` feeds `classify_planning_condition`."""
    n_anchors = n_anchors or cfg.n_anchors
    H = cfg.horizon
    rng = np.random.RandomState(cfg.seed)
    cand = []
    for d in bundle.VAL_DEMOS:
        s, e = bundle.DEMO_START[d], bundle.DEMO_START[d + 1]
        if e - s > H + 2:
            cand += list(range(s, e - H - 1))
    anchors = np.array(rng.choice(cand, min(n_anchors, len(cand)),
                                  replace=False))

    lo = bundle.ACT[bundle.TRAIN_FRAMES].min(0)
    hi = bundle.ACT[bundle.TRAIN_FRAMES].max(0)
    # per-step action-range L2 norm — the Table XII normalizer
    # (6.298 for the 14-DOF bimanual action space)
    action_range_l2 = float(np.sqrt(((hi - lo) ** 2).sum()))
    eef_idx = torch.as_tensor(bundle.EEF_IDX, dtype=torch.long, device=device)

    Z0 = encode_frames(encoder, bundle, anchors, device)
    res = {"cem_error": [], "replay_error": [], "random_error": [],
           "act_dist_l2": [], "act_dist_frac": []}
    for i, t0 in enumerate(anchors):
        z0 = torch.from_numpy(Z0[i]).float()
        goal = torch.from_numpy(bundle.PROP_N[t0 + H][bundle.EEF_IDX]).float()
        a_true = torch.from_numpy(bundle.ACT[t0:t0 + H]).float().to(device)

        a_cem, _ = cem_plan(predictor, decoder, z0, goal, eef_idx, lo, hi,
                            CEMConfig(**{**vars(cfg),
                                         "seed": cfg.seed + i}), device)

        cem_err = float(_final_step_error(predictor, decoder, z0,
                                          a_cem.unsqueeze(0), eef_idx,
                                          goal, device))
        rep_err = float(_final_step_error(predictor, decoder, z0,
                                          a_true.unsqueeze(0), eef_idx,
                                          goal, device))
        a_rand = torch.from_numpy(
            rng.uniform(lo, hi, (1, H, len(lo)))).float().to(device)
        rnd_err = float(_final_step_error(predictor, decoder, z0, a_rand,
                                          eef_idx, goal, device))

        # model-exploitation diagnostic (Table XII): per-step L2 distance
        # between the CEM solution and the ground-truth trajectory,
        # normalized by the per-step action-range L2 norm
        dist_l2 = float(np.sqrt(
            ((a_cem.cpu().numpy() - bundle.ACT[t0:t0 + H]) ** 2)
            .sum(1)).mean())
        res["cem_error"].append(cem_err)
        res["replay_error"].append(rep_err)
        res["random_error"].append(rnd_err)
        res["act_dist_l2"].append(dist_l2)
        res["act_dist_frac"].append(dist_l2 / action_range_l2)

    return {"summary": {k: (float(np.mean(v)), float(np.std(v)))
                        for k, v in res.items()},
            "per_anchor": {k: [float(x) for x in v] for k, v in res.items()}}
