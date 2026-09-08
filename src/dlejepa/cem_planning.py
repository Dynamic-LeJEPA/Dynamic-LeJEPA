"""Latent CEM planning + model-exploitation diagnostic (paper Sec VII-D8).

Protocol: anchor t0 in a held-out demo; goal = GROUND-TRUTH recorded EEF
pose at t0+H (never a model-generated proxy). CEM searches H-step action
sequences, rolling candidates through the frozen predictor and scoring them
by decoded-EEF distance to the goal at every step. Baselines: real-action
replay (the ground-truth solution / expected lower bound) and random rollout.

Diagnostic: if CEM's model-internal cost beats replay while its actions are
far from the recorded ones, the planner is EXPLOITING the learned decoder —
uniformly across ablation conditions in the paper (~26% of the action range).

NOTE: faithful re-implementation of the paper's spec (H=8, pop 256, 4 iters,
elite 32, 200 anchors). Cross-check against the original notebook for
bit-level parity.
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
    """actions: (P, H, A). Cost = mean over steps of squared decoded-EEF
    distance to the goal."""
    P, H, _ = actions.shape
    z = z0.unsqueeze(0).expand(P, -1)
    total = torch.zeros(P, device=z.device)
    for h in range(H):
        z = predictor(z, actions[:, h, :])
        eef = decoder(z)[:, eef_idx]
        total += ((eef - goal) ** 2).sum(dim=1)
    return total / H


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


@torch.no_grad()
def evaluate_cem_planning(bundle, encoder, predictor, decoder, cfg: CEMConfig,
                          device, n_anchors=None):
    """Per-model aggregate: CEM vs replay vs random model-internal cost +
    the action-space exploitation diagnostic."""
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
    act_range = np.maximum(hi - lo, 1e-6)
    eef_idx = torch.as_tensor(bundle.EEF_IDX, dtype=torch.long, device=device)

    Z0 = encode_frames(encoder, bundle, anchors, device)
    res = {"cem_cost": [], "replay_cost": [], "random_cost": [],
           "act_dist_frac": []}
    for i, t0 in enumerate(anchors):
        z0 = torch.from_numpy(Z0[i]).float()
        goal = torch.from_numpy(bundle.PROP_N[t0 + H][bundle.EEF_IDX]).float()
        a_true = torch.from_numpy(bundle.ACT[t0:t0 + H]).float().to(device)

        a_cem, c_cem = cem_plan(predictor, decoder, z0, goal, eef_idx, lo, hi,
                                CEMConfig(**{**vars(cfg),
                                             "seed": cfg.seed + i}), device)
        c_replay = float(_batched_cost(predictor, decoder, z0,
                                       a_true.unsqueeze(0), eef_idx, goal))
        a_rand = torch.from_numpy(
            rng.uniform(lo, hi, (1, H, len(lo)))).float().to(device)
        c_rand = float(_batched_cost(predictor, decoder, z0, a_rand,
                                     eef_idx, goal))

        # model-exploitation diagnostic: normalized action-space distance
        # between the CEM solution and the ground-truth trajectory
        dist = float(np.mean(np.abs(a_cem.cpu().numpy() - bundle.ACT[t0:t0 + H])
                             / act_range[None, :]))
        res["cem_cost"].append(c_cem)
        res["replay_cost"].append(c_replay)
        res["random_cost"].append(c_rand)
        res["act_dist_frac"].append(dist)

    return {k: (float(np.mean(v)), float(np.std(v)))
            for k, v in res.items()}
