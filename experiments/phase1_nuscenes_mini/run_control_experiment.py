#!/usr/bin/env python
"""Phase 1 control experiment (paper Table IV / Prop VI.1): a true N(0, I_K)
must look 'collapsed' under the covariance d_eff estimator when N is small —
proving that low d_eff in low-data regimes is a MEASUREMENT artifact, not
embedding collapse. Optionally compares model embeddings (--z-file .npy)."""
import argparse

import numpy as np

from dlejepa.metrics import compute_effective_dim_from_embeddings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--z-file", default=None,
                    help="npy of model embeddings (N x K)")
    ap.add_argument("--K", type=int, default=256)
    args = ap.parse_args()
    rng = np.random.RandomState(42)
    print(f"{'N':>7} {'N/K':>7} {'d_eff(true N(0,I))':>20}")
    for N in [128, 256, 404, 1280, 2560, 5000, 20000]:
        z = rng.randn(N, args.K).astype(np.float32)
        m = compute_effective_dim_from_embeddings(z)
        print(f"{N:>7} {N / args.K:>7.2f} {m['d_eff']:>20.4f}")
    if args.z_file:
        z = np.load(args.z_file).astype(np.float32)
        m = compute_effective_dim_from_embeddings(z)
        print(f"\nmodel embeddings: N={m['N']}, N/K={m['N_over_K']:.2f}, "
              f"d_eff={m['d_eff']:.4f}, H-ratio={m['h_ratio']:.4f}, "
              f"scale={m['scale_ratio']:.4f}")


if __name__ == "__main__":
    main()
