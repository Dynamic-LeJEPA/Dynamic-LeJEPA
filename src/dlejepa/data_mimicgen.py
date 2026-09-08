"""MimicGen data pipeline (paper Sec VII-D1).

HDF5 -> disk-backed uint8 memmap (extracted once per session), demo-level
70/30 split (analog of the scene-level split; FIXED seed 42 for paired
Wilcoxon), (t-1, t, t+1) triples for physics + prediction, and the fixed
least-squares action->EEF kinematics map W_cal (R^2 = 0.788, fit on TRAIN
demos only — no leakage).
"""
import glob
import os
import pickle
import time
from dataclasses import dataclass, field
from typing import List

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .configs import Phase3Config


@dataclass
class MimicGenBundle:
    IMG: np.memmap
    PROP: np.ndarray
    ACT: np.ndarray
    PROP_N: np.ndarray
    ACT_N: np.ndarray
    lengths: list
    DEMO_START: np.ndarray
    H: int
    W: int
    ACTION_DIM: int
    PROP_DIM: int
    PROP_DIMS: list
    EEF_IDX: np.ndarray
    W_CAL: np.ndarray
    KIN_R2: float
    TRAIN_DEMOS: list
    VAL_DEMOS: list
    TRAIN_FRAMES: np.ndarray
    VAL_FRAMES: np.ndarray
    TRAIN_IP: np.ndarray
    TRAIN_IT: np.ndarray
    TRAIN_INN: np.ndarray
    VAL_IP: np.ndarray
    VAL_IT: np.ndarray
    VAL_INN: np.ndarray
    ACTIVE_ACT_DIMS: np.ndarray
    prop_keys: List[str] = field(default_factory=list)


def _demo_num(k: str) -> int:
    tail = k.split("_")[-1]
    return int(tail) if tail.isdigit() else 0


def find_mimicgen_files(root: str) -> dict:
    hits = {}
    for f in glob.glob(os.path.join(root, "**", "*.hdf5"), recursive=True):
        hits[os.path.splitext(os.path.basename(f))[0]] = f
    return hits


def _extract(hdf5_path, camera_key, prop_keys, cache_dir):
    t0 = time.time()
    img_path = os.path.join(cache_dir, "images_u8.mm")
    prop_path = os.path.join(cache_dir, "proprio.npy")
    act_path = os.path.join(cache_dir, "actions.npy")
    meta_path = os.path.join(cache_dir, "task_meta.pkl")
    with h5py.File(hdf5_path, "r") as f:
        data = f["data"]
        keys = sorted([k for k in data.keys() if k.startswith("demo")],
                      key=_demo_num)
        lengths = [data[k]["actions"].shape[0] for k in keys]
        d0 = data[keys[0]]
        _, H, W, C = d0["obs"][camera_key].shape
        action_dim = d0["actions"].shape[1]
        prop_dims = [d0["obs"][k].shape[1] if k in d0["obs"] else 0
                     for k in prop_keys]
        prop_dim = int(sum(prop_dims))
        n_demos, total = len(keys), int(sum(lengths))
        img_mm = np.memmap(img_path, dtype=np.uint8, mode="w+",
                           shape=(total, H, W, C))
        prop = np.zeros((total, prop_dim), np.float32)
        act = np.zeros((total, action_dim), np.float32)
        demo_start = np.zeros(n_demos + 1, np.int64)
        for di, k in enumerate(keys):
            ep, L = data[k], lengths[di]
            s, e = demo_start[di], demo_start[di] + L
            img_mm[s:e] = ep["obs"][camera_key][()]
            off = 0
            for pk, pd in zip(prop_keys, prop_dims):
                if pd > 0:
                    prop[s:e, off:off + pd] = ep["obs"][pk][()]
                off += pd
            act[s:e] = ep["actions"][()]
            demo_start[di + 1] = e
        img_mm.flush()
        del img_mm
    with open(meta_path, "wb") as f:
        pickle.dump({"lengths": lengths, "demo_start": demo_start, "H": H,
                     "W": W, "action_dim": action_dim, "prop_dims": prop_dims,
                     "prop_dim": prop_dim}, f)
    np.save(prop_path, prop)
    np.save(act_path, act)
    print(f"  extracted {n_demos} demos / {total:,} frames in "
          f"{time.time() - t0:.0f}s -> {img_path} ({total * H * W * C / 1e9:.2f} GB)")
    return lengths, demo_start, H, W, action_dim, prop_dims, prop_dim


def load_mimicgen(hdf5_path: str, cfg: Phase3Config, cache_dir: str) -> MimicGenBundle:
    os.makedirs(cache_dir, exist_ok=True)
    if os.path.isdir(hdf5_path):
        files = find_mimicgen_files(hdf5_path)
        assert files, "no .hdf5 found under --hdf5"
        task = cfg.primary_task if cfg.primary_task in files else sorted(files)[0]
        hdf5_path = files[task]
    meta_path = os.path.join(cache_dir, "task_meta.pkl")
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        lengths, DEMO_START = meta["lengths"], meta["demo_start"]
        H, W, ACTION_DIM, PROP_DIMS, PROP_DIM = (meta["H"], meta["W"],
                                                 meta["action_dim"],
                                                 meta["prop_dims"],
                                                 meta["prop_dim"])
    else:
        lengths, DEMO_START, H, W, ACTION_DIM, PROP_DIMS, PROP_DIM = _extract(
            hdf5_path, cfg.camera_key, cfg.prop_keys, cache_dir)
    IMG = np.memmap(os.path.join(cache_dir, "images_u8.mm"), dtype=np.uint8,
                    mode="r", shape=(int(DEMO_START[-1]), H, W, 3))
    PROP = np.load(os.path.join(cache_dir, "proprio.npy"))
    ACT = np.load(os.path.join(cache_dir, "actions.npy"))

    # EEF indices within the 32-dim proprio vector
    EEF_IDX, off = [], 0
    for k, d in zip(cfg.prop_keys, PROP_DIMS):
        if k in ("robot0_eef_pos", "robot1_eef_pos") and d > 0:
            EEF_IDX += list(range(off, off + d))
        off += d
    EEF_IDX = np.array(EEF_IDX, dtype=np.int64)

    # demo-level split, FIXED seed -> paired Wilcoxon valid
    rng = np.random.RandomState(cfg.split_seed)
    perm = rng.permutation(len(lengths))
    n_train = int(len(lengths) * cfg.train_demo_ratio)
    TRAIN_DEMOS, VAL_DEMOS = sorted(perm[:n_train]), sorted(perm[n_train:])

    def _frames(demos):
        return np.concatenate(
            [np.arange(DEMO_START[d], DEMO_START[d + 1]) for d in demos])

    TRAIN_FRAMES, VAL_FRAMES = _frames(TRAIN_DEMOS), _frames(VAL_DEMOS)

    def _triples(demos):
        ip, it, inn = [], [], []
        for d in demos:
            s, e = DEMO_START[d], DEMO_START[d + 1]
            if e - s < 3:
                continue
            ts = np.arange(1, e - s - 1)
            ip.append(s + ts - 1)
            it.append(s + ts)
            inn.append(s + ts + 1)
        return (np.concatenate(ip), np.concatenate(it), np.concatenate(inn))

    TRAIN_IP, TRAIN_IT, TRAIN_INN = _triples(TRAIN_DEMOS)
    VAL_IP, VAL_IT, VAL_INN = _triples(VAL_DEMOS)

    # normalization stats (train demos only — no leakage)
    PROP_MEAN = PROP[TRAIN_FRAMES].mean(0)
    PROP_STD = np.maximum(PROP[TRAIN_FRAMES].std(0), 1e-6)
    ACT_MEAN, ACT_STD_RAW = ACT[TRAIN_FRAMES].mean(0), ACT[TRAIN_FRAMES].std(0)
    ACT_STD = np.maximum(ACT_STD_RAW, 1e-6)
    PROP_N = ((PROP - PROP_MEAN) / PROP_STD).astype(np.float32)
    ACT_N = ((ACT - ACT_MEAN) / ACT_STD).astype(np.float32)
    ACT_N[:, ACT_STD_RAW < 1e-4] = 0.0                 # kill constant dims
    ACTIVE_ACT_DIMS = ACT_STD_RAW > 0.05

    # ---- fixed kinematics calibration W_cal (decoder physics prior) ----
    def calibrate(max_n=50000, ridge=1.0):
        n = len(TRAIN_IT)
        sel = (np.random.RandomState(7).choice(n, min(max_n, n), replace=False)
               if n > max_n else np.arange(n))
        disp = (PROP_N[TRAIN_INN[sel]][:, EEF_IDX]
                - PROP_N[TRAIN_IT[sel]][:, EEF_IDX])   # (n, 6)
        A = ACT[TRAIN_IT[sel]]                          # (n, 14)
        W = np.linalg.solve(A.T @ A + ridge * np.eye(A.shape[1]), A.T @ disp)
        pred = A @ W
        r2 = float(np.mean(
            1 - ((disp - pred) ** 2).sum(0)
              / np.maximum(((disp - disp.mean(0)) ** 2).sum(0), 1e-12)))
        return W.astype(np.float32), r2

    W_CAL, KIN_R2 = calibrate()
    print(f"MimicGen ready: {len(lengths)} demos, {int(DEMO_START[-1]):,} frames | "
          f"kinematics R^2 (a->dEEF) = {KIN_R2:.3f} | N/K (val) = "
          f"{len(VAL_FRAMES) / cfg.K:.1f}")
    return MimicGenBundle(IMG=IMG, PROP=PROP, ACT=ACT, PROP_N=PROP_N, ACT_N=ACT_N,
                          lengths=lengths, DEMO_START=DEMO_START, H=H, W=W,
                          ACTION_DIM=ACTION_DIM, PROP_DIM=PROP_DIM,
                          PROP_DIMS=PROP_DIMS, EEF_IDX=EEF_IDX, W_CAL=W_CAL,
                          KIN_R2=KIN_R2, TRAIN_DEMOS=TRAIN_DEMOS,
                          VAL_DEMOS=VAL_DEMOS, TRAIN_FRAMES=TRAIN_FRAMES,
                          VAL_FRAMES=VAL_FRAMES, TRAIN_IP=TRAIN_IP,
                          TRAIN_IT=TRAIN_IT, TRAIN_INN=TRAIN_INN,
                          VAL_IP=VAL_IP, VAL_IT=VAL_IT, VAL_INN=VAL_INN,
                          ACTIVE_ACT_DIMS=ACTIVE_ACT_DIMS,
                          prop_keys=cfg.prop_keys)


class MimicGenTripleDS(Dataset):
    """(t-1, t, t+1) windows: physics needs triples, prediction needs pairs."""

    def __init__(self, bundle: MimicGenBundle, ip, it, inn):
        self.b, self.ip, self.it, self.inn = bundle, ip, it, inn

    def __len__(self):
        return len(self.it)

    def __getitem__(self, i):
        b = self.b
        a, t, n = int(self.ip[i]), int(self.it[i]), int(self.inn[i])
        x = torch.from_numpy(np.ascontiguousarray(
            np.stack([b.IMG[a], b.IMG[t], b.IMG[n]])))
        x = x.permute(0, 3, 1, 2).float().div_(255.).sub_(0.5).div_(0.5)
        p = torch.from_numpy(
            np.stack([b.PROP_N[a], b.PROP_N[t], b.PROP_N[n]])).float()
        return {"img_prev": x[0], "img_t": x[1], "img_next": x[2],
                "prop_prev": p[0], "prop_t": p[1], "prop_next": p[2],
                "action": torch.from_numpy(b.ACT[t]).float()}


def make_train_loader(bundle: MimicGenBundle, cfg: Phase3Config, seed: int):
    r = np.random.RandomState(1000 + seed)
    n = len(bundle.TRAIN_IT)
    sel = (r.choice(n, min(cfg.train_triples_cap, n), replace=False)
           if n > cfg.train_triples_cap else np.arange(n))
    ds = MimicGenTripleDS(bundle, bundle.TRAIN_IP[sel], bundle.TRAIN_IT[sel],
                          bundle.TRAIN_INN[sel])
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                      num_workers=cfg.num_workers, pin_memory=True,
                      drop_last=True)


def _to_img_tensor(arr):
    x = torch.from_numpy(np.ascontiguousarray(arr)).permute(0, 3, 1, 2).float()
    return x.div_(255.).sub_(0.5).div_(0.5)


@torch.no_grad()
def encode_frames(encoder, bundle: MimicGenBundle, indices, device, batch=256):
    encoder.eval()
    zs, idx = [], np.asarray(indices)
    for i in range(0, len(idx), batch):
        c = idx[i:i + batch]
        x = _to_img_tensor(bundle.IMG[c]).to(device)
        p = torch.from_numpy(bundle.PROP_N[c]).float().to(device)
        zs.append(encoder(x, p).cpu())
    return torch.cat(zs).numpy()
