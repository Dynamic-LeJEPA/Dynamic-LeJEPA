"""nuScenes pipelines for Phases 1-2 (paper Sec VII-B, VII-C).

Phase 1 (nuScenes-mini): ego-motion e_t = (R, t) estimated by point-cloud
ICP between consecutive LiDAR sweeps.
Phase 2 (Trainval): ego-motion taken from ground-truth ego_pose.json
metadata (bypassing ~3.5 h of ICP with a 1 s lookup); LiDAR is then used
only for sparse-depth supervision of the physics-informed decoder.

Both phases: consecutive camera frames matched to the nearest LiDAR sweep
(< 100 ms), pairs with 0 < dt <= 2 s, scene-level 70/30 split (seed 42).
"""
from __future__ import annotations

import glob
import json
import os
import pickle
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from scipy.linalg import svd
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm

COMPOUND_EXTENSIONS = (".pcd.bin", ".pcd")


def strip_compound_extension(filename: str) -> Tuple[str, str]:
    """Handle compound extensions like .pcd.bin (nuScenes LiDAR blobs)."""
    for cext in COMPOUND_EXTENSIONS:
        if filename.endswith(cext):
            return filename[:-len(cext)], cext.lstrip(".")
    name, ext = os.path.splitext(filename)
    return name, ext.lstrip(".")


class NuScenesParser:
    """Filename-driven parser: works directly from the blob directory layout
    (scene__sensor__timestamp.ext), with no JSON sample tables required."""

    @staticmethod
    def parse_filename(fname: str) -> Optional[dict]:
        base = os.path.basename(fname)
        name, ext = strip_compound_extension(base)
        parts = name.split("__")
        if len(parts) != 3:
            return None
        scene, sensor, ts_str = parts
        try:
            ts = int(ts_str)
        except ValueError:
            return None
        return {"scene": scene, "sensor": sensor, "timestamp": ts,
                "ext": ext, "filepath": fname}

    @staticmethod
    def discover_sensor_files(root_dir: str, sensor_name: str,
                              ext: str = "jpg") -> List[dict]:
        sensor_dir = os.path.join(root_dir, sensor_name)
        if not os.path.isdir(sensor_dir):
            return []
        files = glob.glob(os.path.join(sensor_dir, f"*.{ext}"))
        parsed = [p for f in files if (p := NuScenesParser.parse_filename(f))]
        parsed.sort(key=lambda x: (x["scene"], x["timestamp"]))
        return parsed

    @staticmethod
    def find_nearest_lidar(cam_ts: int, lid_ts: np.ndarray, lid_paths: List[str],
                           max_gap_ns: int = 100_000_000) -> Optional[str]:
        idx = np.searchsorted(lid_ts, cam_ts)
        cands = [(abs(lid_ts[i] - cam_ts), lid_paths[i])
                 for i in (idx - 1, idx, idx + 1)
                 if 0 <= i < len(lid_ts) and abs(lid_ts[i] - cam_ts) < max_gap_ns]
        return min(cands, key=lambda x: x[0])[1] if cands else None

    @staticmethod
    def build_sequences(camera: str = "CAM_FRONT", lidar: str = "LIDAR_TOP",
                        data_root: str = "", use_sweeps: bool = True,
                        max_time_gap_ms: float = 100.0) -> List[dict]:
        """Consecutive-frame pairs with matched LiDAR and 0 < dt <= 2 s."""
        samples_dir = os.path.join(data_root, "samples")
        sweeps_dir = os.path.join(data_root, "sweeps")
        cam = NuScenesParser.discover_sensor_files(samples_dir, camera, "jpg")
        lid = NuScenesParser.discover_sensor_files(samples_dir, lidar, "pcd.bin")
        if use_sweeps and os.path.isdir(sweeps_dir):
            cam += NuScenesParser.discover_sensor_files(sweeps_dir, camera, "jpg")
            lid += NuScenesParser.discover_sensor_files(sweeps_dir, lidar, "pcd.bin")
        print(f"camera files: {len(cam):,} | lidar files: {len(lid):,}")
        if not cam or not lid:
            return []

        lidar_by_scene = defaultdict(lambda: {"ts": [], "paths": []})
        for lf in lid:
            lidar_by_scene[lf["scene"]]["ts"].append(lf["timestamp"])
            lidar_by_scene[lf["scene"]]["paths"].append(lf["filepath"])
        for sc in lidar_by_scene:
            order = np.argsort(lidar_by_scene[sc]["ts"])
            lidar_by_scene[sc]["ts"] = np.array(lidar_by_scene[sc]["ts"])[order]
            lidar_by_scene[sc]["paths"] = [lidar_by_scene[sc]["paths"][i]
                                           for i in order]
        cam_by_scene = defaultdict(list)
        for cf in cam:
            cam_by_scene[cf["scene"]].append(cf)

        max_gap_ns = int(max_time_gap_ms * 1e6)
        sequences, matched, unmatched = [], 0, 0
        for scene, frames in cam_by_scene.items():
            frames.sort(key=lambda x: x["timestamp"])
            if scene not in lidar_by_scene:
                continue
            lts = lidar_by_scene[scene]["ts"]
            lpaths = lidar_by_scene[scene]["paths"]
            matched_frames = []
            for cf in frames:
                lp = NuScenesParser.find_nearest_lidar(cf["timestamp"], lts,
                                                       lpaths, max_gap_ns)
                if lp:
                    matched_frames.append(
                        {"scene": scene, "img_path": cf["filepath"],
                         "lidar_path": lp, "cam_ts": cf["timestamp"]})
                    matched += 1
                else:
                    unmatched += 1
            for i in range(len(matched_frames) - 1):
                a, b = matched_frames[i], matched_frames[i + 1]
                dt = (b["cam_ts"] - a["cam_ts"]) / 1e9
                if 0 < dt <= 2.0:
                    sequences.append(
                        {"scene": scene, "img_t": a["img_path"],
                         "img_t1": b["img_path"], "lidar_t": a["lidar_path"],
                         "lidar_t1": b["lidar_path"],
                         "timestamp_t": a["cam_ts"],
                         "timestamp_t1": b["cam_ts"], "dt": dt})
        print(f"matched: {matched:,} unmatched: {unmatched:,} | "
              f"consecutive pairs: {len(sequences):,}")
        return sequences

    @staticmethod
    def split_by_scene(sequences: List[dict], train_ratio: float = 0.7,
                       seed: int = 42) -> Tuple[List[dict], List[dict]]:
        """Scene-level split (never frame-level): no scene appears in both."""
        scenes = sorted({s["scene"] for s in sequences})
        np.random.seed(seed)
        np.random.shuffle(scenes)
        n_train = int(len(scenes) * train_ratio)
        train_sc, val_sc = set(scenes[:n_train]), set(scenes[n_train:])
        train = [s for s in sequences if s["scene"] in train_sc]
        val = [s for s in sequences if s["scene"] in val_sc]
        print(f"scene split: {len(train_sc)} train scenes / {len(train):,} pairs | "
              f"{len(val_sc)} val scenes / {len(val):,} pairs")
        return train, val


# ============================ LiDAR I/O + ICP ================================

def load_lidar_bin(filepath: str) -> np.ndarray:
    raw = np.fromfile(filepath, dtype=np.float32)
    if len(raw) % 5 != 0:
        raw = raw[: len(raw) - (len(raw) % 5)]
    return raw.reshape(-1, 5)[:, :3]


def farthest_point_sample(points: np.ndarray, n_samples: int) -> np.ndarray:
    N = points.shape[0]
    if N <= n_samples:
        return points
    indices = [np.random.randint(N)]
    distances = np.full(N, np.inf)
    for _ in range(n_samples - 1):
        last = points[indices[-1]]
        distances = np.minimum(distances, ((points - last) ** 2).sum(1))
        indices.append(int(np.argmax(distances)))
    return points[indices]


def _rot_to_axis_angle(R: np.ndarray) -> np.ndarray:
    angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if abs(angle) < 1e-10:
        return np.zeros(3)
    if abs(angle - np.pi) < 1e-10:
        w, v = np.linalg.eig(R)
        axis = np.real(v[:, np.argmin(abs(w - 1))])
        return axis / (np.linalg.norm(axis) + 1e-8) * angle
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return axis / (2 * np.sin(angle) + 1e-8) * angle


def compute_ego_motion_icp(pc1: np.ndarray, pc2: np.ndarray,
                           n_samples: int = 1024, max_iter: int = 30,
                           thresh: float = 2.0) -> np.ndarray:
    """Phase 1: ego-motion e_t = (axis-angle, translation) via LiDAR ICP."""
    src = farthest_point_sample(pc1, n_samples).astype(np.float64)
    tgt = farthest_point_sample(pc2, n_samples).astype(np.float64)
    R_tot, t_tot = np.eye(3), np.zeros(3)
    tree = cKDTree(tgt)
    for _ in range(max_iter):
        src_t = (R_tot @ src.T).T + t_tot
        dists, idx = tree.query(src_t, k=1)
        mask = dists < thresh
        if mask.sum() < 10:
            break
        s_m, t_m = src[mask], tgt[idx[mask]]
        s_c, t_c = s_m.mean(0), t_m.mean(0)
        H = (s_m - s_c).T @ (t_m - t_c)
        U, _, Vt = svd(H)
        R_step = Vt.T @ U.T
        if np.linalg.det(R_step) < 0:
            Vt[-1] *= -1
            R_step = Vt.T @ U.T
        t_step = t_c - R_step @ s_c
        R_tot = R_step @ R_tot
        t_tot = R_step @ t_tot + t_step
    return np.concatenate([_rot_to_axis_angle(R_tot), t_tot]).astype(np.float32)


# ================== Phase 2: ground-truth ego-motion =========================

def load_ego_poses_fast(meta_root: str) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Phase 2: ego poses from ego_pose.json — replaces ~3.5 h of ICP with a
    ~1 s lookup. LiDAR point clouds are then used only for sparse depth."""
    with open(os.path.join(meta_root, "ego_pose.json")) as f:
        data = json.load(f)
    pose_dict = {}
    for item in data:
        pose_dict[item["timestamp"]] = (
            Rot.from_quat(item["rotation"]).as_matrix(),
            np.asarray(item["translation"], dtype=np.float64))
    print(f"loaded {len(pose_dict):,} ground-truth ego-poses")
    return pose_dict


def ego_motion_from_metadata(pose_dict, ts_t: int, ts_t1: int) -> np.ndarray:
    """Returns [aa_x, aa_y, aa_z, tx, ty, tz]; zeros if a sweep is missing."""
    if ts_t not in pose_dict or ts_t1 not in pose_dict:
        return np.zeros(6, dtype=np.float32)
    R1, t1 = pose_dict[ts_t]
    R2, t2 = pose_dict[ts_t1]
    R_rel = R2 @ R1.T
    t_rel = t2 - R_rel @ t1
    return np.concatenate([Rot.from_matrix(R_rel).as_rotvec(),
                           t_rel]).astype(np.float32)


# ================== LiDAR -> sparse depth (decoder supervision) ==============

def project_lidar_to_depth(lidar_xyz: np.ndarray, depth_size: int = 56,
                           max_depth: float = 80.0,
                           min_depth: float = 0.5) -> np.ndarray:
    """LiDAR (x fwd, y left, z up, vehicle frame) -> sparse front-camera
    depth map. ~70 deg HFOV / ~50 deg VFOV approximation."""
    depth_map = np.zeros((depth_size, depth_size), dtype=np.float32)
    if len(lidar_xyz) == 0:
        return depth_map
    x, y, z = lidar_xyz[:, 0], lidar_xyz[:, 1], lidar_xyz[:, 2]
    valid = (x > min_depth) & (x < max_depth)
    x, y, z = x[valid], y[valid], z[valid]
    if len(x) == 0:
        return depth_map
    h = np.arctan2(y, x) / np.radians(35)
    v = np.arctan2(-z, x) / np.radians(25)
    in_fov = (np.abs(h) < 1.0) & (np.abs(v) < 1.0)
    h, v, x = h[in_fov], v[in_fov], x[in_fov]
    if len(x) == 0:
        return depth_map
    u = np.clip(((h + 1) / 2 * depth_size).astype(np.int32), 0, depth_size - 1)
    w = np.clip(((v + 1) / 2 * depth_size).astype(np.int32), 0, depth_size - 1)
    depth_map[w, u] = x
    return depth_map


# ================== Caches + dataset =========================================

def precompute_caches(sequences: List[dict], cache_dir: str,
                      pose_dict: Optional[dict] = None) -> Tuple[dict, dict]:
    """Ego-motion (metadata if available, else ICP) + sparse depth per pair.
    Persisted to pickle and topped up incrementally, so Kaggle/Colab session
    restarts never recompute finished work."""
    os.makedirs(cache_dir, exist_ok=True)
    ego_p = os.path.join(cache_dir, "ego_cache.pkl")
    depth_p = os.path.join(cache_dir, "depth_cache.pkl")
    ego, depth = {}, {}
    if os.path.exists(ego_p):
        with open(ego_p, "rb") as f:
            ego = pickle.load(f)
    if os.path.exists(depth_p):
        with open(depth_p, "rb") as f:
            depth = pickle.load(f)
    unique = {(s["timestamp_t"], s["timestamp_t1"]): s for s in sequences}
    missing = {k: s for k, s in unique.items()
               if k not in ego or k not in depth}
    print(f"pairs required: {len(unique):,} | cached: "
          f"{len(unique) - len(missing):,} | missing: {len(missing):,}")
    if missing:
        for key, s in tqdm(missing.items(), desc="precompute ego+depth"):
            if key not in ego:
                ego[key] = (
                    ego_motion_from_metadata(pose_dict, s["timestamp_t"],
                                             s["timestamp_t1"])
                    if pose_dict is not None
                    else compute_ego_motion_icp(load_lidar_bin(s["lidar_t"]),
                                                load_lidar_bin(s["lidar_t1"])))
            if key not in depth:
                depth[key] = project_lidar_to_depth(load_lidar_bin(s["lidar_t"]))
        with open(ego_p, "wb") as f:
            pickle.dump(ego, f)
        with open(depth_p, "wb") as f:
            pickle.dump(depth, f)
    return ego, depth


class NuScenesPairsDataset(Dataset):
    """Consecutive-frame pairs with cached ego-motion and sparse depth.

    Images are decoded ONCE into a shared uint8 RAM cache: the 3 modes x 10
    seeds x 30 epochs ablation re-visits every image ~900 times, and each
    revisit then costs a cheap normalize instead of a full JPEG decode."""

    _IMG_CACHE: Dict[str, torch.Tensor] = {}

    def __init__(self, sequences: List[dict], img_size: int = 224,
                 ego_cache: Optional[dict] = None,
                 depth_cache: Optional[dict] = None):
        self.sequences = sequences
        self.ego_cache = ego_cache or {}
        self.depth_cache = depth_cache or {}
        self.resize = transforms.Resize((img_size, img_size))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.sequences)

    def _load_image_tensor(self, path: str) -> torch.Tensor:
        cached = NuScenesPairsDataset._IMG_CACHE.get(path)
        if cached is None:
            img = self.resize(Image.open(path).convert("RGB"))
            cached = torch.from_numpy(
                np.array(img, dtype=np.uint8)).permute(2, 0, 1).contiguous()
            NuScenesPairsDataset._IMG_CACHE[path] = cached
        return self.normalize(cached.float() / 255.0)

    def __getitem__(self, idx):
        s = self.sequences[idx]
        key = (s["timestamp_t"], s["timestamp_t1"])
        return {"img_t": self._load_image_tensor(s["img_t"]),
                "img_t1": self._load_image_tensor(s["img_t1"]),
                "ego_motion": torch.from_numpy(self.ego_cache[key]).float(),
                "depth_sparse": torch.from_numpy(self.depth_cache[key]).float(),
                "dt": torch.tensor(s["dt"], dtype=torch.float32),
                "scene": s["scene"], "key": key}
