"""Phase configurations (paper Sec VII). Values are the exact published ones."""
import torch
from dataclasses import dataclass, field
from typing import List


@dataclass
class Phase3Config:
    # data
    primary_task: str = "two_arm_threading"
    camera_key: str = "agentview_image"
    prop_keys: List[str] = field(default_factory=lambda: [
        "robot0_joint_pos", "robot1_joint_pos",          # 14
        "robot0_eef_pos", "robot1_eef_pos",              # 6  <- EEF supervision
        "robot0_eef_quat", "robot1_eef_quat",            # 8
        "robot0_gripper_qpos", "robot1_gripper_qpos"])   # 4
    gripper_act_idx: int = 6
    action_dim: int = 14
    # latent space
    K: int = 256
    sigma: float = 1.0
    # architecture
    img_size: int = 84
    patch: int = 14
    enc_dim: int = 192
    enc_depth: int = 6
    enc_heads: int = 3
    pred_hidden: int = 256
    pred_layers: int = 3
    dec_hidden: int = 256
    # loss weights
    sigreg_weight: float = 20.0            # pre-registered from sweep
    sigreg_num_projections: int = 256
    prediction_weight: float = 1.0
    recon_weight: float = 1.0
    physics_weight: float = 10.0
    encoder_physics_weight: float = 10.0
    # training
    batch_size: int = 64
    num_epochs: int = 15                   # 30 for budget replication
    steps_per_epoch: int = 150             # 15 x 150 x 64 = 144k samples
    lr_encoder: float = 1e-4
    lr_predictor: float = 3e-4
    lr_decoder: float = 3e-4
    weight_decay: float = 1e-5
    num_workers: int = 2
    # ablation protocol
    num_seeds: int = 10
    ablation_modes: List[str] = field(default_factory=lambda: [
        "decoder_physics", "encoder_physics", "no_physics"])
    # splits / evaluation
    train_demo_ratio: float = 0.7
    split_seed: int = 42                   # FIXED -> paired Wilcoxon valid
    train_triples_cap: int = 12000
    eval_frames: int = 20000               # N/K = 78
    val_triples_eval: int = 4000
    probe_train_n: int = 8000
    linear_train_triples: int = 4000
    rollout_h: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    rollout_starts: int = 256
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Phase2Config:
    data_root: str = ""                    # v1.0-trainval01_blobs
    meta_root: str = ""                    # v1.0-trainval metadata (ego_pose.json)
    primary_camera: str = "CAM_FRONT"
    lidar_sensor: str = "LIDAR_TOP"
    use_sweeps: bool = True
    img_size: int = 224
    K: int = 256
    K_ego: int = 64
    K_static: int = 192
    sigma: float = 1.0
    depth_size: int = 56
    edge_thresh: float = 0.1
    # FIXED weights (paper Sec VII-C1): SIGReg 100 -> 1, physics 0.5 -> 10
    sigreg_weight: float = 1.0
    sigreg_num_projections: int = 256
    physics_weight: float = 10.0
    encoder_physics_weight: float = 10.0
    prediction_weight: float = 1.0
    batch_size: int = 32
    num_epochs: int = 30
    lr_encoder: float = 1e-4
    lr_predictor: float = 3e-4
    lr_decoder: float = 3e-4
    weight_decay: float = 1e-5
    num_seeds: int = 10
    train_scene_ratio: float = 0.7
    split_seed: int = 42
    eval_samples: int = 5000               # optional subsample cap (None = all)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
