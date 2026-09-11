"""Phase configurations (paper Sec VII). Values are the exact published ones.

VAE comparison arms (paper Secs. VII-C5, VII-D3) add a second encoder
mechanism per phase. Two DISTINCT operating points are recorded here and
must not be conflated:

  * Phase 2 (nuScenes):  vae_kl_weight = 0.01 — LEGACY WEAK-PRIOR REGIME.
    The KL term is effectively dormant (~0.003 of a ~136 total loss), so
    only the within-arm placement *ordering* is interpreted (paper caveat).
  * Phase 3 (MimicGen):  vae_kl_weight = 5.0  — CALIBRATED, pre-run sweep
    (experiments/phase3_mimicgen/calibrate_vae_kl.py). Primary
    encoder-mechanism test; includes the physics-free vae_no_physics
    reference condition.

Changing either weight invalidates the published numbers — re-run the
calibration sweep and the corresponding ablation first.
"""
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
    # ---- VAE arm (encoder-mechanism comparison, paper Sec. VII-D3) ----
    # Protocol mirrors the SIGReg arm EXACTLY (num_epochs, steps_per_epoch,
    # splits, physics weights) — only the encoder mechanism and its
    # distribution term change (marginal-only KL replaces SIGReg).
    # The marginal-only KL is DELIBERATE: it instantiates the hypothesis
    # that correlated collapse is a property of marginal-only regularizers
    # on low-intrinsic-dimension data (paper Sec. VII-D3).
    vae_kl_weight: float = 5.0             # CALIBRATED pre-run; sweep in
                                           #   experiments/phase3_mimicgen/
                                           #   calibrate_vae_kl.py
                                           #   (candidates 0.5-10.0; kl*w ~2.1
                                           #   vs pred+rec ~2.8, phys*10 ~7.6)
    vae_logvar_clamp: float = 10.0         # numerical safety, standard VAE practice
    vae_plus_cov_weight: float = 0.0       # 0 = vanilla marginal-only VAE (the
                                           #   hypothesis under test); >0 enables
                                           #   the VAE+ stretch (SIGReg+ covariance
                                           #   penalty), released for future work
    include_vae_no_physics: bool = True    # physics-free reference condition —
                                           #   required for the clean scale-vs-
                                           #   structure claim within the arm
    vae_num_seeds: int = 10                # paired with the SIGReg arm's seeds
    vae_eval_deterministic: bool = True    # eval + planning use the posterior
                                           #   mean mu (VAEDeterministic shim):
                                           #   replay floor is not noise-inflated
                                           #   and the covariance estimator
                                           #   (Remark VI.3) applies across arms
    vae_kl_calibrated: bool = True         # False forces drivers to run
                                           #   calibrate_vae_kl.py before the
                                           #   ablation (notebook
                                           #   _VAE_KL_CALIBRATED guard)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def vae_modes(self) -> List[str]:
        """VAE arm conditions (published protocol: 3 conditions x 10 seeds)."""
        modes = ["vae_decoder_physics", "vae_encoder_physics"]
        if self.include_vae_no_physics:
            modes.append("vae_no_physics")
        return modes


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
    # ---- VAE arm (encoder-mechanism comparison, paper Sec. VII-C5) ----
    # !! LEGACY WEAK-PRIOR REGIME — do NOT "fix" this value. The published
    # Phase 2 VAE arm ran at KL weight 0.01, which is effectively DORMANT
    # (~0.003 of a ~136 total loss): the encoder receives almost no
    # distribution-matching gradient. Only the within-arm placement
    # *ordering* (decoder > encoder, paired Wilcoxon p = 0.0020) is
    # interpreted from this arm — the paper's regime caveat
    # (Sec. VII-C5 / Discussion limitations) depends on this value staying
    # exactly as published. The calibrated arm is Phase3Config
    # (vae_kl_weight = 5.0); changing this number without re-running the
    # ablation AND the paper text is a reproducibility break.
    vae_kl_weight: float = 0.01
    vae_logvar_clamp: float = 10.0
    vae_plus_cov_weight: float = 0.0       # unused by the published Phase 2 arm
                                           #   (present for a shared trainer path)
    vae_num_seeds: int = 10
    vae_eval_deterministic: bool = False   # the published Phase 2 arm evaluated
                                           #   on SAMPLED latents (the
                                           #   deterministic-mu shim was
                                           #   introduced in Phase 3); keep False
                                           #   to reproduce the published numbers
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def vae_modes(self) -> List[str]:
        """Published Phase 2 VAE arm: two conditions only — no physics-free
        reference was run in this phase (see paper Sec. VII-C5 caveats;
        the clean scale-vs-structure reference exists only in Phase 3)."""
        return ["vae_decoder_physics", "vae_encoder_physics"]
