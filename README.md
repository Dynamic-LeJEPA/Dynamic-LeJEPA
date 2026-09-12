# Dynamic LeJEPA
![Simulating Humanoid Planning](https://img.shields.io/badge/Research-Simulating_Humanoid_Planning-orange)


### Dynamic LeJEPA: Maximum Entropy Representations for Sequential Prediction and Latent Planning


License: MITPython 3.9+CI

#### The design rule: Encoder maximum entropy · Predictor dynamics · Decoder physics — never the encoder.

### 📖 Overview

This repository contains the official implementation of Dynamic LeJEPA, a theoretically grounded framework for Joint-Embedding Predictive Architectures (JEPAs) in sequential domains like autonomous driving and robotic manipulation. The work resolves a critical paradox: while injecting domain knowledge (physics, kinematics, geometry) into JEPAs consistently degrades performance, we prove through six theorems that the isotropic maximum-entropy embedding is symmetry-stable under sequential prediction losses, and that physics constraints are provably benign on the observation decoder but destructive on the encoder.

🧩 Key Features

Theoretical Guarantees: Six theorems proving that:
Prediction losses do not alter the optimal maximum-entropy embedding distribution (Theorem IV.1)
Physics constraints belong in the decoder, not the encoder (Theorem IV.10/Corollary IV.11)
Sample complexity requirements for reliable distributional validation (Proposition VI.1)
Validated Design Principle: Encoder maximum-entropy · Predictor dynamics · Decoder physics
Cross-Domain Validation: Validated on both autonomous driving (nuScenes) and robotic manipulation (MimicGen)
Robust Experimental Protocol: 110 seeded runs across 2 domains and 2 encoder mechanisms with paired Wilcoxon significance (Holm-corrected p=0.0078)
🛠️ Installation
Prerequisites
Python 3.8+
PyTorch 2.0+
CUDA (recommended)

Setup

Clone the repositorygit clone https://github.com/Dynamic-LeJEPA/Dynamic-LeJEPA.gitcd Dynamic-LeJEPA# Install dependenciespip install -e .[dev] - Run tests to verify installationpytest -q
Data Setup
-For dataset preparation and download instructions, refer to docs/DATA.md. The primary datasets are:

nuScenes (autonomous driving)
MimicGen (bimanual robotic manipulation)
🚀 Quick Start
Phase 1: Debugging & Metric Characterization

### Installation

git clone https://github.com/Dynamic-LeJEPA/Dynamic-LeJEPA.gitcd dynamic-lejepapip 

install -e .[dev]pytest -q        

### verifies the Prop VI.1 measurement floor & SIGReg+ blind-spot fix

Data setup (nuScenes-mini, nuScenes Trainval Part 1, MimicGen two_arm_threading):see docs/DATA.md.

### Quick start
### Phase 3 primary ablation: 3 modes × 10 seeds, 15 epochs, MimicGen threadingpython experiments/phase3_mimicgen/run_ablation.py \    
--hdf5 /path/to/two_arm_threading.hdf5 --epochs 15 --out results/phase3
### Phase 3 budget replication (Sec VII-D6: violation deepens +49%)python experiments/phase3_mimicgen/run_ablation.py \    
--hdf5 /path/to/two_arm_threading.hdf5 --epochs 30 --out results/phase3_budget
#### Latent CEM planning + model-exploitation diagnostic (Sec VII-D8)python experiments/phase3_mimicgen/run_cem_planning.py \    
--hdf5 /path/to/two_arm_threading.hdf5 --ckpt-dir checkpoints/phase3
### Phase 2 ablation: nuScenes Trainval (needs metadata ego-poses, see docs/DATA.md)python experiments/phase2_nuscenes_trainval/run_ablation.py \    
--data-root /path/to/v1.0-trainval01_blobs \    
--meta-root /path/to/v1.0-trainval
### Phase 1 control experiment (Table IV: the d_eff measurement floor)python experiments/phase1_nuscenes_mini/run_control_experiment.py
Approximate cost on a Tesla T4 (the paper's environment): Phase 1 ≈ 1.5 GPU-h;Phase 2 ≈ 45 GPU-h (30 runs); Phase 3 (15 ep) ≈ 45 GPU-h; Phase 3 (30 ep) ≈ 90 GPU-h.All runners auto-resume across sessions (results are checkpointed per seed).

### Key results
Phase 2 — nuScenes Trainval (N/K = 55.5, 10 seeds/condition, covariance d_eff;all Wilcoxon p ≤ 0.002):

Metric	no physics	decoder physics	encoder physics
d_eff	0.9858 ± .004	0.9678 ± .005	0.9394 ± .008
H-ratio	0.9637	0.8165	0.8225
scale ratio	0.9086	0.6045	0.6241
Scale-vs-structure decomposition: both physics conditions lose ~equal scale, but theencoder condition's Δ_structure = 0.028 (2.6× the decoder gap) — the genuine,non-proportional eigen-spectrum distortion predicted by Cor IV.11.

Phase 3 — MimicGen two_arm_threading (N/K = 78, 10 seeds, Holm-corrected p = 0.0059):

Metric	no physics	decoder physics	encoder physics
d_eff	0.1846 ± .004	0.1740 ± .003	0.1380 ± .004
prediction R²	0.784	0.864	0.966
probe R² (proprio)	0.937	0.961	0.984
The two-sided signature of Cor IV.11: encoder physics simultaneously destroys entropyand inflates predictability — the encoder is dragged toward the ≤14-D image of the actionspace. Budget replication (30 ep): encoder gap grows 0.047 → 0.070 (+49%) while allscale/H-ratio effects vanish — the violation is budget-monotone and purely structural.

Phase 3 discovery — correlated collapse & SIGReg⁺: on the ~10-D task manifold,per-dimension variance matching alone yields perfect marginals (scale = 1.0, H-ratio = 1.0)with a rank-6/256 joint covariance (d_eff = 0.022) — a failure mode invisible onhigh-dimensional nuScenes imagery. SIGReg+ (covariance off-diagonal penalty, densercorrelation sampling) restores joint structure (d_eff → 0.18). Included insrc/dlejepa/sigreg.py.

Phase 1 — the measurement floor (Prop VI.1): at N/K = 1.58, d_eff ≈ 0.004 for both themodel and a true N(0, I₂₅₆) control (Table IV) — low d_eff in low-data regimes reflectssample starvation, not embedding collapse. This motivates the N/K ≥ 5 reliability threshold.

⚠️ A cautionary result: offline latent planning ≠ control competence
A latent CEM planner (H = 8, population 256, 4 iterations, elite 32, 200 anchors) built on thefrozen Phase 3 models outperforms real-action replay on model-internal cost in all threeablation conditions — impossible for genuine planning, since replaying the recorded actionsis the ground-truth solution. The model-exploitation diagnostic traces this to the plannerfinding action sequences ~26% of the action range away from the true trajectory while stilllowering the learned decoder's cost. Reproduce with run_cem_planning.py. Closed-loop,simulator-verified execution is the necessary next test (paper Sec VII-D8).

Usage note: two d_eff estimators (Remark VI.3)
dlejepa.metrics implements both estimators from the paper. The covariance estimator(compute_effective_dim_from_embeddings) is used for all final ablation comparisons and issubject to the N < K floor. The diagonal estimator (compute_effective_dim) is only forfast sample-complexity sweeps. Do not compare them numerically.


- Reproducibility & provenance

Demo-level splits fixed at seed 42 → paired Wilcoxon valid; 8 bit-identical seed-levelreproductions verified across sessions.
The exact Kaggle notebooks executed for the paper are preserved under notebooks/.
Committed aggregate results live under results/; per-seed PKLs ship with releases.


