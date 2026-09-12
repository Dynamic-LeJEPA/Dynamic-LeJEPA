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

     # Control experiment on nuScenes-mini
     python experiments/phase1_nuscenes_mini/run_control_experiment.py

Phase 2: Full Theorem Validation (nuScenes Trainval)

    python experiments/phase2_nuscenes_trainval/run_ablation.py \
    --data-root /path/to/v1.0-trainval01_blobs \
    --meta-root /path/to/v1.0-trainval

Phase 3: Cross-Domain Transfer (MimicGen)

    # Primary ablation (15 epochs)
    python experiments/phase3_mimicgen/run_ablation.py \
    --hdf5 /path/to/two_arm_threading.hdf5 --epochs 15 --out results/phase3

    # Budget replication (30 epochs)
    python experiments/phase3_mimicgen/run_ablation.py \
    --hdf5 /path/to/two_arm_threading.hdf5 --epochs 30 --out results/phase3_budget

    # Latent CEM planning + model-exploitation diagnostic
    python experiments/phase3_mimicgen/run_cem_planning.py \
    --hdf5 /path/to/two_arm_threading.hdf5 --ckpt-dir checkpoints/phase3

🧪 Experimental Protocol

The paper employs a three-phase experimental design:

flowchart LR
    A[Phase 1: Debugging & Metric Characterization] --> B[Phase 2: Full Theorem Validation]
    B --> C[Phase 3: Cross-Domain Transfer]
    C --> D[Model-Exploitation Diagnostic]
    
    subgraph A [Phase 1]
        A1[nuScenes-mini<br>N/K=1.58]
        A2[deff floor validation<br>Table V]
        A3[Depth decoding]
    end
    
    subgraph B [Phase 2]
        B1[nuScenes Trainval<br>N/K=55.5]
        B2[Scale-vs-structure decomposition]
        B3[Paired Wilcoxon tests]
    end
    
    subgraph C [Phase 3]
        C1[MimicGen threading<br>N/K=78]
        C2[Correlated collapse discovery]
        C3[Budget replication]
    end
