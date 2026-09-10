# Dynamic LeJEPA
https://img.shields.io/badge/Research-Simulating Humanoid Planning-blue

### Maximum-Entropy Representations for Sequential Prediction and Latent Planning with Theoretical Guarantees

License: MITPython 3.9+CI

#### The design rule: Encoder maximum entropy · Predictor dynamics · Decoder physics — never the encoder.

### What this is

Joint-Embedding Predictive Architectures (JEPAs) are the emerging backbone of latent worldmodels, yet injecting domain knowledge (physics, kinematics, geometry) into them consistentlydegrades performance, with no theoretical explanation. This repository accompanies thepaper "Dynamic LeJEPA", which resolves that paradox with six theorems:

Prediction–entropy separation (Thm IV.1): sequential prediction losses provably do notalter the optimal maximum-entropy embedding distribution.

Physics placement (Thm IV.10 / Cor IV.11): physics constraints are benign on theobservation decoder and destructive on the encoder — explaining why priorphysics-informed JEPA attempts failed.

Sample complexity (Prop VI.1 / Cor VI.2): distributional validation of the theoremsrequires N/K ≥ 5; below that, effective-dimensionality metrics hit a rank floor andreport spurious "collapse".

The theory is validated by a 3-phase, 60-seeded-run protocol across autonomous driving(nuScenes) and bimanual robotic manipulation (MimicGen), with paired Wilcoxon significancein every comparison and a budget replication showing the encoder-physics violation deepenswith training.

The placement principle (Theorem V.1)

Component	Constraint	Max entropy preserved?	Valid?

Encoder f_θ	none (SIGReg only)	Yes	✅

Predictor g_φ	prediction target	N/A	✅

Decoder h_ψ	physics C(ŷ)=0	N/A	✅

Encoder + physics	C(z)=0	No (Cor IV.11)	❌

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


