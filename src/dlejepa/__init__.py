"""Dynamic LeJEPA: maximum-entropy world models with provable physics placement.

Encoder maximum entropy (SIGReg) | predictor dynamics | decoder physics.
Theorems IV.1, IV.4, IV.7, IV.10, IV.13; Proposition VI.1.
"""
__version__ = "1.0.0"

from .seed import seed_everything
from .metrics import (
    compute_effective_dim,
    compute_entropy_ratio,
    compute_effective_dim_from_embeddings,
)
from .sigreg import SIGReg, SIGRegPlus
from .stats import paired_wilcoxon, holm_bonferroni
