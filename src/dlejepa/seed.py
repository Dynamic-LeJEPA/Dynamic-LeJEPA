import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Full determinism. The paper verified 8 bit-identical seed-level
    reproductions across sessions with this routine."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
