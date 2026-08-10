import random

import numpy as np
import torch


def seed_everything(seed):
    """Seed Python, NumPy, and PyTorch once for one reproducible run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def idx_to_mask_tensor(indices, length):
    mask = torch.zeros(length, dtype=torch.bool)
    mask[indices] = True
    return mask


def mask_tensor_to_idx(mask):
    return mask.nonzero(as_tuple=False).view(-1).tolist()
