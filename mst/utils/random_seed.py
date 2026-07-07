"""Reproducible RNG for training and evaluation."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_random_seed(
    seed: int = 42,
    *,
    workers: bool = True,
    cudnn_deterministic: bool = False,
) -> int:
    """Seed Python, NumPy, PyTorch, and (optionally) DataLoader worker processes.

    By default only fixes RNG for init + data order (``cudnn_deterministic=False``).
    Set ``cudnn_deterministic=True`` for stricter (slower) cuDNN reproducibility.
    """
    import pytorch_lightning as pl

    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    pl.seed_everything(seed, workers=workers)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(cudnn_deterministic)
    torch.backends.cudnn.benchmark = not bool(cudnn_deterministic)
    return seed
