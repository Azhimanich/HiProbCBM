"""Utilitas reproduksibilitas (Bab IV.6 - pencatatan manifes eksperimen)."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seeds_for_repetitions(base_seed: int, n_repeats: int = 3) -> list[int]:
    """Bab IV.7.3: seluruh eksperimen dijalankan dengan tiga variasi seed."""
    return [base_seed + i for i in range(n_repeats)]
