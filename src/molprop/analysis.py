"""How similar are test molecules to the training set under each split?"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


def nn_similarity(
    fps: Sequence,
    train_idx: Sequence[int],
    test_idx: Sequence[int],
    bulk: Callable | None = None,
) -> np.ndarray:
    """Tanimoto similarity of each test molecule to its nearest training molecule.

    `bulk(fp, list_of_fps)` must return similarities; it defaults to RDKit's implementation.
    """
    if bulk is None:
        from .chem import bulk_tanimoto as bulk
    train_fps = [fps[i] for i in train_idx]
    return np.array([max(bulk(fps[i], train_fps)) for i in test_idx], dtype=float)


def summarize_similarity(sims: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(sims)),
        "median": float(np.median(sims)),
        "p25": float(np.percentile(sims, 25)),
        "p75": float(np.percentile(sims, 75)),
    }
