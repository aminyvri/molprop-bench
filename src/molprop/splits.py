"""Train/validation/test splitting strategies (no RDKit needed: scaffolds are passed in)."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np

Frac = tuple[float, float, float]
DEFAULT_FRAC: Frac = (0.8, 0.1, 0.1)
SPLIT_NAMES = ("random", "scaffold", "scaffold_det")


def _check_frac(frac: Frac) -> None:
    if len(frac) != 3 or any(f <= 0 for f in frac) or abs(sum(frac) - 1.0) > 1e-9:
        raise ValueError(f"frac must be three positive numbers summing to 1, got {frac}")


def _finish(n: int, train, valid, test):
    parts = [np.array(sorted(p), dtype=int) for p in (train, valid, test)]
    joined = np.concatenate(parts)
    if len(joined) != n or len(set(joined.tolist())) != n:
        raise AssertionError("split is not a partition of the dataset")
    return tuple(parts)


def random_split(n: int, seed: int, frac: Frac = DEFAULT_FRAC):
    """Uniform random split."""
    _check_frac(frac)
    perm = np.random.default_rng(seed).permutation(n)
    n_train = int(round(frac[0] * n))
    n_valid = int(round(frac[1] * n))
    return _finish(n, perm[:n_train], perm[n_train : n_train + n_valid], perm[n_train + n_valid :])


def group_by_scaffold(scaffolds: Sequence[str]) -> list[list[int]]:
    """Indices grouped by scaffold, in order of first appearance."""
    groups: dict[str, list[int]] = defaultdict(list)
    for i, scaffold in enumerate(scaffolds):
        groups[scaffold].append(i)
    return list(groups.values())


def _fill(groups_in_order: list[list[int]], n: int, frac: Frac):
    """Assign whole scaffold groups to train, then validation, then test."""
    train_cut = frac[0] * n
    valid_cut = (frac[0] + frac[1]) * n
    train: list[int] = []
    valid: list[int] = []
    test: list[int] = []
    for group in groups_in_order:
        if len(train) + len(group) > train_cut:
            if len(train) + len(valid) + len(group) > valid_cut:
                test.extend(group)
            else:
                valid.extend(group)
        else:
            train.extend(group)
    return train, valid, test


def scaffold_split_deterministic(scaffolds: Sequence[str], frac: Frac = DEFAULT_FRAC):
    """Classic scaffold split: biggest scaffold groups go to train, rare ones to valid/test.

    Fully deterministic, so there is no split-to-split variance to measure.
    """
    _check_frac(frac)
    n = len(scaffolds)
    groups = sorted(group_by_scaffold(scaffolds), key=lambda g: (-len(g), g[0]))
    return _finish(n, *_fill(groups, n, frac))


def scaffold_split_randomized(scaffolds: Sequence[str], seed: int, frac: Frac = DEFAULT_FRAC):
    """Seeded scaffold split ("balanced scaffold split").

    Scaffold groups are never divided between sets, so train and test share no scaffold.
    Groups too large to fit in validation or test are placed first (in train); the remaining
    groups are shuffled, which gives a different split for each seed.
    """
    _check_frac(frac)
    n = len(scaffolds)
    rng = np.random.default_rng(seed)
    groups = group_by_scaffold(scaffolds)
    valid_size, test_size = frac[1] * n, frac[2] * n
    is_big = [len(g) > valid_size / 2 or len(g) > test_size / 2 for g in groups]
    big = [g for g, flag in zip(groups, is_big) if flag]
    small = [g for g, flag in zip(groups, is_big) if not flag]
    big = [big[i] for i in rng.permutation(len(big))]
    small = [small[i] for i in rng.permutation(len(small))]
    return _finish(n, *_fill(big + small, n, frac))


def make_split(name: str, n: int, scaffolds: Sequence[str], seed: int, frac: Frac = DEFAULT_FRAC):
    if name == "random":
        return random_split(n, seed, frac)
    if name == "scaffold":
        return scaffold_split_randomized(scaffolds, seed, frac)
    if name == "scaffold_det":
        return scaffold_split_deterministic(scaffolds, frac)
    raise ValueError(f"Unknown split {name!r}; choose from {SPLIT_NAMES}")
