"""Run every (split, seed, features, model) combination on one dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .metrics import compute_metrics, selection_score
from .models import FEATURE_SETS, MODELS, build_estimator, param_grid, predict
from .splits import DEFAULT_FRAC, Frac, make_split


@dataclass
class RunConfig:
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    splits: tuple[str, ...] = ("random", "scaffold")
    features: tuple[str, ...] = FEATURE_SETS
    models: tuple[str, ...] = MODELS
    n_estimators: int = 300
    n_jobs: int = -1
    frac: Frac = field(default=DEFAULT_FRAC)


def _combos(cfg: RunConfig) -> list[tuple[str, str]]:
    """(feature set, model) pairs. The dummy baseline ignores features, so it is run once."""
    combos = [("-", "dummy")] if "dummy" in cfg.models else []
    combos += [(f, m) for f in cfg.features for m in cfg.models if m != "dummy"]
    return combos


def _fit_select_evaluate(task, model, feature_set, X, y, train, valid, test, seed, cfg):
    """Fit every grid candidate on train, keep the best on validation, score it on test."""
    best = None
    for params in param_grid(task, model):
        est = build_estimator(
            task, model, params, feature_set, seed, n_train=len(train),
            n_estimators=cfg.n_estimators, n_jobs=cfg.n_jobs,
        )
        est.fit(X[train], y[train])
        valid_metrics = compute_metrics(task, y[valid], predict(task, est, X[valid]))
        score = selection_score(task, valid_metrics)
        if best is None or score > best[0]:
            best = (score, est, params, valid_metrics)
    _, est, params, valid_metrics = best
    test_metrics = compute_metrics(task, y[test], predict(task, est, X[test]))
    return params, valid_metrics, test_metrics


def run_dataset(
    name: str,
    task: str,
    y: Sequence[float],
    scaffolds: Sequence[str],
    features: Mapping[str, np.ndarray],
    cfg: RunConfig,
    verbose: bool = True,
) -> pd.DataFrame:
    """Return one row per (split, seed, feature set, model) with validation and test metrics."""
    y = np.asarray(y)
    n = len(y)
    rows = []
    for seed in cfg.seeds:
        for split_name in cfg.splits:
            train, valid, test = make_split(split_name, n, scaffolds, seed, cfg.frac)
            for feature_set, model in _combos(cfg):
                X = features[next(iter(features))] if feature_set == "-" else features[feature_set]
                params, vm, tm = _fit_select_evaluate(
                    task, model, feature_set, X, y, train, valid, test, seed, cfg
                )
                row = {
                    "dataset": name,
                    "task": task,
                    "split": split_name,
                    "seed": seed,
                    "feature": feature_set,
                    "model": model,
                    "params": json.dumps(params),
                    "n_train": len(train),
                    "n_valid": len(valid),
                    "n_test": len(test),
                }
                row.update({f"valid_{k}": v for k, v in vm.items()})
                row.update({f"test_{k}": v for k, v in tm.items()})
                rows.append(row)
                if verbose:
                    headline = next(iter(tm))
                    print(
                        f"[{name}] seed={seed} split={split_name:<12} {feature_set:<11} {model:<7} "
                        f"test_{headline}={tm[headline]:.3f}",
                        flush=True,
                    )
    return pd.DataFrame(rows)
