"""Evaluation metrics and the rule used to pick hyperparameters."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def classification_metrics(y_true, y_score) -> dict[str, float]:
    """ROC-AUC and PR-AUC; both are NaN when the evaluation set contains a single class."""
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return {"roc_auc": float("nan"), "pr_auc": float("nan")}
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
    }


def compute_metrics(task: str, y_true, y_pred) -> dict[str, float]:
    if task == "regression":
        return regression_metrics(y_true, y_pred)
    if task == "classification":
        return classification_metrics(y_true, y_pred)
    raise ValueError(f"Unknown task {task!r}")


def primary_metric(task: str) -> tuple[str, bool]:
    """Name of the headline metric and whether lower is better."""
    return ("rmse", True) if task == "regression" else ("roc_auc", False)


def selection_score(task: str, metrics: dict[str, float]) -> float:
    """Larger is better; NaN scores lose against any real score."""
    name, lower_is_better = primary_metric(task)
    value = metrics[name]
    if np.isnan(value):
        return float("-inf")
    return -value if lower_is_better else value
