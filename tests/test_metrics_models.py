import numpy as np
import pytest

from molprop.metrics import classification_metrics, primary_metric, regression_metrics, selection_score
from molprop.models import build_estimator, param_grid, predict


def test_regression_metrics_known_values():
    m = regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 4.0])
    assert m["rmse"] == pytest.approx(np.sqrt(1 / 3))
    assert m["mae"] == pytest.approx(1 / 3)


def test_classification_metrics_and_single_class_guard():
    m = classification_metrics([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    assert m["roc_auc"] == pytest.approx(0.75)
    single = classification_metrics([1, 1, 1], [0.2, 0.3, 0.9])
    assert np.isnan(single["roc_auc"]) and np.isnan(single["pr_auc"])


def test_selection_score_orientation_and_nan():
    assert primary_metric("regression") == ("rmse", True)
    assert primary_metric("classification") == ("roc_auc", False)
    assert selection_score("regression", {"rmse": 0.5}) > selection_score("regression", {"rmse": 1.0})
    assert selection_score("classification", {"roc_auc": 0.9}) > selection_score("classification", {"roc_auc": 0.6})
    assert selection_score("classification", {"roc_auc": float("nan")}) == float("-inf")


@pytest.mark.parametrize("task", ["regression", "classification"])
@pytest.mark.parametrize("model", ["dummy", "linear", "rf"])
@pytest.mark.parametrize("feature_set", ["morgan", "descriptors"])
def test_every_model_fits_and_predicts(task, model, feature_set):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 12))
    X[0, 0] = np.nan  # the pipeline must cope with missing values
    y = X[:, 1] + 0.1 * rng.normal(size=80)
    y = y if task == "regression" else (y > np.nanmedian(y)).astype(int)
    for params in param_grid(task, model):
        est = build_estimator(task, model, params, feature_set, seed=0, n_train=80, n_estimators=10, n_jobs=1)
        est.fit(X, y)
        out = predict(task, est, X)
        assert out.shape == (80,)
        assert np.all(np.isfinite(out))
