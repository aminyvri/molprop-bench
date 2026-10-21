"""Estimators and their (deliberately small) hyperparameter grids."""

from __future__ import annotations

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler

FEATURE_SETS = ("morgan", "descriptors")
MODELS = ("dummy", "linear", "rf")


def param_grid(task: str, model: str) -> list[dict]:
    """Candidate settings; the best one is chosen on the validation split only."""
    if model == "dummy":
        return [{}]
    if model == "rf":
        return [{"max_features": "sqrt"}, {"max_features": 0.2}]
    if model == "linear":
        if task == "regression":
            return [{"alpha": a} for a in (0.1, 1.0, 10.0, 100.0)]
        return [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)]
    raise ValueError(f"Unknown model {model!r}")


def _scaler(feature_set: str, n_train: int):
    if feature_set == "descriptors":
        # RDKit descriptors have very heavy tails; map each one to a normal distribution
        return QuantileTransformer(
            n_quantiles=max(10, min(200, n_train)), output_distribution="normal", random_state=0
        )
    return StandardScaler()


def build_estimator(
    task: str,
    model: str,
    params: dict,
    feature_set: str,
    seed: int,
    n_train: int,
    n_estimators: int = 300,
    n_jobs: int = -1,
) -> Pipeline:
    """Median imputation (fit on train only) followed by the model."""
    steps: list = [("impute", SimpleImputer(strategy="median"))]
    if model == "dummy":
        est = DummyRegressor(strategy="mean") if task == "regression" else DummyClassifier(strategy="prior")
    elif model == "linear":
        steps.append(("scale", _scaler(feature_set, n_train)))
        if task == "regression":
            est = Ridge(alpha=params["alpha"])
        else:
            est = LogisticRegression(C=params["C"], max_iter=2000)
    elif model == "rf":
        cls = RandomForestRegressor if task == "regression" else RandomForestClassifier
        est = cls(
            n_estimators=n_estimators,
            max_features=params["max_features"],
            n_jobs=n_jobs,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown model {model!r}")
    steps.append(("model", est))
    return Pipeline(steps)


def predict(task: str, fitted: Pipeline, X):
    """Predicted value (regression) or probability of the positive class (classification)."""
    if task == "regression":
        return fitted.predict(X)
    return fitted.predict_proba(X)[:, 1]
