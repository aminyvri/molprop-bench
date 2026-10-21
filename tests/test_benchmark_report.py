import numpy as np
import pandas as pd

from molprop.analysis import nn_similarity, summarize_similarity
from molprop.benchmark import RunConfig, run_dataset
from molprop.report import build_report, summarize_dataset


def synthetic(task, n=240, seed=0):
    rng = np.random.default_rng(seed)
    X1 = rng.normal(size=(n, 20))
    X2 = rng.normal(size=(n, 10))
    signal = X1[:, 0] - 0.5 * X1[:, 1]
    y = signal + 0.3 * rng.normal(size=n)
    if task == "classification":
        y = (y > np.median(y)).astype(int)
    scaffolds = [f"S{i // 3}" for i in range(n)]
    return y, scaffolds, {"morgan": X1, "descriptors": X2}


def small_cfg():
    return RunConfig(seeds=(0, 1), splits=("random", "scaffold"), n_estimators=10, n_jobs=1)


def test_run_dataset_produces_expected_rows_and_columns():
    for task, metric in (("regression", "test_rmse"), ("classification", "test_roc_auc")):
        y, scaffolds, feats = synthetic(task)
        res = run_dataset("toy", task, y, scaffolds, feats, small_cfg(), verbose=False)
        # per seed and split: 1 dummy + 2 feature sets x 2 models
        assert len(res) == 2 * 2 * (1 + 2 * 2)
        assert {"dataset", "split", "seed", "feature", "model", "params", metric}.issubset(res.columns)
        assert res[metric].notna().all()
        assert (res["n_train"] + res["n_valid"] + res["n_test"] == len(y)).all()
        # the real models should beat the dummy baseline on a task that has signal
        mean = res.groupby("model")[metric].mean()
        if task == "regression":
            assert mean["rf"] < mean["dummy"]
        else:
            assert mean["rf"] > mean["dummy"]


def test_summarize_and_build_report(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for task in ("regression", "classification"):
        y, scaffolds, feats = synthetic(task)
        res = run_dataset(f"toy_{task}", task, y, scaffolds, feats, small_cfg(), verbose=False)
        res.to_csv(raw / f"toy_{task}.csv", index=False)

    table, splits = summarize_dataset(pd.read_csv(raw / "toy_regression.csv"), "regression")
    assert splits == ["random", "scaffold"]
    assert {"method", "delta", "rank_random", "rank_scaffold"}.issubset(table.columns)
    assert table["rank_random"].min() == 1

    out = build_report(tmp_path)
    text = out.read_text()
    assert "toy_regression" in text and "toy_classification" in text
    assert (tmp_path / "figures" / "toy_regression_primary_metric.png").exists()


def test_nn_similarity_with_custom_similarity_function():
    fps = [0.0, 1.0, 2.0, 10.0]
    bulk = lambda fp, others: [1.0 / (1.0 + abs(fp - o)) for o in others]
    sims = nn_similarity(fps, train_idx=[0, 1], test_idx=[2, 3], bulk=bulk)
    assert sims[0] == 0.5 and sims[1] < sims[0]
    summary = summarize_similarity(sims)
    assert set(summary) == {"mean", "median", "p25", "p75"}
