"""Aggregate raw results into tables (Markdown) and figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .datasets import DATASETS  # noqa: E402
from .metrics import primary_metric  # noqa: E402

SPLIT_ORDER = ("random", "scaffold", "scaffold_det")


def _read_dir(folder: Path, pattern: str) -> pd.DataFrame:
    files = sorted(folder.glob(pattern))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def load_results(results_dir: Path) -> pd.DataFrame:
    df = _read_dir(Path(results_dir) / "raw", "*.csv")
    if df.empty:
        raise FileNotFoundError(f"No result files in {Path(results_dir) / 'raw'}. Run `molprop run` first.")
    return df


def method_label(feature: str, model: str, task: str) -> str:
    if model == "dummy":
        return "baseline (mean / class prior)"
    name = {"rf": "random forest", "linear": "ridge" if task == "regression" else "logistic regression"}[model]
    feat = {"morgan": "Morgan counts", "descriptors": "RDKit descriptors"}.get(feature, feature)
    return f"{name} + {feat}"


def summarize_dataset(df: pd.DataFrame, task: str) -> tuple[pd.DataFrame, list[str]]:
    """Mean and std of the headline test metric across seeds, per method and split."""
    metric, lower_is_better = primary_metric(task)
    col = f"test_{metric}"
    agg = df.groupby(["feature", "model", "split"])[col].agg(["mean", "std"]).reset_index()
    splits = [s for s in SPLIT_ORDER if s in set(agg["split"])]

    rows = []
    for (feature, model), sub in agg.groupby(["feature", "model"]):
        row = {"method": method_label(feature, model, task), "model": model}
        for s in splits:
            r = sub[sub["split"] == s]
            row[f"{s}_mean"] = float(r["mean"].iloc[0]) if len(r) else np.nan
            row[f"{s}_std"] = float(r["std"].iloc[0]) if len(r) else np.nan
        rows.append(row)
    res = pd.DataFrame(rows)
    if {"random", "scaffold"} <= set(splits):
        res["delta"] = res["scaffold_mean"] - res["random_mean"]
    for s in splits:
        res[f"rank_{s}"] = res[f"{s}_mean"].rank(ascending=lower_is_better, method="min")
    sort_col = f"rank_{splits[0]}"
    res = res.sort_values(sort_col).reset_index(drop=True)
    return res, splits


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _pm(mean: float, std: float) -> str:
    if np.isnan(mean):
        return "n/a"
    return f"{mean:.3f}" if np.isnan(std) else f"{mean:.3f} ± {std:.3f}"


def dataset_table(res: pd.DataFrame, splits: list[str]) -> str:
    headers = ["method"] + [f"{s}" for s in splits]
    has_delta = "delta" in res
    if has_delta:
        headers.append("Δ (scaffold − random)")
    headers += [f"rank ({s})" for s in splits]
    rows = []
    for _, r in res.iterrows():
        row = [r["method"]] + [_pm(r[f"{s}_mean"], r[f"{s}_std"]) for s in splits]
        if has_delta:
            row.append(f"{r['delta']:+.3f}")
        row += [str(int(r[f"rank_{s}"])) for s in splits]
        rows.append(row)
    return _md_table(headers, rows)


def plot_dataset(res: pd.DataFrame, splits: list[str], dataset: str, task: str, path: Path) -> None:
    metric, lower = primary_metric(task)
    x = np.arange(len(res))
    width = 0.8 / len(splits)
    fig, ax = plt.subplots(figsize=(max(6.0, 1.1 * len(res) + 2), 4.4))
    for i, s in enumerate(splits):
        ax.bar(
            x - 0.4 + width * (i + 0.5), res[f"{s}_mean"], width,
            yerr=res[f"{s}_std"].fillna(0), capsize=3, label=s,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(res["method"], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(metric.upper().replace("_", "-"))
    ax.set_title(f"{dataset}: test {metric} ({'lower' if lower else 'higher'} is better)", fontsize=10)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def audit_table(results_dir: Path) -> str:
    rows = []
    for path in sorted((Path(results_dir) / "audit").glob("*.json")):
        rep = json.loads(path.read_text())
        rows.append([
            path.stem, str(rep["n_raw"]), str(rep["n_invalid_smiles"]), str(rep["n_multi_fragment_records"]),
            str(rep["n_duplicate_rows_removed"]), str(rep["n_conflicting_groups_dropped"]),
            str(rep["n_final"]), str(rep["n_scaffolds"]),
            f"{rep['largest_scaffold_fraction']:.1%}",
        ])
    if not rows:
        return "_No audit files found. Run `molprop audit`._"
    headers = ["dataset", "raw rows", "invalid SMILES", "multi-fragment", "duplicate rows removed",
               "conflicting groups dropped", "final molecules", "scaffolds", "largest scaffold share"]
    return _md_table(headers, rows)


def similarity_table_and_plot(results_dir: Path, figures_dir: Path) -> str:
    sims = _read_dir(Path(results_dir) / "nn", "*.csv")
    if sims.empty:
        return "_No similarity files found._"
    agg = sims.groupby(["dataset", "split"])["median"].agg(["mean", "std"]).reset_index()
    rows = [[r["dataset"], r["split"], _pm(r["mean"], r["std"])] for _, r in agg.iterrows()]
    table = _md_table(["dataset", "split", "median nearest-neighbour Tanimoto (mean ± std over seeds)"], rows)

    datasets = list(dict.fromkeys(agg["dataset"]))
    splits = [s for s in SPLIT_ORDER if s in set(agg["split"])]
    x = np.arange(len(datasets))
    width = 0.8 / len(splits)
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, s in enumerate(splits):
        sub = agg[agg["split"] == s].set_index("dataset").reindex(datasets)
        ax.bar(x - 0.4 + width * (i + 0.5), sub["mean"], width, yerr=sub["std"].fillna(0), capsize=3, label=s)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("median nearest-neighbour similarity\n(test → train, Tanimoto)")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(figures_dir) / "nn_similarity.png", dpi=150)
    plt.close(fig)
    return table + "\n\n![nearest-neighbour similarity](figures/nn_similarity.png)"


def build_report(results_dir: Path) -> Path:
    results_dir = Path(results_dir)
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    df = load_results(results_dir)

    parts = [
        "# Results summary",
        "",
        "Generated by `molprop report`. Numbers are test-set values, mean ± standard deviation over seeds.",
        "Δ = scaffold − random: for RMSE a positive Δ means the model looks worse under the scaffold split; "
        "for ROC-AUC a negative Δ means the same. Ranks are per split (1 = best).",
        "",
        "## Data audit",
        "",
        audit_table(results_dir),
    ]
    for name in df["dataset"].unique():
        sub = df[df["dataset"] == name]
        task = sub["task"].iloc[0]
        res, splits = summarize_dataset(sub, task)
        plot_dataset(res, splits, name, task, figures_dir / f"{name}_primary_metric.png")
        metric, _ = primary_metric(task)
        desc = DATASETS[name].description if name in DATASETS else ""
        n_seeds = sub["seed"].nunique()
        parts += [
            "",
            f"## {name}",
            "",
            f"{desc}. Headline metric: test {metric.upper().replace('_', '-')}, {n_seeds} seeds.",
            "",
            dataset_table(res, splits),
            "",
            f"![{name}](figures/{name}_primary_metric.png)",
        ]
    parts += [
        "",
        "## Test-to-train similarity",
        "",
        "Median Tanimoto similarity (Morgan bits, radius 2) between each test molecule and its nearest "
        "training molecule. Lower values mean the test set is further from what the model was trained on.",
        "",
        similarity_table_and_plot(results_dir, figures_dir),
        "",
    ]
    out = results_dir / "summary.md"
    out.write_text("\n".join(parts))
    return out
