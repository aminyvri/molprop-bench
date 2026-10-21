"""Command line interface: download, audit, run, report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .benchmark import RunConfig, run_dataset
from .datasets import DATASETS, download, load_raw, processed_path
from .models import FEATURE_SETS, MODELS
from .splits import SPLIT_NAMES, make_split


def _specs(names):
    unknown = [n for n in names if n not in DATASETS]
    if unknown:
        raise SystemExit(f"Unknown dataset(s) {unknown}; choose from {sorted(DATASETS)}")
    return [DATASETS[n] for n in names]


def cmd_download(args) -> None:
    for spec in _specs(args.datasets):
        download(spec, args.data_dir, force=args.force)


def cmd_audit(args) -> None:
    from .audit import audit_and_clean

    audit_dir = args.results_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    for spec in _specs(args.datasets):
        clean, report = audit_and_clean(load_raw(spec, args.data_dir), spec.task, args.conflict_threshold)
        out = processed_path(spec, args.data_dir)
        out.parent.mkdir(parents=True, exist_ok=True)
        clean.to_csv(out, index=False)
        (audit_dir / f"{spec.name}.json").write_text(json.dumps(report, indent=2))
        print(
            f"[audit] {spec.name}: {report['n_raw']} raw -> {report['n_final']} molecules "
            f"({report['n_invalid_smiles']} invalid, {report['n_duplicate_rows_removed']} duplicate rows removed, "
            f"{report['n_conflicting_groups_dropped']} conflicting groups dropped, "
            f"{report['n_scaffolds']} scaffolds)"
        )


def cmd_run(args) -> None:
    from . import chem
    from .analysis import nn_similarity, summarize_similarity

    seeds = 2 if args.quick else args.seeds
    n_estimators = 100 if args.quick else args.n_estimators
    cfg = RunConfig(
        seeds=tuple(range(seeds)),
        splits=tuple(args.splits),
        features=FEATURE_SETS,
        models=MODELS,
        n_estimators=n_estimators,
    )
    raw_dir = args.results_dir / "raw"
    nn_dir = args.results_dir / "nn"
    raw_dir.mkdir(parents=True, exist_ok=True)
    nn_dir.mkdir(parents=True, exist_ok=True)

    for spec in _specs(args.datasets):
        path = processed_path(spec, args.data_dir)
        if not path.exists():
            raise SystemExit(f"{path} not found. Run `molprop audit` first.")
        df = pd.read_csv(path)
        df["scaffold"] = df["scaffold"].fillna("")
        smiles = df["smiles"].tolist()
        print(f"[run] {spec.name}: {len(df)} molecules, computing features ...", flush=True)
        features = {
            "morgan": chem.morgan_counts(smiles),
            "descriptors": chem.rdkit_descriptors(smiles)[0],
        }
        results = run_dataset(spec.name, spec.task, df["target"].to_numpy(), df["scaffold"].tolist(), features, cfg)
        results.to_csv(raw_dir / f"{spec.name}.csv", index=False)

        fps = chem.morgan_bitvects(smiles)
        rows = []
        for seed in cfg.seeds:
            for split_name in cfg.splits:
                train, _, test = make_split(split_name, len(df), df["scaffold"].tolist(), seed, cfg.frac)
                summary = summarize_similarity(nn_similarity(fps, train, test))
                rows.append({"dataset": spec.name, "split": split_name, "seed": seed, **summary})
        pd.DataFrame(rows).to_csv(nn_dir / f"{spec.name}.csv", index=False)
        print(f"[run] {spec.name}: saved results to {raw_dir / (spec.name + '.csv')}")


def cmd_report(args) -> None:
    from .report import build_report

    out = build_report(args.results_dir)
    print(f"[report] wrote {out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="molprop", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--data-dir", type=Path, default=Path("data"))
        p.add_argument("--results-dir", type=Path, default=Path("results"))
        p.add_argument("--datasets", nargs="+", default=list(DATASETS), metavar="NAME")

    p = sub.add_parser("download", help="download the MoleculeNet CSV files")
    common(p)
    p.add_argument("--force", action="store_true", help="re-download even if the file exists")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("audit", help="validate, standardise, deduplicate and write cleaned data")
    common(p)
    p.add_argument("--conflict-threshold", type=float, default=1.0,
                   help="regression: drop duplicate groups whose labels differ by more than this")
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser("run", help="train and evaluate all models under all splits")
    common(p)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--splits", nargs="+", default=["random", "scaffold"], choices=SPLIT_NAMES)
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--quick", action="store_true", help="2 seeds and 100 trees, for a fast sanity check")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("report", help="build results/summary.md and figures")
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.set_defaults(func=cmd_report)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
