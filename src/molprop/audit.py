"""Data audit and cleaning.

Every step is counted so the cleaning is documented rather than silent:
invalid SMILES, multi-fragment records, duplicates and conflicting labels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import chem


def audit_and_clean(
    raw: pd.DataFrame,
    task: str,
    conflict_threshold: float = 1.0,
) -> tuple[pd.DataFrame, dict]:
    """Clean a raw (smiles, target) frame and report what was changed.

    Steps
    -----
    1. Drop rows with a missing target or a SMILES RDKit cannot parse.
    2. Keep the largest fragment (removes salts and solvents), then canonicalise.
    3. Merge duplicates (same canonical SMILES):
       - regression: average the labels; drop the group if the labels differ by more than
         `conflict_threshold` (in target units), because the value is then ambiguous;
       - classification: drop the group if the labels disagree.
    4. Compute the Bemis-Murcko scaffold of each remaining molecule.
    """
    if task not in {"regression", "classification"}:
        raise ValueError(f"Unknown task {task!r}")

    n_raw = len(raw)
    missing_target = raw["target"].isna()
    n_missing_target = int(missing_target.sum())
    frame = raw.loc[~missing_target]

    invalid_examples: list[str] = []
    n_invalid = 0
    n_multi_fragment = 0
    records: list[tuple[str, float]] = []
    for smi, y in zip(frame["smiles"], frame["target"]):
        mol = chem.parse_smiles(smi)
        if mol is None:
            n_invalid += 1
            if len(invalid_examples) < 5:
                invalid_examples.append(str(smi))
            continue
        if chem.num_fragments(mol) > 1:
            n_multi_fragment += 1
            mol = chem.largest_fragment(mol)
        records.append((chem.canonical_smiles(mol), float(y)))

    groups: dict[str, list[float]] = {}
    for smi, y in records:
        groups.setdefault(smi, []).append(y)

    kept_smiles, kept_target, kept_merged = [], [], []
    n_conflicting_groups = 0
    n_duplicate_rows_removed = 0
    for smi, ys in groups.items():
        n_duplicate_rows_removed += len(ys) - 1
        if len(ys) > 1:
            spread = max(ys) - min(ys)
            conflict = spread > conflict_threshold if task == "regression" else spread > 0
            if conflict:
                n_conflicting_groups += 1
                continue
        kept_smiles.append(smi)
        kept_target.append(float(np.mean(ys)) if task == "regression" else int(round(ys[0])))
        kept_merged.append(len(ys))

    scaffolds = [chem.murcko_scaffold(chem.parse_smiles(s)) for s in kept_smiles]
    clean = pd.DataFrame(
        {"smiles": kept_smiles, "target": kept_target, "scaffold": scaffolds, "n_merged": kept_merged}
    )

    scaffold_counts = clean["scaffold"].value_counts()
    report: dict = {
        "task": task,
        "n_raw": n_raw,
        "n_missing_target": n_missing_target,
        "n_invalid_smiles": n_invalid,
        "invalid_examples": invalid_examples,
        "n_multi_fragment_records": n_multi_fragment,
        "n_duplicate_rows_removed": n_duplicate_rows_removed,
        "n_conflicting_groups_dropped": n_conflicting_groups,
        "conflict_threshold": conflict_threshold if task == "regression" else None,
        "n_final": len(clean),
        "n_scaffolds": int(clean["scaffold"].nunique()),
        "n_acyclic": int((clean["scaffold"] == "").sum()),
        "largest_scaffold_fraction": float(scaffold_counts.iloc[0] / len(clean)) if len(clean) else 0.0,
    }
    if task == "regression":
        report["target_mean"] = float(clean["target"].mean())
        report["target_std"] = float(clean["target"].std())
        report["target_min"] = float(clean["target"].min())
        report["target_max"] = float(clean["target"].max())
    else:
        report["positive_fraction"] = float(clean["target"].mean())
    return clean, report
