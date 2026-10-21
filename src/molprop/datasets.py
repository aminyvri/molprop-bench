"""Dataset specifications and download helpers (MoleculeNet CSV files)."""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

BASE_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    filename: str
    smiles_col: str
    target_col: str
    task: str  # "regression" or "classification"
    description: str

    @property
    def url(self) -> str:
        return f"{BASE_URL}/{self.filename}"


DATASETS: dict[str, DatasetSpec] = {
    "esol": DatasetSpec(
        name="esol",
        filename="delaney-processed.csv",
        smiles_col="smiles",
        target_col="measured log solubility in mols per litre",
        task="regression",
        description="Aqueous solubility (log mol/L)",
    ),
    "lipophilicity": DatasetSpec(
        name="lipophilicity",
        filename="Lipophilicity.csv",
        smiles_col="smiles",
        target_col="exp",
        task="regression",
        description="Octanol/water distribution coefficient (logD, pH 7.4)",
    ),
    "bbbp": DatasetSpec(
        name="bbbp",
        filename="BBBP.csv",
        smiles_col="smiles",
        target_col="p_np",
        task="classification",
        description="Blood-brain barrier penetration (binary)",
    ),
}


def raw_path(spec: DatasetSpec, data_dir: Path) -> Path:
    return Path(data_dir) / "raw" / spec.filename


def processed_path(spec: DatasetSpec, data_dir: Path) -> Path:
    return Path(data_dir) / "processed" / f"{spec.name}.csv"


def download(spec: DatasetSpec, data_dir: Path, force: bool = False) -> Path:
    """Download the raw CSV unless it is already on disk."""
    path = raw_path(spec, data_dir)
    if path.exists() and not force:
        print(f"[download] {spec.name}: already present at {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {spec.name}: {spec.url}")
    try:
        with urllib.request.urlopen(spec.url, timeout=60) as resp:
            path.write_bytes(resp.read())
    except OSError as exc:
        raise RuntimeError(
            f"Could not download {spec.url} ({exc}). "
            f"Download it manually and place it at {path}."
        ) from exc
    return path


def load_raw(spec: DatasetSpec, data_dir: Path) -> pd.DataFrame:
    """Read the raw CSV and return a frame with columns `smiles` and `target`."""
    path = raw_path(spec, data_dir)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `molprop download` first.")
    df = pd.read_csv(path)
    missing = [c for c in (spec.smiles_col, spec.target_col) if c not in df.columns]
    if missing:
        raise KeyError(f"{path.name} is missing expected columns {missing}; found {list(df.columns)}")
    out = df[[spec.smiles_col, spec.target_col]].copy()
    out.columns = ["smiles", "target"]
    return out
