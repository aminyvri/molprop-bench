"""All RDKit-dependent code lives here so the rest of the package stays importable without it."""

from __future__ import annotations

import numpy as np

try:
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import Descriptors, rdFingerprintGenerator
    from rdkit.Chem.MolStandardize import rdMolStandardize
    from rdkit.Chem.Scaffolds import MurckoScaffold
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "RDKit is required for this step. Install it with `pip install rdkit` "
        "(or `conda install -c conda-forge rdkit`)."
    ) from exc

RDLogger.DisableLog("rdApp.*")  # invalid SMILES are counted in the audit, not spammed to stderr


def parse_smiles(smiles):
    """Return an RDKit Mol, or None if the SMILES is missing or cannot be parsed."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    return Chem.MolFromSmiles(smiles)


def num_fragments(mol) -> int:
    return len(Chem.GetMolFrags(mol))


def largest_fragment(mol):
    """Strip counter-ions and solvents by keeping the largest fragment."""
    return rdMolStandardize.LargestFragmentChooser().choose(mol)


def canonical_smiles(mol) -> str:
    return Chem.MolToSmiles(mol)


def murcko_scaffold(mol) -> str:
    """Bemis-Murcko scaffold as SMILES (empty string for acyclic molecules)."""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def _mol_or_raise(smiles: str):
    mol = parse_smiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot parse SMILES: {smiles!r}")
    return mol


def morgan_counts(smiles, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Morgan (ECFP-like) count fingerprints, log1p-transformed. Shape (n, n_bits)."""
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    out = np.zeros((len(smiles), n_bits), dtype=np.float32)
    for i, smi in enumerate(smiles):
        out[i] = gen.GetCountFingerprintAsNumPy(_mol_or_raise(smi))
    return np.log1p(out)


def morgan_bitvects(smiles, radius: int = 2, n_bits: int = 2048) -> list:
    """Bit-vector fingerprints used for Tanimoto similarity."""
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return [gen.GetFingerprint(_mol_or_raise(smi)) for smi in smiles]


def bulk_tanimoto(fp, fps) -> list[float]:
    return list(DataStructs.BulkTanimotoSimilarity(fp, fps))


def rdkit_descriptors(smiles) -> tuple[np.ndarray, list[str]]:
    """All RDKit 2D descriptors. Failed or non-finite values become NaN; all-NaN columns are dropped."""
    names = [name for name, _ in Descriptors.descList]
    out = np.full((len(smiles), len(names)), np.nan, dtype=np.float64)
    for i, smi in enumerate(smiles):
        mol = _mol_or_raise(smi)
        for j, (_, fn) in enumerate(Descriptors.descList):
            try:
                out[i, j] = fn(mol)
            except Exception:
                pass  # leave NaN
    out[~np.isfinite(out)] = np.nan
    out[np.abs(out) > 1e10] = np.nan  # a few descriptors (e.g. Ipc) overflow for large molecules
    keep = ~np.all(np.isnan(out), axis=0)
    return out[:, keep], [n for n, k in zip(names, keep) if k]
