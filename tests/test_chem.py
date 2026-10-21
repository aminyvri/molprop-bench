"""Tests for the RDKit-dependent code; skipped automatically if RDKit is not installed."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("rdkit")

from molprop import chem  # noqa: E402
from molprop.audit import audit_and_clean  # noqa: E402


def test_parse_and_canonical_smiles():
    assert chem.parse_smiles("not_a_smiles") is None
    assert chem.parse_smiles(None) is None
    a = chem.canonical_smiles(chem.parse_smiles("CCO"))
    b = chem.canonical_smiles(chem.parse_smiles("OCC"))
    assert a == b


def test_largest_fragment_removes_counter_ion():
    mol = chem.parse_smiles("CC(=O)[O-].[Na+]")
    assert chem.num_fragments(mol) == 2
    kept = chem.largest_fragment(mol)
    assert chem.num_fragments(kept) == 1


def test_murcko_scaffold():
    assert chem.murcko_scaffold(chem.parse_smiles("Cc1ccccc1")) == "c1ccccc1"
    assert chem.murcko_scaffold(chem.parse_smiles("CCO")) == ""


def test_fingerprint_and_descriptor_shapes():
    smiles = ["CCO", "c1ccccc1", "CC(=O)O"]
    fp = chem.morgan_counts(smiles, n_bits=512)
    assert fp.shape == (3, 512) and np.all(fp >= 0)
    desc, names = chem.rdkit_descriptors(smiles)
    assert desc.shape[0] == 3 and desc.shape[1] == len(names) > 100
    assert len(chem.morgan_bitvects(smiles)) == 3


def test_audit_regression_counts_invalid_duplicates_and_conflicts():
    raw = pd.DataFrame(
        {
            "smiles": ["CCO", "OCC", "not_a_smiles", "CCN", "CCC", "CCC", "CC(=O)[O-].[Na+]"],
            "target": [1.0, 1.2, 0.0, 3.0, 0.0, 5.0, 2.0],
        }
    )
    clean, rep = audit_and_clean(raw, "regression", conflict_threshold=1.0)
    assert rep["n_raw"] == 7
    assert rep["n_invalid_smiles"] == 1
    assert rep["n_multi_fragment_records"] == 1
    assert rep["n_conflicting_groups_dropped"] == 1  # propane labels 0.0 and 5.0
    ethanol = clean[clean["smiles"] == chem.canonical_smiles(chem.parse_smiles("CCO"))]
    assert len(ethanol) == 1 and ethanol["target"].iloc[0] == pytest.approx(1.1)
    assert rep["n_final"] == len(clean) == 3  # ethanol, ethylamine, acetate


def test_audit_classification_drops_label_conflicts():
    raw = pd.DataFrame({"smiles": ["CCO", "OCC", "CCN"], "target": [1, 0, 1]})
    clean, rep = audit_and_clean(raw, "classification")
    assert rep["n_conflicting_groups_dropped"] == 1
    assert list(clean["smiles"]) == [chem.canonical_smiles(chem.parse_smiles("CCN"))]
