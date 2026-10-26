# molprop-bench

How much of a molecular property model's reported accuracy comes from **how the data were cleaned and split**?

This repository benchmarks simple, well-understood models on three public MoleculeNet datasets
under two evaluation protocols (random split and scaffold split), documents every data-cleaning step,
and measures how far test molecules are from the training set under each protocol.

It is a small benchmark and reproduction study, not a new method. The goal is a careful, reproducible
answer to one question: *do the conclusions you draw about a model survive a harder, more realistic split?*

## Why this matters

Property-prediction papers often report results on random splits. In chemical datasets, close analogues of the
same scaffold end up on both sides of such a split, so the test set partly measures memorisation of near
neighbours. Scaffold splits keep every Bemis–Murcko scaffold entirely in one set, which is a stricter test of
generalisation to new chemistry. Data quality problems (invalid structures, salts, duplicates with conflicting
labels) can inflate or distort either protocol, so they are audited and reported here rather than fixed silently.

## What it does

1. **Download** ESOL (aqueous solubility, regression), Lipophilicity (logD, regression) and BBBP
   (blood–brain barrier penetration, classification) from the MoleculeNet CSV files.
2. **Audit and clean** (`molprop audit`). Every change is counted in `results/audit/<dataset>.json`:
   - rows with missing targets or SMILES that RDKit cannot parse are dropped;
   - multi-fragment records are reduced to the largest fragment (salts, solvents), then canonicalised;
   - duplicates (same canonical SMILES) are merged; regression labels are averaged, and a group is dropped if its
     labels differ by more than a threshold (default 1.0 target unit); classification groups with disagreeing
     labels are dropped;
   - Bemis–Murcko scaffolds are computed (acyclic molecules share the empty scaffold).
3. **Split** three ways (`src/molprop/splits.py`), 80/10/10:
   - `random`: uniform random split;
   - `scaffold`: seeded "balanced" scaffold split. Whole scaffold groups are assigned to one set, and large groups go
     to training first. The remaining groups are shuffled by seed, so each seed gives a different split;
   - `scaffold_det`: the classic deterministic scaffold split (optional, `--splits random scaffold scaffold_det`).
4. **Model** (`src/molprop/models.py`):
   - features: Morgan count fingerprints (radius 2, 2048 bits, log1p) and RDKit 2D descriptors;
   - models: mean/prior baseline, ridge or logistic regression, random forest;
   - a tiny grid per model, **selected on the validation set only**; the test set is touched once per run.
5. **Evaluate**: RMSE, MAE and R² for regression; ROC-AUC and PR-AUC for classification. Results are averaged over seeds.
6. **Analyse** (`src/molprop/analysis.py`): median Tanimoto similarity between each test molecule and its nearest
   training molecule, for each split, to show *why* the numbers change.

## Quick start

```bash
git clone <this repository>
cd molprop-bench
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

make test        # unit tests (the RDKit tests run only if RDKit is installed)
make quick       # 2 seeds, 100 trees: a fast end-to-end check
make all         # full run: 5 seeds, 300 trees (can take a while on a laptop CPU)
```

Without `make`, run the steps directly:

```bash
molprop download
molprop audit
molprop run            # or: molprop run --seeds 10 --splits random scaffold scaffold_det
molprop report
```

If the download is blocked, download the three CSV files (`delaney-processed.csv`, `Lipophilicity.csv`, `BBBP.csv`)
manually and place them in `data/raw/`.

Outputs:

| path | content |
|---|---|
| `data/processed/*.csv` | cleaned molecules with scaffolds (git-ignored) |
| `results/audit/*.json` | counts for every cleaning step |
| `results/raw/*.csv` | one row per (split, seed, features, model) with validation and test metrics |
| `results/nn/*.csv` | test-to-train similarity statistics |
| `results/summary.md`, `results/figures/*.png` | tables and plots produced by `molprop report` |

## Results

Results below are from a run with 2 seeds and the random + scaffold splits (RDKit 2023+, scikit-learn 1.x). Numbers are test-set values (mean ± std). Δ = scaffold − random: positive Δ for RMSE (or negative Δ for ROC-AUC) means the model looks worse under the harder split. Ranks are per split (1 = best).

### Data audit

| dataset | raw rows | invalid SMILES | multi-fragment | duplicate rows removed | conflicting groups dropped | final molecules | scaffolds | largest scaffold share |
|---|---|---|---|---|---|---|---|---|
| esol | 1128 | 0 | 0 | 11 | 1 | 1116 | 269 | 28.2 % |
| lipophilicity | 4200 | 0 | 1 | 0 | 0 | 4200 | 2408 | 1.8 % |
| bbbp | 2050 | 11 | 105 | 73 | 11 | 1955 | 1016 | 6.5 % |

### ESOL (aqueous solubility, RMSE ↓)

| method | random | scaffold | Δ (scaffold − random) | rank (random) | rank (scaffold) |
|---|---|---|---|---|---|
| ridge + RDKit descriptors | 0.677 ± 0.002 | 0.937 ± 0.101 | **+0.259** | 1 | 2 |
| random forest + RDKit descriptors | 0.685 ± 0.035 | 0.925 ± 0.051 | **+0.240** | 2 | 1 |
| random forest + Morgan counts | 0.857 ± 0.014 | 1.227 ± 0.285 | **+0.370** | 3 | 3 |
| ridge + Morgan counts | 1.209 ± 0.074 | 1.379 ± 0.074 | +0.169 | 4 | 4 |
| baseline (mean) | 2.061 ± 0.020 | 2.245 ± 0.152 | +0.184 | 5 | 5 |

### Lipophilicity (logD, RMSE ↓)

| method | random | scaffold | Δ (scaffold − random) | rank (random) | rank (scaffold) |
|---|---|---|---|---|---|
| random forest + RDKit descriptors | 0.686 ± 0.002 | 0.755 ± 0.005 | +0.069 | 1 | 1 |
| random forest + Morgan counts | 0.722 ± 0.032 | 0.852 ± 0.007 | +0.130 | 2 | 2 |
| ridge + RDKit descriptors | 0.824 ± 0.020 | 0.900 ± 0.039 | +0.076 | 3 | 3 |
| ridge + Morgan counts | 0.960 ± 0.048 | 1.031 ± 0.049 | +0.071 | 4 | 4 |
| baseline (mean) | 1.145 ± 0.066 | 1.201 ± 0.050 | +0.056 | 5 | 5 |

### BBBP (blood–brain barrier, ROC-AUC ↑)

| method | random | scaffold | Δ (scaffold − random) | rank (random) | rank (scaffold) |
|---|---|---|---|---|---|
| random forest + RDKit descriptors | 0.942 ± 0.010 | 0.942 ± 0.024 | −0.000 | 1 | 1 |
| random forest + Morgan counts | 0.928 ± 0.014 | 0.926 ± 0.016 | −0.002 | 2 | 2 |
| logistic + RDKit descriptors | 0.900 ± 0.010 | 0.891 ± 0.003 | −0.009 | 3 | 4 |
| logistic + Morgan counts | 0.894 ± 0.005 | 0.897 ± 0.008 | +0.003 | 4 | 3 |
| baseline (class prior) | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.000 | 5 | 5 |

### Test-to-train nearest-neighbour similarity

Median Tanimoto similarity (Morgan fingerprints, radius 2) of each test molecule to its nearest training neighbour. Scaffold splits systematically place the test set further from the training chemistry.

| dataset | random | scaffold |
|---|---|---|
| esol | 0.544 ± 0.020 | 0.399 ± 0.042 |
| lipophilicity | 0.702 ± 0.014 | 0.585 ± 0.003 |
| bbbp | 0.562 ± 0.005 | 0.448 ± 0.015 |

**Take-away.** On ESOL the scaffold split raises RMSE by 0.24–0.37 for the stronger models; ranking between ridge and random forest also swaps. On Lipophilicity the degradation is milder but still consistent. On BBBP ROC-AUC is almost unchanged. The drop in nearest-neighbour Tanimoto confirms that the scaffold protocol removes the near-analogue leakage that random splits permit. Even simple models therefore look substantially weaker once the evaluation protocol is made more realistic.

## Limitations

- Only simple baselines. No graph neural networks or pretrained models, so conclusions apply to these models.
- Small hyperparameter grids, one 80/10/10 split per seed, and a small number of seeds. Standard deviations across
  seeds are indicative, not formal confidence intervals.
- Models are trained on the training split only (no refit on train + validation).
- Scaffold splits depend on the scaffold definition (Bemis–Murcko, no chirality); other definitions would give
  different splits. Acyclic molecules all share one scaffold group.
- Duplicate handling uses a simple threshold on label disagreement, which is a judgement call. Change it with
  `molprop audit --conflict-threshold`.
- MoleculeNet labels come from heterogeneous sources and contain experimental noise that no split can remove.

## Project layout

```
src/molprop/
  datasets.py   dataset specs and download
  chem.py       all RDKit code (parsing, scaffolds, fingerprints, descriptors)
  audit.py      validation, standardisation, deduplication, reporting
  splits.py     random and scaffold splits
  models.py     estimators and small hyperparameter grids
  metrics.py    metrics and the validation-based selection rule
  benchmark.py  runs every (split, seed, features, model) combination
  analysis.py   test-to-train nearest-neighbour similarity
  report.py     Markdown tables and figures
  cli.py        `molprop` command
tests/          unit tests (splits, metrics, models, benchmark/report, RDKit steps)
```

## Reproducibility

Seeds are fixed (`0 .. seeds-1`) and used for splits and random forests. Library versions are listed in
`requirements.txt`; exact numbers can still differ slightly across RDKit and scikit-learn versions.

## References

- Wu et al., *MoleculeNet: a benchmark for molecular machine learning*, Chemical Science, 2018.
- Bemis and Murcko, *The properties of known drugs. 1. Molecular frameworks*, J. Med. Chem., 1996.
- RDKit: open-source cheminformatics, <https://www.rdkit.org>.

## License

MIT, see `LICENSE`.
