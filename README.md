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

> **TODO (fill in after running):** paste the headline table(s) from `results/summary.md` here and describe, in a
> few sentences, what you observe: how much worse the models look under the scaffold split, whether the ranking of
> methods changes, and what the nearest-neighbour similarity shows. Report only what your own run produced.

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
