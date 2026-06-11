# Usage Guide

This guide explains how to reproduce the experiments.

## Option A — Google Colab (easiest, free GPU)

1. Go to [colab.research.google.com](https://colab.research.google.com/) and upload a notebook from `notebooks/`.
2. **Enable GPU:** Runtime → Change runtime type → Hardware accelerator → **T4 GPU**.
3. Run all cells from top to bottom. The dataset downloads automatically from Kaggle.

### Which notebook to use

| Notebook | What it does | Approx. time (T4) |
|----------|--------------|-------------------|
| `1_TB_Detection.ipynb` | Trains the five models, reports metrics at the default threshold | ~1.5 h |
| `2_TB_Detection_threshold.ipynb` | Adds decision-threshold optimization (max F1 on validation) | ~1.5 h |
| `3_TB_MultiSeed_3seed.ipynb` | Runs all models over 3 seeds, reports mean ± std + significance tests | ~4-5 h |
| `4_TB_MultiSeed_5seed.ipynb` | Same as above but with 5 seeds (stronger statistics) | ~8 h |

### Surviving disconnections (multi-seed notebooks)

The multi-seed notebooks save each (model, seed) result to your Google Drive as
soon as it finishes. If the Colab session disconnects:

1. Reopen the notebook in a new session.
2. Run it again from the top.
3. Already-finished combinations are detected and **skipped automatically**;
   it continues from where it stopped.

Results are written to `MyDrive/tb_multiseed_3seed/` (or `tb_multiseed/` for 5 seeds):
`all_runs.csv`, `summary_mean_std.csv`, `significance_tests.csv`.

## Option B — Local machine

```bash
git clone <your-repo-url>
cd tb-screening-repo
pip install -r requirements.txt
```

Download the dataset manually (requires a free Kaggle account):

```bash
kaggle datasets download -d tawsifurrahman/tuberculosis-tb-chest-xray-dataset
unzip tuberculosis-tb-chest-xray-dataset.zip -d data_raw
# arrange into data/main/Normal and data/main/Tuberculosis
```

Train one model from the command line:

```bash
python src/tb_pipeline.py --data_dir data/main --model EfficientNetB0 --seed 42
```

## Security note about Kaggle credentials

Your `kaggle.json` API token is a **password**. Never paste it into a notebook
cell, screenshot, or commit it to GitHub. The `.gitignore` in this repository
already blocks it. In Colab, use the Secrets panel (the key icon) instead of
hard-coding the token. The dataset is public, so for most uses you do not need
a token in the code at all.

## Reproducibility caveats

- Exact metric values depend on random seeds, GPU, and library versions; small
  differences from the paper are expected and normal.
- With only 3 seeds, statistical significance tests have limited power — see the
  paper's discussion. Use `4_TB_MultiSeed_5seed.ipynb` for stronger statistics.
- The decision threshold is always selected on the validation set, never the
  test set, to avoid data leakage.
