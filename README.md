# Sequential Deconfounder

This project extends the deconfounder framework (Wang and Blei, 2019) to a
temporal setting for continuous, multi-cause ad exposures. A deep state-space
model infers user-specific latent trajectories that act as substitute
confounders, enabling causal estimation of next-day and cumulative purchase
effects.

## Project Overview

Pipeline:
1. Data preprocessing 
2. Assignment model (state-space model)
3. Posterior predictive checks
4. Outcome estimation

## Requirements

Install dependencies from `requirements.txt`:
```
pip install -r requirements.txt
```

Tested with Python 3.9+.

## Data

This project uses the Alibaba *Ali_Display_Ad_Click* dataset (Taobao display ads). Experiments in the report use a random subset (N=12000) of users sampled from the full dataset. To obtain the raw data, please visit https://tianchi.aliyun.com/dataset/56 .

1. Download the dataset from the source.
2. Place raw files in `data/raw/` (or use subsample).
3. Run preprocessing to build tensors and covariates.

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── configs/                 
│   ├── base.yaml
│   ├── train.yaml
│   ├── ppc.yaml
│   └── effects.yaml
├── data/
│   ├── raw/                  # place raw dataset files here (not tracked)
│   ├── processed/            # generated tensors/csvs (not tracked)
│   └── README.md
├── notebooks/                # demo analysis
│   └── Structure.ipynb
├── results/                  # outputs
│   ├── models/               
│   ├── ppc/
│   ├── effects/
│   └── figures/
├── scripts/                  
│   ├── preprocess.py
│   ├── train.py
│   ├── run_ppc.py
│   ├── estimate_effects.py
│   └── run_full_pipeline.py
└── src/
    └── sequential_deconfounder/
        ├── data/             # preprocessing pipeline
        ├── models/           # DVAE model
        ├── inference/        # training code
        ├── diagnostics/      # PPC
        ├── outcomes/         # effect estimation
        └── utils/
```

## Running (Suggested Organization)

The running section should mirror the pipeline and be split into four steps.
Each step should have a single command (CLI) and a notebook alternative.
Run commands from the repo root with `PYTHONPATH=src` or install the package.

### 1. Preprocess Data

```
PYTHONPATH=src python scripts/preprocess.py \
  --config configs/base.yaml
```

Example config for your current local paths:
```
python scripts/preprocess.py \
  --samples_csv /Users/dolly/Desktop/Sequential_Deconfounder/Dataset/Click_Subsample/samples.csv \
  --users_csv /Users/dolly/Desktop/Sequential_Deconfounder/Dataset/Click_Subsample/users.csv \
  --features_csv /Users/dolly/Desktop/Sequential_Deconfounder/Dataset/Click_Subsample/features.csv \
  --behaviors_csv /Users/dolly/Desktop/Sequential_Deconfounder/behavior_log.csv \
  --out_dir /Users/dolly/Desktop/Sequential_Deconfounder/Dataset/Click_Subsample \
  --exposure_start 2017-05-05 \
  --exposure_end 2017-05-12 \
  --max_users 2000 \
  --seed 42
```

### 2. Train Assignment Model

```
PYTHONPATH=src python scripts/train.py \
  --config configs/train.yaml
```

Example (explicit paths):
```
python scripts/train.py \
  --data_npz /Users/dolly/Desktop/Sequential_Deconfounder/Dataset/Click_Subsample/dvae_inputs.npz \
  --model_out /Users/dolly/Desktop/Sequential_Deconfounder/results/models/dvae.pt \
  --log_out /Users/dolly/Desktop/Sequential_Deconfounder/results/models/train_log.csv \
  --latent_dim 200 \
  --hidden_dim 256 \
  --num_epochs 200 \
  --batch_size 64 \
  --device cuda
```

### 3. Posterior Predictive Checks

```
PYTHONPATH=src python scripts/run_ppc.py \
  --config configs/ppc.yaml
```

Example (explicit paths):
```
PYTHONPATH=src python scripts/run_ppc.py \
  --data_npz data/processed/dvae_inputs.npz \
  --model_ckpt results/models/dvae.pt \
  --out_dir results/ppc \
  --holdout_steps 2 \
  --device cuda
```

Notes:
- `mask_type` supports `fixed` or `time_varying` (masked held-out set used for PPC).


### 4. Outcome Estimation

```
PYTHONPATH=src python scripts/estimate_effects.py \
  --config configs/effects.yaml
```

Example (explicit paths):
```
PYTHONPATH=src python scripts/estimate_effects.py \
  --data_npz data/processed/dvae_inputs.npz \
  --model_ckpt results/models/dvae.pt \
  --buy_csv data/processed/buy.csv \
  --out_dir results/effects \
  --fig_dir results/figures \
  --quantiles 0.05 0.25 0.5 0.75 0.95 \
  --qte_method dual \
  --device cuda
```

Notes:
- `qte_method` can be `dual` (multi-quantile, notebook-aligned) or `residualized` (single-quantile).
- If `qte_method=dual`, use `--quantiles` to control the list of quantiles.

### 5. Full Pipeline (Optional)

```
PYTHONPATH=src python scripts/run_full_pipeline.py \
  --preprocess_config configs/base.yaml \
  --train_config configs/train.yaml \
  --ppc_config configs/ppc.yaml \
  --effects_config configs/effects.yaml
```

If you prefer notebooks, keep `notebooks/Structure.ipynb`


## Reproducibility

- Set random seeds in training.
- Save configs and final checkpoints.
- Save PPC figures and effect tables to `results/`.

## References

- Wang, Y., & Blei, D. M. (2019). The blessings of multiple causes. https://doi.org/10.1080/01621459.2019.1686987
