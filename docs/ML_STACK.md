# ML Stack, Ensembling & Encoding — Quick Reference

A succinct guide to the main machine-learning libraries and the
stacking/ensemble and encoding techniques used in this project, and **how to
select different encoding strategies for validation**. For *why* each technique
was kept or discarded, see [EXPERIMENT_JOURNEY.md](EXPERIMENT_JOURNEY.md); for the
numbers, see [RESULTS.md](RESULTS.md).

## Core ML libraries

| Library | Role here |
| --- | --- |
| **scikit-learn** | Pipelines, `GridSearchCV` / `RandomizedSearchCV`, `StratifiedKFold`, calibration (`CalibratedClassifierCV` + `FrozenEstimator`), metrics, `KBinsDiscretizer`, `QuantileTransformer`, `RandomForest`, `LogisticRegression`, `AdaBoost`, `KMeans`/`GaussianMixture`, the `MLPRegressor` (auto/denoising encoder). The backbone everything plugs into. |
| **imbalanced-learn** | Imbalance-aware pipeline support for the ~1.3% positive class. We rely mainly on per-model class weighting rather than aggressive resampling. |
| **XGBoost** | Gradient-boosted trees; imbalance via `scale_pos_weight`. Strong tabular performer. |
| **LightGBM** | Fast histogram-based boosting; `scale_pos_weight`. Usually the quickest strong model. |
| **CatBoost** | Ordered boosting, robust defaults; imbalance via `auto_class_weights="Balanced"`. |
| **optbinning** | Supervised **optimal binning** + **WOE** encoding (interpretable, monotone bins). |
| **TensorFlow / Keras** | CPU dense neural net (`tensorflow_dnn`) and the reusable NN embedding. Lazily imported. |
| **MLflow** | Experiment tracking for fit and inference (params, metrics, curves, WOE tables, artifacts). |
| **joblib** | Versioned model persistence, and the trained-embedding cache (`embedding_cache.py`). |
| **FastAPI** | REST API (`fastapi_app/`) serving the saved model's `predict_proba` over HTTP; Pydantic request/response validation and the OpenAPI docs (`/docs`, `/redoc`) come from it. |
| **uvicorn** | ASGI server running the FastAPI app (`uv run uvicorn fastapi_app.main:app`). |

The framework runs entirely on **CPU** by default; LightGBM, XGBoost and
CatBoost detect a usable GPU at training time and switch automatically
(`gpu_available()` / `lightgbm_cuda_available()` in `models/_common.py`), and
the TensorFlow embedding uses any visible GPU with no code change.

## Model roster

### Kept — recommended defaults

- **LightGBM / XGBoost / CatBoost** (`DEFAULT_MODELS`) — class-weighted gradient
  boosters; the consistent winners on this tabular, imbalanced problem. XGBoost
  is the strongest single model; the blend of the three is the best overall.
- **CatBoost focal** — native focal loss works and earns weight in the blend.
- **`logreg_woe`** — logistic regression over WOE-encoded features; the
  interpretable baseline (PR-AUC 0.069 standalone, but useful for auditing).
- **RandomForest** — solid interpretable tree ensemble; kept as comparison baseline.

### Discarded (kept in the codebase as comparison baselines)

| Model | PR-AUC | Why discarded |
| --- | --- | --- |
| `logreg` (plain) | 0.069 | Linear; can't capture feature interactions. [§3](EXPERIMENT_JOURNEY.md#3-models-tested--discarded-kept-and-why-ensembling--calibration) |
| `adaboost` | 0.021 | Depth-1 stumps abort under imbalance; NaN CV folds. [§3](EXPERIMENT_JOURNEY.md#3-models-tested--discarded-kept-and-why-ensembling--calibration) |
| `cluster` (KMeans/GMM) | 0.013 | Unsupervised split ≈ base rate; classes overlap completely. [§3](EXPERIMENT_JOURNEY.md#3-models-tested--discarded-kept-and-why-ensembling--calibration) |
| `tensorflow_dnn` | ~0.08 | Weak ceiling standalone; slow on CPU. Embedding is the valuable part — see below. [§3](EXPERIMENT_JOURNEY.md#3-models-tested--discarded-kept-and-why-ensembling--calibration) |
| LightGBM/XGBoost focal | unstable | Hand-rolled custom objective collapses at ~1.3% imbalance (all-positive, NaN CV); native-API rewrite also collapses — the instability is the imbalance itself. [§3–4](EXPERIMENT_JOURNEY.md#4-training-optimisation--splits-search-and-rare-positive-focus) |
| SMOTE / oversampling | — | Class weights + boosters + PR-AUC outperform aggressive resampling under extreme imbalance. [§1](EXPERIMENT_JOURNEY.md#1-data-nature-candidate-algorithms-and-evaluation-metrics) |

## Feature transformers (selectable via `--feature-mode`)

### Recommended path (in order of proven PR-AUC)

1. **`engineered`** — deterministic `FeatureEngineer`: 8 raw counts → 52 features
   (presence/graded flags, log transforms, length densities, structural ratios,
   content-word intensities, p95 high-density flags, pairwise interactions,
   leakage-safe logz). Row-wise and target-free — leakage-safe by construction.
   **+33% over raw** (0.190 → 0.252 on 100k; blend 0.311 on full 524k).
   [§2 v1](EXPERIMENT_JOURNEY.md#2-feature-engineering--versions-and-the-transformation-sequence)

2. **`engineered_nnembed`** — engineering + frozen NN embedding: appends 16 `nn_*`
   activations from the dense net's embedding layer (default `embedding_dim=16`).
   The net trains **once on the train split only** — never retrained per CV fold
   (that caused +50% leakage; see [§5](EXPERIMENT_JOURNEY.md#5-nn-embedding--the-net-is-weak-but-its-embedding-helps)).
   On the full dataset: **+0.015 PR-AUC per booster** (e.g. XGBoost 0.4235 → 0.4385);
   hurts on 100k (too few positives to train the embedding well).

### Discarded / available for validation only

- **`raw`** — integer counts straight in; boosters bin internally. Solid baseline
  (0.209 on 100k) and beats all pre-discretising encodings — but `engineered`
  surpasses it significantly.

- **`binned_woe`** — optimal binning + WOE. Discarded as default: PR-AUC 0.111
  vs 0.209 for `raw` — pre-discretising throws away resolution the boosters split
  on. Kept as the natural input to `logreg_woe`.
  [§2 v0](EXPERIMENT_JOURNEY.md#2-feature-engineering--versions-and-the-transformation-sequence)

- **`quantile`** — `KBinsDiscretizer` (quantile). Discarded: PR-AUC 0.101.
  Same reason as WOE — boosters split better on raw integers.
  [§2 v0](EXPERIMENT_JOURNEY.md#2-feature-engineering--versions-and-the-transformation-sequence)

- **`target`** — cross-fit mean-target encoding. Discarded: PR-AUC 0.043, the
  worst of all encodings; sparse counts are a poor fit for smooth target averages.
  [§2 v0](EXPERIMENT_JOURNEY.md#2-feature-engineering--versions-and-the-transformation-sequence)

- **`autoencoder`** — unsupervised MLP bottleneck features (CPU). Experimental;
  slower, non-deterministic, far less interpretable. Not systematically measured;
  not on the recommended path.

- **`engineered_smooth`** — winsorize + quantile transform after engineering.
  Discarded: PR-AUC 0.236 vs 0.247 plain `engineered`; boosters are already
  robust to outliers and smoothing removes the resolution they split on.
  [§2 discarded](EXPERIMENT_JOURNEY.md#2-feature-engineering--versions-and-the-transformation-sequence)

- **`engineered_denoise`** — denoising autoencoder reconstruction after engineering.
  Discarded: PR-AUC 0.145, the worst post-engineering step; too much signal lost
  in reconstruction.
  [§2 discarded](EXPERIMENT_JOURNEY.md#2-feature-engineering--versions-and-the-transformation-sequence)

- **`engineered_noemail`** — engineering without the `logz_num_email_addresses`
  feature. Ablation diagnostic only.

- **`ix_raw` / `ix_engineered`** — prepends explicit pairwise `InteractionFeatures`
  before the base encoding. Experimental; interactions are already captured inside
  `engineered`; little extra gain expected.

## Why not plain accuracy

With ~1.3% phishing, a model that always says "legit" is ~98.7% accurate and
useless. We rank by **PR-AUC (average precision)** and report recall, precision,
F1, MCC and the confusion matrix at a tuned threshold. Accuracy is never the
headline.

## Ensembling

### Per-model comparison

Each algorithm is tuned independently (`GridSearchCV` or `RandomizedSearchCV`
over `StratifiedKFold`, scored by PR-AUC; controlled by `--search grid|random`)
and compared on a held-out test set.

### Probability calibration

Before combining, each best model's probabilities are calibrated on the
validation set (`CalibratedClassifierCV` wrapping a `FrozenEstimator`). Blending
and threshold tuning then operate on trustworthy probabilities.

### Blending (the default combiner here)

A **weighted average** of the calibrated base-model probabilities. Weights are
searched on the validation set to maximise PR-AUC (equal-weight average is always
a candidate). Simple, robust, and low-leakage — the blend flows through the same
threshold/metrics/persistence path as a single model.

### Stacking (implemented)

A logistic **meta-model** (`core/stacking.py`) fit on the base models' validation
probabilities — the held-out set they were not trained on, keeping it
leakage-safe within the train/val/test split. Enable with `--stacking`. In
experiments it lands just behind the blend on PR-AUC but reaches **higher
recall**, which can be preferable operationally.

### Calibration: sigmoid vs isotonic

Base probabilities are calibrated on validation before blending/stacking.
`--calibration sigmoid` (Platt, robust with little data, default) or
`--calibration isotonic` (non-parametric, needs more validation samples).

### Focal loss boosters

Focal loss down-weights easy negatives to focus on hard examples. **CatBoost
focal** (`catboost_focal`, native loss) works well and even contributes to the
winning blend. The **LightGBM/XGBoost focal** variants use a hand-rolled custom
objective that is correct on balanced data but unstable at this ~1.3% imbalance,
so they are available but not in the default roster (class-weighted boosters are
the robust choice).

## Threshold tuning

The operating point is chosen on validation, not fixed at 0.5:

- **recall-target** — minimum recall X with precision ≥ Y (false negatives cost
  more in phishing);
- **max-F1**;
- **manual** — explicit value;
- **cost** — minimise `fn_cost*FN + fp_cost*FP` (default FN = 10× FP), the most
  direct way to encode the business cost of a missed phishing email.

## How to validate different encodings

```bash
# Compare encodings for the same model set:
uv run python scripts/run_experiments.py --csv data/email_phishing_data.csv --feature-mode raw
uv run python scripts/run_experiments.py --csv data/email_phishing_data.csv --feature-mode binned_woe
uv run python scripts/run_experiments.py --csv data/email_phishing_data.csv --feature-mode quantile
uv run python scripts/run_experiments.py --csv data/email_phishing_data.csv --feature-mode target
```

Each run produces a PR-AUC-sorted comparison and logs to MLflow, so encoding
strategies can be compared side by side.
