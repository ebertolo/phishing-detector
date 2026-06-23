# ML Stack, Ensembling & Encoding — Quick Reference

A succinct guide to the main machine-learning libraries and the
stacking/ensemble and encoding techniques used in this project, and **how to
select different encoding strategies for validation**. For *why* each technique
was kept or discarded, see [EXPERIMENT_JOURNEY.md](EXPERIMENT_JOURNEY.md); for the
numbers, see [RESULTS.md](RESULTS.md).

## Core ML libraries

| Library | Role here |
|---|---|
| **scikit-learn** | Pipelines, `GridSearchCV`, `StratifiedKFold`, calibration (`CalibratedClassifierCV` + `FrozenEstimator`), metrics, `KBinsDiscretizer`, `QuantileTransformer`, `RandomForest`, `LogisticRegression`, `AdaBoost`, `KMeans`/`GaussianMixture`, the `MLPRegressor` (auto/denoising encoder). The backbone everything plugs into. |
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
the TensorFlow embedding uses any visible GPU with no code change — see
[`docs/colab_experiments.ipynb`](colab_experiments.ipynb) for a GPU-runtime walkthrough.

## Model roster

Gradient boosters (LightGBM / XGBoost / CatBoost, plus focal variants),
RandomForest, two logistic regressions (`logreg` plain, `logreg_woe` over WOE),
and four additional independent algorithms:

- **`tensorflow_dnn`** — Keras dense net: `Dense(40)→BN→Dropout(0.4)` →
  `Dense(20)→BN→Dropout(0.4)` → skip-`Concatenate` → 16-d `embedding` →
  `UnitNormalization` → `tanh` score in [-1, 1] (mapped to a probability and
  calibrated). The BatchNorm / unit-normalization / concatenate operations are
  the "between-layer operations without their own gradient descent". Optimizers
  compared in the grid: **SGD (Nesterov), RMSprop, Adam, Nadam**; LR ∈ {0.1, 0.05};
  epochs capped at 2000 with PR-AUC early stopping. Serialized via a sidecar
  `.keras` file so it round-trips through the joblib version folder. Excluded
  from the default roster (slow on CPU, low ceiling here).
- **`logreg`** — plain balanced logistic regression over the chosen feature mode.
- **`cluster`** — unsupervised KMeans / GaussianMixture (2 groups) used as a
  classifier; labels only align which group is "phishing". Diagnostic, weak.
- **`adaboost`** — AdaBoost over shallow balanced trees (depth ≥ 2 to avoid the
  degenerate-stump abort under imbalance).

## Feature transformers (selectable via feature modes)

Beyond the encodings, three post-engineering steps compose after `FeatureEngineer`:

- **`engineered_nnembed`** — appends the 20 `nn_*` activations of the dense net's
  embedding layer (leakage-safe: the net trains on the fold's train split). The
  one new addition that *helped* a booster (see `RESULTS.md`).
- **`engineered_smooth`** — `FeatureSmoothing`: winsorize (clip 1/99 percentiles)
  then quantile/rank transform. Robust to noise/outliers; flags pass through.
- **`engineered_denoise`** — denoising autoencoder reconstruction (Gaussian
  corruption at fit, reconstruct clean input). A common fraud/rare-event
  smoothing technique; here it discards too much signal.

## Why not plain accuracy

With ~1.3% phishing, a model that always says "legit" is ~98.7% accurate and
useless. We rank by **PR-AUC (average precision)** and report recall, precision,
F1, MCC and the confusion matrix at a tuned threshold. Accuracy is never the
headline.

## Encoding strategies (selectable per experiment)

The integer count features are turned into "ranges of suspicious behaviour"
rather than raw magnitudes. Each strategy is a leakage-safe, sklearn-compatible
transformer that plugs into the model pipeline. Select one with `--feature-mode`
(CLI) or the dropdown (UI):

| `feature_mode` | Technique | Idea | Leakage safety |
|---|---|---|---|
| `engineered` | **Feature engineering** | `FeatureEngineer` expands the 8 raw counts into **52 features**: presence/graded flags, log transforms, length densities, structural ratios, content-word intensities, p95 high-density flags, pairwise interactions, and a leakage-safe logz. **Recommended default** — biggest PR-AUC win. Full catalogue in EXPERIMENT_JOURNEY.md. | Row-wise + a few train-fit stats; target-free. |
| `raw` | none | Integer counts straight in; boosters bin internally. | n/a |
| `binned_woe` | Optimal binning + **WOE** | Supervised bins → Weight of Evidence. Most interpretable; natural input to logistic regression. | Fit per CV fold (uses target). |
| `quantile` | **KBinsDiscretizer** (quantile) | Unsupervised equal-frequency bins → ordinal indices. Stable, target-free. | Quantiles learned at fit time only. |
| `target` | **Target encoding (cross-fit)** | Replace each value with the out-of-fold mean target. Preserves signal across many distinct counts. | **Out-of-fold** encoding on train; smoothing toward the global prior. |
| `autoencoder` | MLP **autoencoder** (CPU) | Unsupervised non-linear compression; the learned latent code becomes the features. Optional/experimental — see note. | Encoder fit on train only. |

### Weight of Evidence (WOE)

For a bin, `WOE = ln(P(feature | phishing) / P(feature | legit))`. Positive WOE =
the range leans phishing. It linearises the relationship for logistic regression
and is fully auditable (one number per bin).

### Target encoding with cross-fitting

Naive target encoding leaks (a row sees its own label). We avoid this with
**out-of-fold**: the training set is split into folds and each fold is encoded
using statistics computed on the *other* folds, with smoothing toward the global
positive rate so rare values do not overfit. Transform-time encoding uses the
full-train mapping.

### Autoencoder note

The autoencoder mode trains an unsupervised `MLPRegressor` to reconstruct the
(scaled) features and uses the bottleneck activations as the new representation.
It runs on CPU with no extra dependencies, but it is **not the recommended
default**: it is slower, non-deterministic, and far less interpretable than the
binning/WOE path, which is the project's priority. It is provided so it can be
**validated and compared** like any other encoding.

## Ensembling

### Per-model comparison
Each algorithm is tuned independently (`GridSearchCV` over `StratifiedKFold`,
scored by PR-AUC) and compared on a held-out test set.

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
