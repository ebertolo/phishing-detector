# Design — Phishing Detection Validation Framework

## Context

The goal is a validation framework for phishing-detection algorithms on a
**highly imbalanced** dataset (~1.3% phishing, 6 949 positives out of 524 846
rows). On such data plain accuracy is meaningless — a model that always predicts
"legit" scores ~98.7% while being useless — so the framework treats **PR-AUC
(average precision) as the primary metric** and never reports accuracy as a
headline. The deliverable is a Streamlit application backed by an importable,
UI-free core, the same core served over HTTP by a FastAPI REST API
(`fastapi_app/`), plus command-line entry points for headless runs.

The modeling strategy that proved out is **feature engineering first, then a
calibrated ensemble**: a deterministic `FeatureEngineer` (presence flags + log
transforms + ratios) turns the integer counts into the strongest available
signal, optionally augmented with a frozen NN embedding; a lean blend of
class-weighted gradient boosters is calibrated on validation and its operating
point tuned for recall/precision rather than accuracy. Several encodings (WOE,
quantile, target, autoencoder) and other models remain selectable for comparison
but are not the default — see [EXPERIMENT_JOURNEY.md](EXPERIMENT_JOURNEY.md) for
what was kept vs discarded and why.

## Dataset

Integer count features with a binary target (`label_1`, 1 = phishing):

| feature | meaning |
|---|---|
| `num_words`, `num_unique_words`, `num_stopwords` | message size / vocabulary |
| `num_links`, `num_unique_domains` | link structure |
| `num_email_addresses` | embedded addresses |
| `num_spelling_errors` | text quality |
| `num_urgent_keywords` | urgency cues |

Because the values are counts, the framework reasons about **ranges of
behaviour** ("6+ links") rather than absolute values ("17 vs 18").

## Architecture

Four layers, with a hard rule: **core / features / models / experiments /
observability contain no Streamlit or FastAPI imports.** The UI and CLIs are thin
shells over them.

```
src/phishing/
  core/
    metrics.py        imbalance-aware metric set (PR-AUC primary, never accuracy)
    splits.py         stratified train/val/test + StratifiedKFold
    dataset.py        stratified 95/5 split + stratified-sample helper (CLI)
    thresholding.py   operating point: recall-target / max-F1 / manual / cost-sensitive
    blending.py       weighted average of calibrated base-model probabilities
    blend_model.py    BlendModel — serializable blend (save/load like any model)
    stacking.py       logistic meta-model over base validation probabilities
    persistence.py    versioned save/load: models/<name>__<ts>/{model.joblib, metadata.json}
    wrapper.py        ModelWrapper: fit/predict/predict_proba/validate/metrics/save/load
    data.py           schema (features + target), CSV load, eval-vs-inference detection
  features/
    engineering.py    FeatureEngineer — 8 raw -> 52 features (flags/logs/densities/
                      ratios/interactions); recommended default. See feature catalogue
                      in EXPERIMENT_JOURNEY.md. Embedding input auto-sizes to this count.
    binning.py        OptimalBinningWOE — supervised, leakage-safe, WOE output
    kbins.py          QuantileBinner — unsupervised quantile discretisation
    target_encoding.py CrossFitTargetEncoder — out-of-fold mean-target encoding
    autoencoder.py    AutoencoderEncoder — CPU MLP latent + denoising reconstruction
    interactions.py   InteractionFeatures — explicit pairwise products
    smoothing.py      FeatureSmoothing — winsorize + quantile (noise-robust)
    nn_embedding.py   NNEmbedding — dense-net embedding, default 16-d (leakage-safe, pre-trained)
    selection.py      mutual information + RandomForest importance
  models/             one file per algorithm, common build()/param_grid() interface
    lightgbm_model.py xgboost_model.py catboost_model.py randomforest_model.py
    lightgbm_focal_model.py xgboost_focal_model.py catboost_focal_model.py  (_focal.py helper)
    logreg_model.py logreg_woe_model.py
    cluster_model.py  (KMeans/GMM)  adaboost_model.py  tensorflow_dnn_model.py  (_tf_net.py helper)
  observability/
    tracking.py       MLflow logging for fit and inference
    plots.py          PR / ROC / confusion / importance figures
  experiments/
    runner.py         GridSearchCV-over-StratifiedKFold per model + blend + stacking
app/
  streamlit_app.py    dataset selection + overview
  pages/              1_Explore · 2_Importance · 3_Train_and_Compare · 4_Inference
scripts/
  make_sample_data.py  synthetic imbalanced demo dataset
  run_experiments.py   CLI: run the experiment suite on a CSV (boosters + blend/stacking)
  cli.py               CLI: train a single model or run inference on a CSV
  embedding_experiment.py  pre-train the NN embedding once, reuse frozen features
  experiment_top3.py   one-off: top-3-features experiment (kept for reference)
```

The model roster (see `models.ALL_MODELS`): the recommended boosters
(LightGBM/XGBoost/CatBoost), CatBoost focal, RandomForest, two logistic
regressions; plus models kept as comparison baselines / experiments —
LightGBM/XGBoost focal, clustering, AdaBoost, and the TensorFlow dense net.
`DEFAULT_MODELS` is the three recommended boosters. See
[EXPERIMENT_JOURNEY.md](EXPERIMENT_JOURNEY.md) for what was discarded and why.

### The generic wrapper

`ModelWrapper` wraps any estimator or pipeline (including a blend) so the rest of
the application is algorithm-agnostic. It exposes `fit`, `predict`,
`predict_proba` (with a `decision_function` fallback), `set_threshold`,
`metrics`, `validate`, `save`, and `load`. These are exactly the methods the
FastAPI service (`fastapi_app/`, see below) calls — `load()` at startup and
`predict_proba()` per request, with no modeling logic duplicated there.

### One file per algorithm

Each model module exposes `NAME`, `build(feature_mode, y)` returning an unfitted
sklearn/imbalanced-learn `Pipeline`, and `param_grid()` for GridSearch. A
registry (`models.ALL_MODELS`) lets the runner and UI stay generic. Imbalance is
handled inside each estimator — `scale_pos_weight` (LightGBM/XGBoost),
`auto_class_weights="Balanced"` (CatBoost), `class_weight="balanced"` (LogReg/RF)
— so no aggressive oversampling is needed.

## Feature pipeline

`feature_mode` is chosen per experiment and selects an optional **feature
engineering** layer plus an **encoding**:

- **`engineered`** (recommended default) — the deterministic `FeatureEngineer`
  prepends presence flags (`has_emails`, `has_links`, ...), log transforms
  (`log_*`) and ratios (`lexical_diversity`, `spelling_rate`, ...) to the raw
  counts. Data analysis showed the raw counts are heavy-tailed and zero-inflated,
  so *presence* carries more signal than magnitude; this lifted PR-AUC ~33% over
  raw (see `RESULTS.md`). The transform is row-wise and target-free, so it is
  leakage-safe by construction.
- **`raw`** — integer counts straight into the model (boosters bin internally).
- **`binned_woe`** — `OptimalBinningWOE`: supervised optimal binning → WOE; the
  most explainable representation and the natural input to `logreg_woe` (which
  always uses WOE regardless of the flag).
- **`quantile`** — `QuantileBinner` (`KBinsDiscretizer`), unsupervised ordinal bins.
- **`target`** — `CrossFitTargetEncoder`, out-of-fold mean-target encoding.
- **`autoencoder`** — `AutoencoderEncoder`, CPU MLP latent features (experimental).
- **`engineered_<enc>`** — feature engineering followed by any encoding above.

Supervised encoders (WOE, target) are **leakage-safe**: fit inside each CV fold
(WOE) or via out-of-fold statistics (target), applying learned mappings only at
transform time. On this data, `engineered` on raw counts beat every pre-encoding
variant, so it is the default; the encodings remain selectable for validation.

## Recommended default

Based on the experiments: **gradient boosters (LightGBM / CatBoost / XGBoost)
with class weighting, on `engineered` features, combined with a weighted blend,
and a threshold chosen for the operating cost (recall-target or cost-sensitive).**
On the **full dataset**, augmenting the engineered features with the frozen NN
embedding (pre-trained once on the train split) gives a small extra lift (blend
0.31 → 0.33). The full roster and all encodings/post-steps stay available for
comparison but are not run by default. See [EXPERIMENT_JOURNEY.md](EXPERIMENT_JOURNEY.md)
for the evidence behind these choices and what was discarded.

## Training / experiment loop

For each selected model the runner:

1. builds the pipeline for the chosen `feature_mode`;
2. runs **`GridSearchCV` over `StratifiedKFold`** (default 5 folds) scored by
   PR-AUC (the refit metric);
3. **calibrates** the best estimator's probabilities on the validation set
   (`CalibratedClassifierCV` over a `FrozenEstimator`) so blending and thresholds
   act on trustworthy probabilities;
4. **tunes the decision threshold** on validation (recall-target with precision
   floor / max-F1 / manual);
5. computes validation and test metrics;
6. logs params, CV/val/test metrics, PR/ROC/confusion curves, and WOE tables to
   MLflow.

An optional **blend** averages the calibrated base-model probabilities with
weights optimised on validation against PR-AUC; an optional **stacking**
meta-model (logistic regression over the base validation probabilities) is also
available. Both flow through the same threshold/metrics/persistence path as a
regular model, and the blend is packaged as a serializable `BlendModel` so the
winning blend can be saved and loaded like any single version.

## Metrics

Reported set (positive/phishing class focus): **PR-AUC (primary)**, ROC-AUC,
precision, recall, F1, MCC, and the confusion matrix at the chosen threshold.
Accuracy is deliberately excluded from the headline.

## Threshold tuning

The operating point is a business choice, evaluated on validation scores:

- **recall-target** — lowest threshold reaching recall ≥ X while precision ≥ Y
  (false negatives are costlier than false positives in phishing);
- **max-F1** — threshold maximising F1;
- **manual** — a user/CLI-supplied value, with live PR/recall/precision curves in
  the UI;
- **cost-sensitive** — minimises `fn_cost*FN + fp_cost*FP` (default FN = 10× FP),
  the most direct encoding of the business cost of a missed phishing email.

## Observability (MLflow)

Local file store (`./mlruns`); UI via
`MLFLOW_ALLOW_FILE_STORE=true uv run mlflow ui --backend-store-uri file:./mlruns`
(the env var opts out of MLflow's filesystem-backend deprecation, required to
start the UI server — training/inference writes to `./mlruns` regardless).
Override the backend with `MLFLOW_TRACKING_URI`.

- **Fit** — per model: hyperparameters and best combo, CV/validation/test
  metrics, the serialised model, PR/ROC/confusion figures, and per-feature
  binning/WOE tables.
- **Inference** — per batch: sample count, model version, threshold, predicted
  positive rate, and (evaluation mode) metrics against labels. No per-row data is
  logged.

## Persistence & versioning

Saving writes `models/<name>__<timestamp>/` containing `model.joblib` and
`metadata.json` (algorithm, feature list, threshold, feature mode, validation and
test metrics). Versions can be listed, compared and selected for inference
without retraining.

## Caching

The hyperparameter search and the NN embedding training are the expensive,
repeatable steps in the training pipeline (a cold full-dataset run takes
close to an hour — see [RESULTS.md](RESULTS.md#hyperparameter-and-embedding-caching)
for measured timings). Two on-disk caches in `src/phishing/core/` remove that
cost on a repeated run with an unchanged configuration:

- **`param_cache.py`** — caches each model's winning hyperparameters in
  `best_params/<hash>.json`, keyed by model name, feature mode, search
  method/budget, CV folds, and the training columns.
- **`embedding_cache.py`** — caches the fitted NN embedding (the Keras model,
  via joblib) in `embeddings/<hash>/`, keyed by every architecture
  hyperparameter plus the input feature columns and training-row count.

Both keys are deterministic hashes of their inputs, so any change to the
configuration they depend on (a different feature set, a wider search budget,
a different embedding width) naturally produces a different key and misses
the cache, rather than risking a stale result. Neither cache is consulted for
the decision threshold, which is applied after training and does not affect
either key. `scripts/best_model_report.py` and `run_experiments.py` use both
caches by default; pass `--force-search` / `--force-embedding-search` to
bypass them.

## REST API (`fastapi_app/`)

A FastAPI service serves the best-trained model over HTTP, reusing
`ModelWrapper`/`persistence.list_versions` exactly as described above — this
package contains no modeling logic of its own. At startup it loads the most
recently saved version in `models/` (or a specific one via the
`MODEL_VERSION_DIR` environment variable) and keeps it in memory for the
process lifetime. `POST /predict` accepts a batch of samples (the same eight
raw features as everywhere else in the framework) and returns, per sample, a
continuous `phishing_likelihood` in [0, 1] as the primary output, alongside a
convenience `is_phishing` flag at the model's tuned threshold. `GET /health`
and the OpenAPI docs (`/docs`, `/redoc`) are public; `/predict` requires a
fixed Bearer token (`API_TOKEN` environment variable). See
[`fastapi_app/README.md`](../fastapi_app/README.md) for how to run it and
authenticate.

## Two run modes

- **Evaluation mode** — input has the `label_1` column → predict **and** compute
  metrics.
- **Inference mode** — input has no labels → predictions only.

## Interfaces

- **Streamlit UI** — `uv run streamlit run app/streamlit_app.py`: load data →
  explore → rank features → train & compare → save versions → infer.
- **CLI (experiments)** — `uv run python scripts/run_experiments.py --csv <file>`:
  stratified 95/5 split, run the model suite, print the comparison, optionally
  save the best version.
- **CLI (train/infer)** — `uv run python scripts/cli.py train|infer ...`: headless
  single-model training or batch inference on a CSV from the command line.
- **CLI (embedding)** — `uv run python scripts/embedding_experiment.py --csv <file>`:
  pre-train the NN embedding once on the train split, freeze it, and evaluate
  boosters that reuse the frozen features (default 16-d).
- **REST API** — `uv run uvicorn fastapi_app.main:app --port 8000`: serves a
  saved model version over HTTP (`POST /predict`); see the "REST API" section
  above and [`fastapi_app/README.md`](../fastapi_app/README.md).

## Tech stack & a notable decision

Python **3.12** (not 3.13), uv, scikit-learn, imbalanced-learn, XGBoost,
LightGBM, CatBoost, optbinning, **TensorFlow (CPU)**, MLflow, joblib, FastAPI
and uvicorn (the REST API). **uv is the project standard — pip is not used.**
Python was pinned to 3.12 because
`optbinning` 0.21 depends on `ortools`, which currently ships no Python 3.13
wheels; this is the only deviation from the original 3.13 target in the project
guidance.

## Verification

- `uv run pytest` — end-to-end smoke tests (split → CV/GridSearch → calibrate →
  blend/stacking → threshold → metrics → save → load → predict) plus coverage of
  the encodings, the new models (NN, logreg, cluster, AdaBoost), the NN embedding,
  smoothing/denoising, cost threshold, BlendModel round-trip, the two caches, and
  the FastAPI endpoints (`fastapi_app/tests/`, included via the same `pytest`
  invocation — `testpaths` covers both `tests/` and `fastapi_app/tests/`).
- `uv run python scripts/make_sample_data.py` — synthetic 1% dataset.
- `uv run python scripts/run_experiments.py --csv data/email_phishing_data.csv`
  — full suite on the real data.
- `uv run streamlit run app/streamlit_app.py` — UI boot (health endpoint 200).
