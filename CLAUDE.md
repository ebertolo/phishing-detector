# CLAUDE.md

Project guidance for Claude Code — **general directives** that any future plan
and code must respect. The framework is built; for the architecture see
[docs/DESIGN.md](docs/DESIGN.md), for the experiment history and design decisions
see [docs/EXPERIMENT_JOURNEY.md](docs/EXPERIMENT_JOURNEY.md).

> **Language policy:** All source code, comments, docstrings, identifiers, log
> messages, UI strings, and documentation **must be written in English**. This
> guidance file is the only document written in Portuguese (for the maintainer);
> everything produced inside the project is English-only.

---

## 1. Project Purpose

This is a **validation framework for phishing-detection algorithms** on a
**highly imbalanced dataset** (~1% positive / phishing vs. ~99% negative).

Because of the imbalance, **plain accuracy is meaningless** and must never be
used as the headline metric. Evaluation always reports metrics suited to rare-
positive problems:

- **Precision, Recall, F1** (focus on the positive/phishing class)
- **PR-AUC (Average Precision)** — primary ranking metric for imbalance
- **ROC-AUC** (secondary)
- **Confusion matrix** at a chosen threshold
- **MCC** (Matthews correlation coefficient)
- Threshold-tuning support (the operating point is a deliberate choice, not a
  fixed 0.5)

The framework must let the maintainer **compare, ensemble, and pipeline**
multiple algorithms to find the best performer.

---

## 2. Application (Streamlit UI)

The deliverable is a **Streamlit** application. It must support the full loop:

1. **Select a dataset** — either from a known folder **or** by file upload.
2. **Explore features** — display the ranges / basic statistics of the main
   features.
3. **Assess feature importance** — run **PCA** or **RandomForest** (or
   equivalent) to evaluate the weight/contribution of each feature.
4. **Train / select models** — run one or more algorithms, compare them using
   the imbalance-aware metrics above.
5. **Persist the winning model** to disk (versioned — see §5).
6. **Inference** — load a chosen saved model version and predict over an input
   file containing **one or more samples**, showing results in a **DataFrame**
   and offering a **download** of the predictions.

The application must operate in **two modes**:

- **Evaluation mode** — input dataset has labels → run predictions **and**
  compute validation metrics.
- **Inference mode** — input has **no labels** → produce predictions only.

Version selection: the user can **save multiple model versions**, **choose which
version** to use for inference, and **run validation metrics** against any
selected version.

---

## 3. Code Architecture

### 3.1 Generic model wrapper

A **single class** must wrap any estimator so the rest of the app is agnostic to
the underlying algorithm:

- It accepts **either a single model or a pipeline of models** as a parameter.
- It exposes **generic `fit` and `predict`** methods that work uniformly for any
  wrapped model or pipeline.
- It also exposes `predict_proba` (or `decision_function` fallback),
  `validate` (compute the imbalance-aware metrics), `save`, and `load`.

The class is designed so methods can be **imported and served by a future
FastAPI application** — i.e. a clean, importable class with `fit`, `predict`,
`validate`, `metrics`, `save`, `load`, etc. **No Streamlit imports inside the
core class / model modules.** The UI layer and the model/core layer must stay
separated so the same classes serve both Streamlit now and FastAPI later.

### 3.2 One file per algorithm

**Each model / algorithm gets its own `.py` file.** Each such file exposes its
model factory/config through the common wrapper interface so it can be combined
into ensembles or pipelines.

### 3.3 Notebook-style cell execution

Methods must be written so they can be **executed individually** as cells using a
VS Code extension that emulates Jupyter notebook output (e.g. `# %%` cell
markers). Favor small, self-contained, independently runnable functions/methods
over deeply coupled scripts, so any step (load → explore → fit → validate →
predict) can be run and inspected in isolation.

---

## 4. Tech Stack

- **Python 3.12** (pinned — `optbinning`/`ortools` have no 3.13 wheels yet),
  inside a **container** (see §6). **uv** is the package manager — never pip.
- **Streamlit** for the UI.
- **scikit-learn** — pipelines, metrics, `KBinsDiscretizer`, calibration,
  cross-validation, RandomForest / LogisticRegression / AdaBoost / KMeans.
- **imbalanced-learn** — available, but resampling (SMOTE) was tested and **not
  used**: class weights + boosters outperform it here (see §9).
- **XGBoost / LightGBM / CatBoost** — gradient boosting (the recommended models).
- **optbinning** — optimal binning + WOE.
- **TensorFlow (CPU)** — dense net and the reusable NN embedding.
- **joblib** — versioned model persistence.

### Feature engineering / transformers (all leakage-safe, composable in the pipeline)

- **`FeatureEngineer`** (the recommended default `engineered` mode) — presence
  flags + log transforms + ratios; the biggest single lever on PR-AUC.
- `KBinsDiscretizer`, **optimal binning + WOE**, **target encoding with
  cross-fitting**, autoencoder, interactions, smoothing, NN embedding.

Encoders/binners fit on train folds only (cross-fitting where supervised).

---

## 5. Model Persistence & Versioning

- Trained models are **persisted to disk** and **loaded for inference**.
- Support **multiple saved versions**; the user can **choose which version** to
  load for inference or validation.
- Each saved version should carry **metadata** (algorithm/pipeline description,
  training timestamp, feature list, and the validation metrics achieved) so a
  version can be identified and compared without retraining.
- Persistence format: **joblib** artifacts in a dedicated models directory.

---

## 6. Containerization

- Provide a **container with Python 3.12** preloaded with Streamlit and all
  model libraries from §4 (Dockerfile + docker-compose; the MLflow UI runs as a
  second compose service).
- The container runs the Streamlit application as the primary process.
- Keep dependencies pinned/reproducible (uv).

---

## 7. Future FastAPI Service

Design every core method to be **servable by FastAPI later**:

- Core logic lives in importable classes (the §3.1 wrapper and per-algorithm
  modules), **not** inside Streamlit callbacks.
- Public methods: `fit`, `predict`, `predict_proba`, `validate`, `metrics`,
  `save`, `load`.
- No UI/framework coupling in the core layer.

---

## 8. Working Agreement for Claude

- **English only** for all code, comments, and docs (this file excepted).
- Treat **accuracy as a forbidden headline metric**; always report the imbalance-
  aware set from §1.
- Keep the **core/model layer free of Streamlit and FastAPI imports**.
- One algorithm per `.py` file; route everything through the generic wrapper.
- Write methods as **independently runnable** (notebook cell friendly).
- Guard against **data leakage** in all encoders/binners (cross-fitting / fit on
  train only).
- **uv only** (never pip); **one background experiment/training job at a time**.

---

## 9. Decisions that proved out (read before changing modeling)

Hard-won lessons from the experiments — respect these unless new evidence
overturns them. Full evidence in [docs/EXPERIMENT_JOURNEY.md](docs/EXPERIMENT_JOURNEY.md).

- **Feature engineering beats model/encoding tuning.** `FeatureEngineer`
  (flags + log + ratios) is the top lever; it is the default and the reason
  PR-AUC rose from ~0.21 (raw) to ~0.31 (engineered + blend).
- **Don't rebalance.** No SMOTE/oversampling — class weights + boosters + PR-AUC
  win under extreme imbalance.
- **`raw` counts beat pre-discretising encodings** (WOE/quantile/target) for these
  count features — boosters split better than pre-binned input. Encodings stay
  available for comparison, not as defaults.
- **High mutual information ≠ predictive value.** Weak features matter in
  combination; boosters are invariant to monotonic transforms (z-scores add
  nothing).
- **Never retrain a supervised feature generator per CV fold** (it leaks). Train
  it once on the train split and freeze it — this is how the NN embedding is used.
- **Judge data-hungry generators on the full dataset, not a sample** (the NN
  embedding hurts on 100k but helps on the full 524k).
- **Fix all seeds** (sklearn `random_state`, TensorFlow via
  `keras.utils.set_random_seed`) so results are reproducible.
- **Best result so far:** blend (LightGBM+XGBoost+CatBoost) on `engineered`
  features + frozen NN embedding, 90/5/5 split → **test PR-AUC ≈ 0.37**.
- **The threshold is a business choice** (recall-target / max-F1 / cost-sensitive),
  never a fixed 0.5; missed phishing (FN) usually costs more than a false alarm.
