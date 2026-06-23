# Phishing Detection — Validation Framework

This repository delivers a validation framework for phishing-detection
algorithms on a **highly imbalanced** dataset (~1% phishing). Plain accuracy is
never the headline; the framework reports **PR-AUC (primary)**, recall,
precision, F1, MCC, ROC-AUC and a confusion matrix at a deliberately chosen
operating point. The model's primary output is a continuous **likelihood of
phishing** (a probability in [0, 1]), not a forced binary label — every
interface below exposes that probability directly, with a binary flag at the
tuned threshold offered alongside it for convenience.

The project runs entirely on **CPU** by default — every dependency and
default configuration was chosen so the full loop (train, compare, save,
infer) works with no GPU. **GPU support was also implemented** for the parts
that benefit most from it (LightGBM/XGBoost/CatBoost training, the
TensorFlow NN embedding): it is detected automatically and used when
available, with no configuration needed, and falls back to CPU when it is
not — see [docs/colab_experiments.ipynb](docs/colab_experiments.ipynb) for a
Colab GPU walkthrough.

## What's delivered

| component | what it is |
|---|---|
| **Core package** `src/phishing/` | The reusable engine — feature engineering, models, training loop, metrics, persistence — with no UI/framework imports, so the same logic backs every interface below. See [docs/DESIGN.md](docs/DESIGN.md). |
| **Experiment scripts** `scripts/` | Headless CLIs (`run_experiments.py`, `cli.py`, `best_model_report.py`, ...) for running experiments from a console, a batch job, or CI/CD. See [Command-line usage](#command-line-usage-headless-no-ui) below. |
| **Streamlit application** `app/` | An interactive UI for Engineering, Data, or Product to explore data, compare models, tune thresholds, and run inference without writing code. See [Quickstart](#quickstart-uv-is-the-project-standard--no-pip) below. |
| **Google Colab notebook** `docs/colab_experiments.ipynb` | Runs the same experiments with easy access to a free GPU runtime, for faster iteration than a local CPU. See [Running on Google Colab](#running-on-google-colab-gpu) below. |
| **REST API** `fastapi_app/` | A FastAPI service serving the best-trained model, with fixed-token authentication and a public Swagger/ReDoc documentation page. See [REST API](#rest-api) below. |
| **MLflow** `./mlruns` | Experiment tracking — every training run and inference batch can be logged and compared. See [Experiment tracking with MLflow](#experiment-tracking-with-mlflow) below. |
| **Documentation** `docs/` | The technical design, the experiment journey (what was tried, kept, and discarded, and why), concrete results tables, and the prioritised backlog. See [Documentation](#documentation) below. |

## Architecture

```
src/phishing/
  core/          metrics · splits · thresholding · blending · blend_model · stacking · persistence · wrapper · data
  features/      engineering · binning(optimal+WOE) · kbins · target_encoding · autoencoder · interactions · smoothing · nn_embedding · selection
  models/        one file per algorithm (boosters, focal variants, logreg, RF, cluster, AdaBoost, TensorFlow DNN)
  observability/ MLflow tracking (fit + inference) · diagnostic plots
  experiments/   GridSearchCV-over-StratifiedKFold runner + blend + stacking
app/             Streamlit UI (streamlit_app.py + pages/)
fastapi_app/     REST API (main.py, auth.py, schemas.py, routers/) — see REST API below
```

The core/feature/model/experiment layers contain **no Streamlit or FastAPI
imports**, so the same `ModelWrapper` (`fit`, `predict`, `predict_proba`,
`validate`, `metrics`, `save`, `load`) backs the Streamlit app, the CLIs, and
the REST API identically — no modeling logic is duplicated in any of them.

## Modeling approach

- **Feature engineering first:** a deterministic, leakage-safe `FeatureEngineer`
  (presence flags + log transforms + ratios) is the biggest single lever — the
  default mode `engineered`. Several encodings (`raw`, WOE, quantile, target,
  autoencoder) and post-engineering steps (smoothing, denoising, NN embedding)
  are selectable per experiment for comparison.
- **Models (recommended):** LightGBM, XGBoost, CatBoost (class-weighted boosters).
  Also available: CatBoost focal, RandomForest, logistic regression (`logreg`
  and `logreg_woe`), and — kept as comparison baselines / experiments — clustering
  (KMeans/GMM), AdaBoost, LightGBM/XGBoost focal, and a TensorFlow dense net.
- **Imbalance:** class weights / `scale_pos_weight` per model, stratified splits
  and CV, PR-AUC as the refit metric — no aggressive oversampling (SMOTE hurts
  here; see `docs/EXPERIMENT_JOURNEY.md`).
- **Ensemble:** weighted **blend** (serializable via `BlendModel`) and logistic
  **stacking**; calibration is sigmoid or isotonic.
- **Threshold:** chosen on validation — recall-target with precision floor,
  max-F1, manual, or cost-sensitive (`fn_cost`/`fp_cost`).
- **NN embedding:** the dense net's embedding (default 16-d, set via
  `--embedding-dim`) can be pre-trained once on the train split and reused as
  frozen features — a small but real gain on the full dataset (see
  `scripts/embedding_experiment.py`).

See **[docs/EXPERIMENT_JOURNEY.md](docs/EXPERIMENT_JOURNEY.md)** for how the model
evolved, what was discarded, and why.

## Quickstart (uv is the project standard — no pip)

```bash
# 1. Install dependencies
uv sync

# 2. Generate a synthetic imbalanced sample dataset (optional)
uv run python scripts/make_sample_data.py

# 3. Launch the app
uv run streamlit run app/streamlit_app.py

# 4. In a second terminal, launch the MLflow UI (local file store ./mlruns)
MLFLOW_ALLOW_FILE_STORE=true uv run mlflow ui --backend-store-uri file:./mlruns
```

Then open the Streamlit app, load `data/sample_phishing.csv`, and walk the
sidebar pages: **Explore → Importance → Train & Compare → Inference**.

## Command-line usage (headless, no UI)

Everything works from the terminal. `uv sync` installs the package, so the
scripts import it directly. Two entry points cover the common needs:

### `run_experiments.py` — compare models on a dataset

```bash
# Recommended run: boosters + blend on engineered features, save the best version
uv run python scripts/run_experiments.py --csv data/email_phishing_data.csv --save-best
```

Useful flags (all optional):

| flag | what it does |
|---|---|
| `--models lightgbm xgboost catboost` | which models to run (default: the 3 boosters) |
| `--feature-mode engineered` | feature/encoding strategy (default `engineered`) |
| `--stacking` | also build a logistic stacking meta-model |
| `--threshold-mode cost --fn-cost 10 --fp-cost 1` | cost-sensitive operating point |
| `--sample 100000` | run on a stratified sample (fast iteration) |
| `--save-best` | persist the best model/blend as a version |

### `cli.py` — train one model, or run inference

```bash
# Train a single model and save a version
uv run python scripts/cli.py train --csv data/email_phishing_data.csv \
    --model lightgbm --feature-mode engineered --threshold-mode max_f1

# Predict with the latest saved version (add --version <dir> to pick one)
uv run python scripts/cli.py infer --csv data/email_phishing_data.csv --out predictions.csv
```

Inference auto-detects the mode: if the CSV has the `label_1` column it computes
metrics (evaluation mode), otherwise it outputs predictions only.

### Reproduce the best result

```bash
# Blend + XGBoost on engineered features + frozen NN embedding, 90/5/5 split,
# prints both confusion matrices on the clean 5% test holdout.
# Defaults to RandomizedSearch (--search random --n-iter 40), which is what
# reproduces the headline blend PR-AUC ~0.44; add --stacking for the meta-model.
uv run python scripts/best_model_report.py --csv data/email_phishing_data.csv
```

### Split convention

The experiment scripts use a **stratified** split — train / validation
(calibration + threshold) / test (clean holdout). The default is **90/5/5**
(`best_model_report.py`); `run_experiments.py` defaults to ≈76/19/5. Every
partition preserves the ~1.3% phishing rate. Pass `--val-fraction` /
`--test-fraction` to change the ratio.

## Experiment tracking with MLflow

Every training run can be logged to **MLflow** — params, CV/validation/test
metrics, PR/ROC/confusion curves, WOE tables, and the model artifact — so you can
compare experiments later in the MLflow UI. Inference batches log a summary
(sample count, model version, threshold, predicted positive rate, eval metrics).

Tracking uses a **local file store** at `./mlruns` by default (override with the
`MLFLOW_TRACKING_URI` env var). Runs land in two experiments: **`phishing-fit`**
(training) and **`phishing-inference`** (predictions).

**1 — Make sure runs are logged.** The CLIs that train log to MLflow by default:

```bash
# logs to MLflow (use --no-mlflow to turn off)
uv run python scripts/run_experiments.py --csv data/email_phishing_data.csv --save-best
uv run python scripts/cli.py train --csv data/email_phishing_data.csv --model lightgbm
```

The experiment scripts log **only when you pass `--mlflow`** (off by default so
quick checks don't clutter the store):

```bash
uv run python scripts/best_model_report.py  --csv data/email_phishing_data.csv \
    --search random --n-iter 40 --stacking --mlflow
uv run python scripts/embedding_experiment.py --csv data/email_phishing_data.csv --mlflow
uv run python scripts/per_model_embedding.py  --csv data/email_phishing_data.csv --mlflow
```

**2 — Open the UI to view and compare runs:**

```bash
MLFLOW_ALLOW_FILE_STORE=true uv run mlflow ui --backend-store-uri file:./mlruns
# then open http://localhost:5000
```

`MLFLOW_ALLOW_FILE_STORE=true` is required because MLflow treats the local
filesystem backend (`./mlruns`) as deprecated ("maintenance mode") and refuses
to start the UI server without this opt-out — this only affects the `mlflow ui`
process itself; training/inference runs are logged to `./mlruns` regardless.

In the UI: pick the `phishing-fit` experiment, sort runs by `test_pr_auc`, and use
the compare view to overlay PR curves and parameters across models/feature modes.

## Running on Google Colab (GPU)

**[docs/colab_experiments.ipynb](docs/colab_experiments.ipynb)** clones the repo
into a Colab GPU runtime and runs the same `scripts/*.py` CLIs documented above —
no separate Colab-only code path. LightGBM, XGBoost and CatBoost auto-detect the
GPU (`device`/`task_type` set automatically in `src/phishing/models/_common.py`'s
`gpu_available()`, used by `lightgbm_model.py` / `xgboost_model.py` /
`catboost_model.py`); TensorFlow (the NN embedding) uses any visible GPU with no
code change. CPU-only also works — every cell falls back automatically. The
notebook also includes a guide for editing an existing model's hyperparameters or
adding a brand-new one (`src/phishing/models/<name>.py` + register in
`models/__init__.py`), directly from a Colab cell.

## REST API

`fastapi_app/` serves the best-trained model over HTTP, reusing the same
`ModelWrapper` as the Streamlit app and the CLIs — no modeling logic lives in
the API itself.

```bash
# 1. Train and save at least one model version first (see Quickstart/CLI above).
# 2. Run the API.
uv run uvicorn fastapi_app.main:app --reload --port 8000
# 3. Open the interactive docs (public, no token needed):
#    http://localhost:8000/docs   (Swagger UI)
#    http://localhost:8000/redoc  (ReDoc)
```

`POST /predict` is the only protected endpoint: send
`Authorization: Bearer <token>`, where `<token>` is the `API_TOKEN`
environment variable (default `changeme-dev-token` for local development).
`GET /health` needs no authentication.

```bash
curl -X POST http://localhost:8000/predict \
    -H "Authorization: Bearer changeme-dev-token" \
    -H "Content-Type: application/json" \
    -d '{"samples": [{"num_words": 120, "num_unique_words": 80, "num_stopwords": 40, "num_links": 3, "num_unique_domains": 2, "num_email_addresses": 1, "num_spelling_errors": 2, "num_urgent_keywords": 1}]}'
```

The response's primary field is `phishing_likelihood` — a continuous
probability, not a forced binary label. See
**[fastapi_app/README.md](fastapi_app/README.md)** for the full walkthrough
(authentication details, picking a specific model version via
`MODEL_VERSION_DIR`, and running the API's own test suite).

## Model versioning

Saving a model writes `models/<name>__<timestamp>/` containing `model.joblib`
and `metadata.json` (algorithm, feature list, threshold, feature mode, and the
validation/test metrics). The Inference page lists versions and lets you pick
which one to predict or validate with.

## Documentation

- **[docs/EXPERIMENT_JOURNEY.md](docs/EXPERIMENT_JOURNEY.md)** — how the model
  evolved: every major experiment, what was kept vs discarded, and why (the
  lessons from extreme imbalance and weak count features).
- **[docs/DESIGN.md](docs/DESIGN.md)** — architecture, layers, feature pipeline,
  training loop, persistence.
- **[docs/ML_STACK.md](docs/ML_STACK.md)** — libraries, models, encodings,
  ensembling and threshold techniques.
- **[docs/RESULTS.md](docs/RESULTS.md)** — concrete comparison tables (encodings,
  models, ensembles, embedding, tuning, full-dataset results).
- **[docs/Next_Steps.md](docs/Next_Steps.md)** — prioritised improvement backlog
  (what's been tested, what's open) with impact × effort × risk.
- **[docs/colab_experiments.ipynb](docs/colab_experiments.ipynb)** — run the main
  experiments on a Google Colab GPU runtime, and a guide to modifying/adding models.
- **[fastapi_app/README.md](fastapi_app/README.md)** — REST API: how to run it
  locally, authenticate, call `/predict`, and run its test suite.

## Tests

```bash
uv run pytest
```

Runs the full suite, including `fastapi_app/tests/` (`pyproject.toml`'s
`testpaths` covers both `tests/` and `fastapi_app/tests/` in one invocation).
The API tests train a small model on synthetic data and do not require a
pre-trained version or a running server.
