"""Train, cross-validate, compare, threshold-tune and save models."""

import os

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd
import streamlit as st
from _mlflow_panel import _mlflow_ui_reachable, render_mlflow_help

from phishing.core import data as data_mod
from phishing.core.splits import stratified_split
from phishing.experiments.runner import ThresholdConfig, run_experiments
from phishing.models import ALL_MODELS, DEFAULT_MODELS
from phishing.models._common import FEATURE_MODES

st.title("🧪 Train & Compare")

if "df" not in st.session_state:
    st.warning("Load a dataset on the main page first.")
    st.stop()

if not st.session_state.get("has_labels"):
    st.warning("Training needs labels (evaluation-mode dataset).")
    st.stop()

df = st.session_state["df"]
X, y = data_mod.split_X_y(df)

st.subheader("Compute")
gpu_col, _ = st.columns(2)
with gpu_col:
    use_gpu = st.checkbox(
        "Use GPU when available", value=True,
        help="LightGBM/XGBoost/CatBoost switch to GPU training automatically when "
        "a real GPU backend is detected (XGBoost device=cuda, CatBoost "
        "task_type=GPU; LightGBM only if this install has a true CUDA build — "
        "the default PyPI wheel does not, and falls back to CPU correctly). "
        "Uncheck to force CPU everywhere (sets PHISHING_FORCE_CPU=1), e.g. to "
        "reproduce the CPU numbers in docs/RESULTS.md.",
    )
    os.environ["PHISHING_FORCE_CPU"] = "0" if use_gpu else "1"

st.subheader("Models")
chosen = st.multiselect(
    "Algorithms to run (hyperparameter search + StratifiedKFold)",
    options=list(ALL_MODELS.keys()),
    default=DEFAULT_MODELS,
    help="Default = recommended gradient boosters. Add others to compare.",
)

st.subheader("Ensembling")
col_ens1, col_ens2 = st.columns(2)
with col_ens1:
    build_blend = st.checkbox(
        "Build weighted blend", value=True,
        help="Weighted average of calibrated base-model probabilities, weights "
        "optimised on validation. The usual winner — see docs/EXPERIMENT_JOURNEY.md.",
    )
with col_ens2:
    build_stacking = st.checkbox(
        "Build logistic stacking", value=False,
        help="A logistic-regression meta-model over base-model probabilities. "
        "~Equal PR-AUC to the blend, sometimes a touch more recall.",
    )
if not chosen:
    st.caption("Select at least one model above.")
elif len(chosen) < 2 and (build_blend or build_stacking):
    st.caption("Blend/stacking need at least 2 models — select more to enable them.")

col1, col2 = st.columns(2)
with col1:
    feature_mode = st.selectbox(
        "Feature / encoding strategy (per experiment)",
        FEATURE_MODES,
        index=FEATURE_MODES.index("engineered"),
        help="engineered (recommended) = presence flags + log + ratios; "
        "raw = integer counts; binned_woe = optimal-binning + WOE; "
        "quantile = KBins; target = cross-fit target encoding; "
        "autoencoder = CPU MLP latent features. Prefix 'engineered_' combines "
        "engineering with an encoding. 'engineered_nnembed' appends the NN "
        "embedding (see below). logreg_woe always uses WOE.",
    )
    calibration = st.selectbox("Calibration method", ["sigmoid", "isotonic"])
with col2:
    n_splits = st.slider(
        "CV folds", 3, 10, 3,
        help="StratifiedKFold splits used inside the hyperparameter search on "
        "the training set. Default 3 matches the project's best-model report "
        "(faster; 5 is the framework's general default and reduces variance).",
    )
    log_mlflow = st.checkbox(
        "Log to MLflow", value=True,
        help="Writes this run to ./mlruns. The MLflow UI server must be running "
        "separately to browse/compare logged runs — see the panel near the "
        "bottom of this page.",
    )

uses_nn_embedding = feature_mode.endswith("nnembed")
embedding_kwargs = None
if uses_nn_embedding:
    st.subheader("NN embedding architecture")
    st.caption(
        "Input layer width is fixed by the raw + engineered feature count "
        "(not configurable here). Layer widths and dropout below apply only "
        "because 'engineered_nnembed' is selected; the embedding retrains "
        "inside every CV fold for this mode (leakage-safe, but not cached — "
        "see scripts/best_model_report.py for the once-trained, cached variant)."
    )
    ec1, ec2, ec3 = st.columns(3)
    with ec1:
        hidden1_dim = st.number_input("Layer 1 width (Dense)", 8, 256, 40, 4)
        dropout1 = st.slider("Layer 1 dropout", 0.0, 0.9, 0.4, 0.05)
    with ec2:
        hidden2_dim = st.number_input("Layer 2 width (Dense)", 8, 256, 20, 4)
        dropout2 = st.slider("Layer 2 dropout", 0.0, 0.9, 0.4, 0.05)
    with ec3:
        embedding_dim = st.number_input(
            "Embedding (output) width", 4, 128, 20, 4,
            help="Width of the reusable embedding layer fed to the boosters.",
        )
        patience = st.number_input(
            "Early-stop patience", 5, 500, 50, 5,
            help="Epochs without a val_pr_auc improvement before stopping.",
        )
    embedding_kwargs = dict(
        hidden1_dim=hidden1_dim, hidden2_dim=hidden2_dim,
        dropout1=dropout1, dropout2=dropout2,
        embedding_dim=embedding_dim, patience=patience,
    )

st.subheader("Hyperparameter search")
col3, col4 = st.columns(2)
with col3:
    search_method = st.radio(
        "Method", ["random", "grid"], horizontal=True,
        help="random (recommended) = RandomizedSearchCV over a wider space; the "
        "single biggest PR-AUC lever after feature engineering (+0.065 in "
        "docs/EXPERIMENT_JOURNEY.md). grid = exhaustive search over a small grid.",
    )
    n_iter = 40
    if search_method == "random":
        n_iter = st.slider("Random search iterations", 5, 100, 40, 5)
with col4:
    force_search = st.checkbox(
        "Force re-search (ignore cached best params)", value=False,
        help="A winning hyperparameter combination is cached per (model, "
        "feature mode, search config, embedding architecture, training columns) "
        "in best_params/*.json and reused automatically on a matching re-run. "
        "Check this to ignore the cache and search again.",
    )

st.subheader("Threshold (operating point on validation)")
thr_mode = st.radio("Mode", ["recall_target", "max_f1", "manual", "cost"], horizontal=True)
recall_target, precision_floor, manual_value = 0.90, 0.30, 0.5
fn_cost, fp_cost = 10.0, 1.0
if thr_mode == "recall_target":
    c1, c2 = st.columns(2)
    recall_target = c1.slider("Recall target", 0.50, 0.99, 0.90, 0.01)
    precision_floor = c2.slider("Precision floor", 0.0, 0.90, 0.30, 0.01)
elif thr_mode == "manual":
    manual_value = st.slider("Manual threshold", 0.0, 1.0, 0.5, 0.01)
elif thr_mode == "cost":
    c1, c2 = st.columns(2)
    fn_cost = c1.number_input(
        "False-negative cost (fn)", 1.0, 200.0, 10.0, 1.0,
        help="How many times costlier a missed phishing email (FN) is than a "
        "false alarm (FP). Higher fn -> lower threshold -> more phishing caught, "
        "more false alarms. PR-AUC is unchanged; only the FP/FN split moves.",
    )
    fp_cost = c2.number_input("False-positive cost (fp)", 1.0, 200.0, 1.0, 1.0)

st.subheader("Data split")
st.caption(
    "**Train (90%)** fits the models and the hyperparameter search. "
    "**Validation (5%)** calibrates probabilities, tunes the decision threshold, "
    "and optimises blend weights. **Test (5%)** is a clean, untouched holdout "
    "used only once, to report the final confusion matrix and metrics."
)
split_default = st.checkbox("Use the recommended 90% / 5% / 5% split", value=True)
if split_default:
    val_size, test_size = 0.05, 0.05
else:
    c1, c2 = st.columns(2)
    val_size = c1.slider("Validation %", 0.02, 0.30, 0.05, 0.01)
    test_size = c2.slider("Test %", 0.02, 0.30, 0.05, 0.01)
    st.caption(f"Train = {1 - val_size - test_size:.0%} / Validation = {val_size:.0%} / Test = {test_size:.0%}")

if st.button("Run experiments", type="primary", disabled=not chosen):
    split = stratified_split(X, y, val_size=val_size, test_size=test_size)
    st.caption(f"Split positive rates: {split.positive_rates()}")
    thr_cfg = ThresholdConfig(
        mode=thr_mode,
        recall_target=recall_target,
        precision_floor=precision_floor,
        manual_value=manual_value,
        fn_cost=fn_cost,
        fp_cost=fp_cost,
    )
    status = st.empty()
    with st.spinner("Training ..."):
        results, ensembles = run_experiments(
            split,
            model_names=chosen,
            feature_mode=feature_mode,
            threshold_cfg=thr_cfg,
            build_blend=build_blend,
            build_stacking=build_stacking,
            calibration=calibration,
            n_splits=n_splits,
            log_mlflow=log_mlflow,
            search_method=search_method,
            n_iter=n_iter,
            force_search=force_search,
            embedding_kwargs=embedding_kwargs,
            progress=lambda m: status.write(m),
        )
    status.empty()
    st.session_state["results"] = results
    st.session_state["ensembles"] = ensembles or {}
    st.session_state["split"] = split

    # Append a snapshot to this session's run history for later comparison —
    # one row per model/ensemble produced by this click of "Run experiments".
    history = st.session_state.setdefault("run_history", [])
    run_ts = pd.Timestamp.now().strftime("%H:%M:%S")
    run_label = f"{run_ts} · {feature_mode} · {search_method}"
    for r in results:
        history.append(
            {
                "run": run_label, "model": r.name,
                "test_pr_auc": r.test_metrics["pr_auc"],
                "test_recall": r.test_metrics["recall"],
                "test_precision": r.test_metrics["precision"],
                "test_f1": r.test_metrics["f1"],
                "test_mcc": r.test_metrics["mcc"],
                "threshold": r.threshold, "threshold_mode": thr_mode,
            }
        )
    for ens_name, info in (ensembles or {}).items():
        history.append(
            {
                "run": run_label, "model": ens_name,
                "test_pr_auc": info["test_metrics"]["pr_auc"],
                "test_recall": info["test_metrics"]["recall"],
                "test_precision": info["test_metrics"]["precision"],
                "test_f1": info["test_metrics"]["f1"],
                "test_mcc": info["test_metrics"]["mcc"],
                "threshold": info["threshold"], "threshold_mode": thr_mode,
            }
        )

if "results" in st.session_state:
    results = st.session_state["results"]
    ensembles = st.session_state.get("ensembles", {})

    rows = []
    for r in results:
        rows.append(
            {
                "model": r.name,
                "cv_pr_auc": r.cv_pr_auc,
                "val_pr_auc": r.val_metrics["pr_auc"],
                "val_recall": r.val_metrics["recall"],
                "val_precision": r.val_metrics["precision"],
                "val_f1": r.val_metrics["f1"],
                "val_mcc": r.val_metrics["mcc"],
                "test_pr_auc": r.test_metrics["pr_auc"],
                "test_recall": r.test_metrics["recall"],
                "test_precision": r.test_metrics["precision"],
                "threshold": r.threshold,
            }
        )
    for ens_name, info in ensembles.items():
        rows.append(
            {
                "model": ens_name,
                "cv_pr_auc": np.nan,
                "val_pr_auc": info["val_metrics"]["pr_auc"],
                "val_recall": info["val_metrics"]["recall"],
                "val_precision": info["val_metrics"]["precision"],
                "val_f1": info["val_metrics"]["f1"],
                "val_mcc": info["val_metrics"]["mcc"],
                "test_pr_auc": info["test_metrics"]["pr_auc"],
                "test_recall": info["test_metrics"]["recall"],
                "test_precision": info["test_metrics"]["precision"],
                "threshold": info["threshold"],
            }
        )

    table = pd.DataFrame(rows).sort_values("test_pr_auc", ascending=False)
    st.subheader("Comparison (sorted by test PR-AUC)")
    st.dataframe(table, width="stretch")
    if "blend" in ensembles:
        b = ensembles["blend"]
        st.caption(f"Blend weights: {dict(zip(b['names'], np.round(b['weights'], 3)))}")
    if "stacking" in ensembles:
        st.caption(f"Stacking meta-coefficients: {ensembles['stacking']['meta_coefficients']}")

    st.subheader("Confusion matrix & key scores (held-out test)")
    name_to_metrics = {r.name: r.test_metrics for r in results}
    name_to_metrics.update({n: info["test_metrics"] for n, info in ensembles.items()})
    detail_pick = st.selectbox("Model", list(name_to_metrics.keys()), key="detail_pick")
    m = name_to_metrics[detail_pick]
    score_cols = st.columns(5)
    score_cols[0].metric("PR-AUC", f"{m['pr_auc']:.3f}")
    score_cols[1].metric("Recall", f"{m['recall']:.3f}")
    score_cols[2].metric("Precision", f"{m['precision']:.3f}")
    score_cols[3].metric("F1", f"{m['f1']:.3f}")
    score_cols[4].metric("MCC", f"{m['mcc']:.3f}")
    cm_col, _ = st.columns(2)
    with cm_col:
        from phishing.observability import plots

        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        st.pyplot(plots.confusion_figure(cm))
        st.caption(
            f"TN={m['tn']:,} FP={m['fp']:,} FN={m['fn']:,} TP={m['tp']:,} "
            f"· n={m['n_samples']:,} ({m['n_positives']:,} phishing)"
        )

    st.subheader("Save a model version")
    saveable = [r.name for r in results]
    pick = st.selectbox("Model to persist", saveable)
    if st.button("💾 Save selected model"):
        chosen_result = next(r for r in results if r.name == pick)
        path = chosen_result.wrapper.save(
            cv_pr_auc=chosen_result.cv_pr_auc,
            val_metrics=chosen_result.val_metrics,
            test_metrics=chosen_result.test_metrics,
            best_params=chosen_result.best_params,
        )
        st.success(f"Saved version → {path}")

st.divider()
st.subheader("Compare with previous experiments")

hist_tab, mlflow_tab = st.tabs(["This session", "MLflow (all runs)"])
with hist_tab:
    history = st.session_state.get("run_history", [])
    if not history:
        st.caption("No runs yet this session — click \"Run experiments\" above to start.")
    else:
        hist_df = pd.DataFrame(history).sort_values("test_pr_auc", ascending=False)
        st.dataframe(hist_df, width="stretch")
        if st.button("Clear session history"):
            st.session_state["run_history"] = []
            st.rerun()
with mlflow_tab:
    st.caption(
        "Every run logged with \"Log to MLflow\" checked lands in the "
        "`phishing-fit` experiment, including ones from other sessions, the "
        "CLIs, and the Colab notebook — the full cross-session history. "
        "**This comparison needs the MLflow UI server running separately** — "
        "it is not started automatically by Streamlit."
    )
    render_mlflow_help(expanded=not _mlflow_ui_reachable())
