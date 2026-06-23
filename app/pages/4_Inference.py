"""Inference: load a saved model version and predict over an input file.

Two modes, decided by whether the input file carries the ``label_1`` column:

- Evaluation mode (labels present) : predict **and** compute imbalance-aware metrics.
- Inference mode  (no labels)      : produce predictions only.

A per-batch summary is logged to MLflow. Predictions are downloadable as CSV.
"""

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd
import streamlit as st

from phishing.core import data as data_mod
from phishing.core.metrics import compute_metrics
from phishing.core.persistence import list_versions
from phishing.core.wrapper import ModelWrapper
from phishing.observability import tracking

st.title("🚀 Inference")

versions = list_versions()
if not versions:
    st.warning("No saved model versions yet. Train and save one on the Train & Compare page.")
    st.stop()

labels = [
    f"{v['metadata'].get('name', '?')} · {v['metadata'].get('created_at', '?')} "
    f"· test PR-AUC={v['metadata'].get('extra', {}).get('test_metrics', {}).get('pr_auc', float('nan')):.3f}"
    for v in versions
]
idx = st.selectbox("Model version", range(len(versions)), format_func=lambda i: labels[i])
selected = versions[idx]
meta = selected["metadata"]

with st.expander("Version metadata"):
    st.json(meta)

st.subheader("Input file")
use_loaded = st.checkbox("Use the dataset loaded on the main page", value="df" in st.session_state)
input_df = None
if use_loaded and "df" in st.session_state:
    input_df = st.session_state["df"]
else:
    up = st.file_uploader("CSV with one or more samples", type=["csv"])
    if up is not None:
        input_df = data_mod.load_csv(up)

threshold = st.slider(
    "Decision threshold",
    0.0,
    1.0,
    float(meta.get("threshold", 0.5)),
    0.01,
    help="Defaults to the threshold saved with this version; adjust to re-tune the operating point.",
)

if input_df is not None and st.button("Run prediction", type="primary"):
    wrapper = ModelWrapper.load(selected["path"])
    X = data_mod.feature_frame(input_df)
    proba = wrapper.predict_proba(X)
    pred = (proba >= threshold).astype(int)

    out = input_df.copy()
    out["phishing_proba"] = proba
    out["phishing_pred"] = pred

    st.subheader("Predictions")
    st.dataframe(out.head(100), width="stretch")

    eval_metrics = None
    if data_mod.has_labels(input_df):
        _, y = data_mod.split_X_y(input_df)
        m = compute_metrics(np.asarray(y), proba, threshold)
        eval_metrics = m.as_dict()
        st.subheader("Evaluation metrics")
        cols = st.columns(5)
        cols[0].metric("PR-AUC", f"{m.pr_auc:.3f}")
        cols[1].metric("Recall", f"{m.recall:.3f}")
        cols[2].metric("Precision", f"{m.precision:.3f}")
        cols[3].metric("F1", f"{m.f1:.3f}")
        cols[4].metric("MCC", f"{m.mcc:.3f}")
        st.write("Confusion matrix [[tn, fp], [fn, tp]]:")
        st.write(m.confusion)

    # Per-batch MLflow summary (no per-row data).
    version_name = f"{meta.get('name', 'model')}__{meta.get('created_at', '')}"
    try:
        tracking.log_inference(
            model_version=version_name,
            n_samples=len(out),
            threshold=threshold,
            predicted_positive_rate=float(pred.mean()),
            eval_metrics=eval_metrics,
        )
        st.caption("Logged batch summary to MLflow.")
    except Exception as exc:  # pragma: no cover - logging must not block predictions
        st.caption(f"MLflow logging skipped: {exc}")

    st.download_button(
        "⬇️ Download predictions CSV",
        out.to_csv(index=False).encode("utf-8"),
        file_name="predictions.csv",
        mime="text/csv",
    )
