"""Streamlit entry point — UI layer only.

All logic lives in the importable ``phishing`` core/feature/model/experiment
layers; this file (and the pages) only orchestrate widgets and render results.
The dataset chosen here is shared across pages via ``st.session_state``.
"""

# %%
import _bootstrap  # noqa: F401  -- adds ../src to sys.path when run from repo root

from pathlib import Path

import pandas as pd
import streamlit as st
from _mlflow_panel import render_mlflow_help

from phishing.core import data as data_mod

DATA_DIR = Path("data")

st.set_page_config(page_title="Phishing Detection Framework", layout="wide")


def _load_into_session(df: pd.DataFrame, source_name: str) -> None:
    """Store the loaded dataframe and its mode in session state."""
    st.session_state["df"] = df
    st.session_state["source_name"] = source_name
    st.session_state["has_labels"] = data_mod.has_labels(df)


def main() -> None:
    st.title("📧 Phishing Detection — Validation Framework")
    st.caption(
        "Imbalanced (~1% phishing). Headline metric is PR-AUC, never accuracy. "
        "Load training data, explore, rank features, train & compare models, "
        "save versions, and run inference."
    )

    st.header("1 · Select a dataset")
    col_upload, col_folder = st.columns(2)

    with col_upload:
        st.subheader("Upload a CSV")
        uploaded = st.file_uploader("CSV file", type=["csv"])
        if uploaded is not None:
            df = data_mod.load_csv(uploaded)
            _load_into_session(df, uploaded.name)
            st.success(f"Loaded {uploaded.name} ({len(df):,} rows).")

    with col_folder:
        st.subheader("Pick from data/ folder")
        DATA_DIR.mkdir(exist_ok=True)
        csvs = sorted(p.name for p in DATA_DIR.glob("*.csv"))
        if csvs:
            choice = st.selectbox("Available files", ["—"] + csvs)
            if choice != "—" and st.button("Load selected"):
                df = data_mod.load_csv(DATA_DIR / choice)
                _load_into_session(df, choice)
                st.success(f"Loaded {choice} ({len(df):,} rows).")
        else:
            st.info("No CSV files found in ./data — drop files there or upload.")

    if "df" in st.session_state:
        df = st.session_state["df"]
        mode = "Evaluation (labels present)" if st.session_state["has_labels"] else "Inference (no labels)"
        st.divider()
        st.metric("Rows", f"{len(df):,}")
        st.write(f"**Mode:** {mode}")
        if st.session_state["has_labels"]:
            X, y = data_mod.split_X_y(df)
            st.write(f"**Phishing rate:** {data_mod.positive_rate(y):.4%}")
        st.dataframe(df.head(20), width="stretch")
        st.info("Use the pages in the sidebar: Explore → Importance → Train & Compare → Inference.")
    else:
        st.warning("Load a dataset to begin.")

    st.divider()
    render_mlflow_help()


if __name__ == "__main__":
    main()
