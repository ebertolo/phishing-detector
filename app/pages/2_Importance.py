"""Feature-importance assessment via RandomForest and mutual information."""

import _bootstrap  # noqa: F401

import pandas as pd
import streamlit as st

from phishing.core import data as data_mod
from phishing.features import selection

st.title("📊 Feature importance")

if "df" not in st.session_state:
    st.warning("Load a dataset on the main page first.")
    st.stop()

if not st.session_state.get("has_labels"):
    st.warning("Feature importance needs labels (evaluation-mode dataset).")
    st.stop()

df = st.session_state["df"]
X, y = data_mod.split_X_y(df)

method = st.radio("Method", ["RandomForest importance", "Mutual information"], horizontal=True)

if st.button("Compute"):
    with st.spinner("Computing ..."):
        if method.startswith("RandomForest"):
            scores = selection.random_forest_importance(X, y)
        else:
            scores = selection.mutual_information(X, y)
    st.session_state["importance"] = scores

if "importance" in st.session_state:
    scores = st.session_state["importance"]
    st.subheader("Ranking")
    st.dataframe(scores.to_frame(), width="stretch")
    st.bar_chart(scores)
