"""Feature exploration: ranges and basic statistics of the main features."""

import _bootstrap  # noqa: F401

import streamlit as st

from phishing.core import data as data_mod

st.title("🔎 Explore features")

if "df" not in st.session_state:
    st.warning("Load a dataset on the main page first.")
    st.stop()

df = st.session_state["df"]
X = data_mod.feature_frame(df)

st.subheader("Feature statistics")
stats = X.describe().T
stats["dtype"] = X.dtypes.astype(str)
stats["n_unique"] = X.nunique()
st.dataframe(stats, width="stretch")

st.subheader("Per-feature distributions")
feature = st.selectbox("Feature", list(X.columns))
st.bar_chart(X[feature].value_counts().sort_index())

if st.session_state.get("has_labels"):
    _, y = data_mod.split_X_y(df)
    st.subheader("Class balance")
    st.write(f"Phishing rate: **{data_mod.positive_rate(y):.4%}**")
    st.bar_chart(y.value_counts().rename({0: "legit", 1: "phishing"}))
