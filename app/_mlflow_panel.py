"""Shared MLflow-UI help panel for the Streamlit pages.

MLflow tracking writes to ``./mlruns`` automatically whenever a page runs with
"Log to MLflow" checked — but *viewing* and comparing those runs needs the
separate ``mlflow ui`` server, which Streamlit does not start on its own. This
module renders one consistent expandable panel with the exact command and a
live reachability check, so the instructions live in one place.
"""

from __future__ import annotations

import urllib.request

import streamlit as st

MLFLOW_UI_URL = "http://localhost:5000"


def _mlflow_ui_reachable(url: str = MLFLOW_UI_URL, timeout: float = 0.5) -> bool:
    """Best-effort check whether an MLflow UI server is listening on ``url``.

    Used only to decide whether the help panel should start expanded (likely
    not running yet) or collapsed (already up) — never blocks on failure.
    """
    try:
        with urllib.request.urlopen(url, timeout=timeout):
            return True
    except Exception:
        return False


def render_mlflow_help(expanded: bool | None = None) -> None:
    """Render the expandable "How to run the MLflow UI" panel.

    ``expanded`` defaults to auto-detecting reachability (expanded when the UI
    looks down, collapsed when it looks up); pass an explicit bool to override.
    """
    reachable = _mlflow_ui_reachable()
    if expanded is None:
        expanded = not reachable

    status = "🟢 detected at " + MLFLOW_UI_URL if reachable else "🔴 not detected"
    with st.expander(f"How to run the MLflow UI — status: {status}", expanded=expanded):
        st.markdown(
            "Training and inference always log to the local file store "
            "(`./mlruns`) when **Log to MLflow** is checked — that part needs "
            "nothing extra. But **comparing runs** (sorting by PR-AUC, "
            "overlaying PR curves, browsing past experiments) needs the MLflow "
            "UI server, which is a **separate process** you start once and "
            "leave running alongside Streamlit."
        )
        st.markdown("**1. In a separate terminal, from the project root:**")
        st.code(
            "MLFLOW_ALLOW_FILE_STORE=true uv run mlflow ui "
            "--backend-store-uri file:./mlruns",
            language="bash",
        )
        st.caption(
            "`MLFLOW_ALLOW_FILE_STORE=true` is required — MLflow treats the "
            "local filesystem backend as deprecated (\"maintenance mode\") and "
            "refuses to start the UI server without this opt-out. This only "
            "affects the `mlflow ui` process; logging itself is unaffected."
        )
        st.markdown(f"**2. Open [{MLFLOW_UI_URL}]({MLFLOW_UI_URL})** and pick the "
                     "`phishing-fit` experiment (training runs) or "
                     "`phishing-inference` (prediction batches).")
        st.markdown(
            "**3. Keep it running** — it serves runs from this Streamlit "
            "session, the CLIs (`run_experiments.py`, `best_model_report.py`, "
            "`cli.py train`), and the Colab notebook alike, since they all "
            "write to the same `./mlruns` store."
        )
        if not reachable:
            st.warning(
                "No MLflow UI server detected on localhost:5000 right now — "
                "runs are still being logged, you just can't browse/compare "
                "them until the server above is running."
            )
