"""Observability layer: MLflow experiment and inference tracking.

Depends only on ``mlflow`` (no Streamlit/FastAPI), so it can be called from the
core experiment runner, the Streamlit UI, or a future API alike.
"""
