"""TensorFlow dense neural network model (CPU).

A sklearn-compatible wrapper around the Keras dense network in ``_tf_net.py``.
The network outputs a tanh score in [-1, 1] mapped to a probability; the runner's
existing calibration step (sigmoid/isotonic) refines it. Imbalance is handled
with balanced class weights. Optimizers (SGD / RMSprop / Adam / Nadam) and
learning rate are compared through the GridSearch param grid.

Persistence: Keras models do not pickle cleanly through joblib, so the wrapper
implements ``__getstate__`` / ``__setstate__`` — the Keras model is written to /
read from a sidecar ``.keras`` file next to the joblib artifact (the path is
remembered in the pickled state). This integrates with the existing
``ModelWrapper.save`` / ``persistence.py`` without changing them.
"""

# %%
from __future__ import annotations

import os
import tempfile

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler

from ._common import make_pipeline
from ._tf_net import build_keras_model, class_weight_dict

NAME = "tensorflow_dnn"


# %%
class _TFDenseNet(BaseEstimator, ClassifierMixin):
    """Keras dense net with standardisation, early stopping and tanh->proba."""

    def __init__(
        self,
        optimizer: str = "adam",
        learning_rate: float = 0.05,
        epochs: int = 300,
        batch_size: int = 512,
        dropout: float = 0.4,
        patience: int = 20,
        random_state: int = 42,
    ) -> None:
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.patience = patience
        self.random_state = random_state

    # %%
    def fit(self, X, y):
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        from tensorflow import keras

        # Seed Python/NumPy/TensorFlow in one call so weight init, batch shuffling
        # and dropout are reproducible run-to-run.
        keras.utils.set_random_seed(self.random_state)

        X = np.asarray(X, dtype="float32")
        y = np.asarray(y).astype("float32")
        self.classes_ = np.unique(y.astype(int))
        self.scaler_ = StandardScaler()
        Xs = self.scaler_.fit_transform(X).astype("float32")

        self.model_ = build_keras_model(
            n_features=Xs.shape[1],
            optimizer=self.optimizer,
            learning_rate=self.learning_rate,
            dropout=self.dropout,
        )
        # Monitor PR-AUC (the metric that matters here) rather than loss, so
        # early stopping keeps the best-ranking epoch and the net does not stall
        # in a collapsed all-positive state on rare-class folds.
        es = keras.callbacks.EarlyStopping(
            monitor="pr_auc", mode="max", patience=self.patience,
            restore_best_weights=True,
        )
        self.model_.fit(
            Xs, y,
            epochs=min(self.epochs, 2000),  # spec cap of 2000 iterations
            batch_size=self.batch_size,
            class_weight=class_weight_dict(y),
            callbacks=[es],
            verbose=0,
        )
        return self

    # %%
    def predict_proba(self, X):
        X = np.asarray(X, dtype="float32")
        Xs = self.scaler_.transform(X).astype("float32")
        score = self.model_.predict(Xs, verbose=0).ravel()  # tanh score in [-1, 1]
        p = np.clip((score + 1.0) / 2.0, 0.0, 1.0)           # map to probability
        return np.column_stack([1 - p, p])

    # %%
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    # %%
    def __getstate__(self):
        """Pickle everything except the Keras model, which is stored separately.

        The Keras model is serialised to bytes of a ``.keras`` file and kept in
        the state, so the whole wrapper still round-trips through a single joblib
        artifact (no external path management needed by callers).
        """
        state = self.__dict__.copy()
        model = state.pop("model_", None)
        if model is not None:
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "m.keras")
                model.save(path)
                with open(path, "rb") as fh:
                    state["_model_bytes"] = fh.read()
        return state

    # %%
    def __setstate__(self, state):
        model_bytes = state.pop("_model_bytes", None)
        self.__dict__.update(state)
        if model_bytes is not None:
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
            from tensorflow import keras

            from ._tf_net import _tanh_bce

            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "m.keras")
                with open(path, "wb") as fh:
                    fh.write(model_bytes)
                self.model_ = keras.models.load_model(
                    path, safe_mode=False, custom_objects={"_tanh_bce": _tanh_bce}
                )


# %%
def build(feature_mode: str = "engineered", y=None, embedding_kwargs: dict | None = None):
    """Unfitted TensorFlow dense-net pipeline for the given feature mode."""
    return make_pipeline(_TFDenseNet(), feature_mode, embedding_kwargs)


# %%
def param_grid() -> dict:
    """Grid comparing optimizers and learning rates (epochs capped by early stop)."""
    return {
        "model__optimizer": ["sgd", "rmsprop", "adam", "nadam"],
        "model__learning_rate": [0.1, 0.05],
        "model__epochs": [200],
    }
