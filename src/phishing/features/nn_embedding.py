"""Neural-network embedding transformer (reusable 20-d representation).

Trains the same dense network used by ``tensorflow_dnn`` (on the training data
only, so it is leakage-safe) and exposes the 20 activations of its named
``embedding`` layer as new features ``nn_0 .. nn_19``. Any model can then consume
this embedding via the ``engineered_nnembed`` feature mode, "using the last layer
as an embedding / 20 extra features for the other models" as requested.

TensorFlow is imported lazily. The embedding sub-model and scaler are stored on
the instance; like the DNN model, the Keras object is serialised to bytes for a
clean joblib round-trip.
"""

# %%
from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

from ..models._tf_net import (
    EMBEDDING_LAYER,
    build_keras_model,
    class_weight_dict,
)


# %%
def _momentum_ramp_callback(optimizer, start, end, total_epochs):
    """Build a Keras callback that linearly ramps SGD momentum start -> end."""
    from tensorflow import keras

    class _MomentumRamp(keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            frac = min(epoch / max(total_epochs - 1, 1), 1.0)
            m = start + (end - start) * frac
            # Keras 3 SGD exposes momentum as an attribute.
            try:
                optimizer.momentum = float(m)
            except Exception:
                pass

    return _MomentumRamp()


# %%
class NNEmbedding(BaseEstimator, TransformerMixin):
    """Train the dense net and emit its 20-d embedding-layer activations.

    Parameters
    ----------
    optimizer, learning_rate, epochs, batch_size : passed to the network.
    keep_raw : bool
        Keep the input columns alongside the ``nn_*`` embedding columns.
    """

    def __init__(
        self,
        optimizer: str = "adam",
        learning_rate: float = 0.05,
        epochs: int = 150,
        batch_size: int = 256,
        keep_raw: bool = True,
        momentum_schedule: bool = False,
        patience: int = 50,
        random_state: int = 42,
        embedding_dim: int = 20,
        periodic: bool = False,
        cosine_schedule: bool = False,
        dropout: float = 0.4,
        hidden1_dim: int = 40,
        hidden2_dim: int = 20,
        dropout1: float | None = None,
        dropout2: float | None = None,
    ) -> None:
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.keep_raw = keep_raw
        self.momentum_schedule = momentum_schedule
        self.patience = patience
        self.random_state = random_state
        self.embedding_dim = embedding_dim
        self.periodic = periodic
        self.cosine_schedule = cosine_schedule
        self.dropout = dropout
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim
        self.dropout1 = dropout1
        self.dropout2 = dropout2

    # %%
    def fit(self, X, y) -> "NNEmbedding":
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        from tensorflow import keras

        # Reproducible weight init / shuffling / dropout for the embedding net.
        keras.utils.set_random_seed(self.random_state)

        if y is None:
            raise ValueError("NNEmbedding trains a supervised net and needs y at fit time.")
        X = pd.DataFrame(X)
        self.feature_names_in_ = list(X.columns)
        Xv = X.to_numpy(dtype="float32")
        yv = np.asarray(y).astype("float32")

        self.scaler_ = StandardScaler()
        Xs = self.scaler_.fit_transform(Xv).astype("float32")

        model = build_keras_model(
            n_features=Xs.shape[1],
            optimizer=self.optimizer,
            learning_rate=self.learning_rate,
            embedding_dim=self.embedding_dim,
            hidden1_dim=self.hidden1_dim,
            hidden2_dim=self.hidden2_dim,
            dropout=self.dropout,
            dropout1=self.dropout1,
            dropout2=self.dropout2,
            periodic=self.periodic,
        )
        # Early stopping on the validation PR-AUC (a 10% internal holdout) so the
        # train-vs-val gap is observable as an overfit signal for dropout tuning.
        es = keras.callbacks.EarlyStopping(
            monitor="val_pr_auc", mode="max", patience=self.patience,
            restore_best_weights=True,
        )
        callbacks = [es]
        # Optional momentum schedule for SGD: ramp 0.5 -> 0.95 over training so
        # early epochs explore and later ones converge smoothly.
        if self.momentum_schedule and self.optimizer.lower() == "sgd":
            callbacks.append(_momentum_ramp_callback(
                model.optimizer, start=0.5, end=0.95,
                total_epochs=min(self.epochs, 2000)))
        # Optional cosine learning-rate decay over the full epoch budget.
        if self.cosine_schedule:
            total = min(self.epochs, 2000)
            base_lr = self.learning_rate
            callbacks.append(keras.callbacks.LearningRateScheduler(
                lambda e, _lr: float(0.5 * base_lr * (1 + np.cos(np.pi * min(e, total) / total)))
            ))

        history = model.fit(
            Xs, yv, epochs=min(self.epochs, 2000), batch_size=self.batch_size,
            class_weight=class_weight_dict(yv), callbacks=[*callbacks], verbose=0,
            validation_split=0.1, shuffle=True,
        )
        # Record epochs trained and the train/val PR-AUC at the best epoch — the
        # gap is the overfit signal used when tuning dropout / embedding width.
        self.n_epochs_trained_ = len(history.history.get("loss", []))
        vh = history.history.get("val_pr_auc", [])
        th = history.history.get("pr_auc", [])
        if vh:
            best = int(np.argmax(vh))
            self.val_pr_auc_ = float(vh[best])
            self.train_pr_auc_ = float(th[best]) if best < len(th) else float("nan")
            self.overfit_gap_ = self.train_pr_auc_ - self.val_pr_auc_
        else:
            self.val_pr_auc_ = self.train_pr_auc_ = self.overfit_gap_ = float("nan")
        # Sub-model that outputs the named embedding layer.
        self._embed_model = keras.Model(
            inputs=model.inputs, outputs=model.get_layer(EMBEDDING_LAYER).output
        )
        return self

    # %%
    def transform(self, X) -> pd.DataFrame:
        X = pd.DataFrame(X)
        Xs = self.scaler_.transform(X.to_numpy(dtype="float32")).astype("float32")
        emb = self._embed_model.predict(Xs, verbose=0)
        cols = [f"nn_{i}" for i in range(emb.shape[1])]
        emb_df = pd.DataFrame(emb, columns=cols, index=X.index)
        if self.keep_raw:
            return pd.concat([X.reset_index(drop=True), emb_df.reset_index(drop=True)], axis=1)
        return emb_df

    # %%
    def get_feature_names_out(self, input_features=None):
        return np.array([f"nn_{i}" for i in range(self.embedding_dim)])

    # %%
    def __getstate__(self):
        state = self.__dict__.copy()
        model = state.pop("_embed_model", None)
        if model is not None:
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "e.keras")
                model.save(path)
                with open(path, "rb") as fh:
                    state["_embed_bytes"] = fh.read()
        return state

    # %%
    def __setstate__(self, state):
        embed_bytes = state.pop("_embed_bytes", None)
        self.__dict__.update(state)
        if embed_bytes is not None:
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
            from tensorflow import keras

            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "e.keras")
                with open(path, "wb") as fh:
                    fh.write(embed_bytes)
                self._embed_model = keras.models.load_model(path, safe_mode=False)
