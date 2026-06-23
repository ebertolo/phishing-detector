"""Shared TensorFlow/Keras dense-network builder.

Used by both the standalone ``tensorflow_dnn`` model and the ``NNEmbedding``
feature transformer so the architecture lives in one place. TensorFlow is
imported lazily inside the functions to keep ``import phishing`` light and avoid
a hard dependency for code paths that never touch the NN.

Architecture (per the project spec, input width adapted to the real feature
count):

    input (n features, standardised)
      -> Dense(40, relu) -> BatchNormalization -> Dropout(0.4)
      -> Dense(20, relu) -> BatchNormalization -> Dropout(0.4)
      -> [skip] Concatenate(input, h2)            # no gradient of its own
      -> Dense(embedding_dim, name="embedding")   # reusable embedding, default 16-d
      -> L2 normalize (Lambda)                    # no trainable params
      -> Dense(1, activation="tanh")              # score in [-1, 1]

The BatchNormalization, L2-normalize, and Concatenate (skip) operations are the
"between-layer operations that do not require their own gradient descent" the
spec asked for. An optional 1-D pooling variant is documented but off by default
because the feature vector is unordered (pooling over arbitrary feature order is
rarely meaningful for tabular data).
"""

# %%
from __future__ import annotations

import numpy as np

EMBEDDING_LAYER = "embedding"
EMBEDDING_DIM = 16  # default embedding width (parameterizable per call)


# %%
def make_periodic_layer(n_frequencies: int = 8, sigma: float = 1.0):
    """Build a serializable Periodic (Fourier) numerical-embedding layer.

    Implements the Yandex "periodic embeddings for numerical features" idea
    (arXiv 2203.05556): each scalar feature x is mapped to
    ``[sin(2π·c·x), cos(2π·c·x)]`` for a vector of learned frequency coefficients
    ``c`` (one set per feature), then flattened. Frequencies are initialised from
    ``N(0, sigma)`` — the initialisation scale matters for performance. Built as a
    proper ``keras.layers.Layer`` subclass so the model still reloads cleanly.
    """
    import tensorflow as tf
    from tensorflow import keras

    @keras.utils.register_keras_serializable(package="phishing")
    class PeriodicEmbedding(keras.layers.Layer):
        def __init__(self, n_frequencies=8, sigma=1.0, **kwargs):
            super().__init__(**kwargs)
            self.n_frequencies = n_frequencies
            self.sigma = sigma

        def build(self, input_shape):
            self.n_features = int(input_shape[-1])
            # Static output width: n_features * (2 * n_frequencies) [sin + cos].
            self.out_dim = self.n_features * 2 * self.n_frequencies
            # One learned frequency vector per feature: shape (n_features, k).
            self.coeffs = self.add_weight(
                name="frequencies",
                shape=(self.n_features, self.n_frequencies),
                initializer=keras.initializers.RandomNormal(stddev=self.sigma),
                trainable=True,
            )

        def call(self, x):
            # x: (batch, n_features) -> (batch, n_features, 1) * (n_features, k)
            xf = tf.expand_dims(x, -1) * self.coeffs  # (batch, n_features, k)
            two_pi = 2.0 * np.pi
            feats = tf.concat([tf.sin(two_pi * xf), tf.cos(two_pi * xf)], axis=-1)
            # Reshape to a STATIC output width so downstream Dense layers infer shape.
            return tf.reshape(feats, [-1, self.out_dim])

        def compute_output_shape(self, input_shape):
            n_features = int(input_shape[-1])
            return (input_shape[0], n_features * 2 * self.n_frequencies)

        def get_config(self):
            cfg = super().get_config()
            cfg.update({"n_frequencies": self.n_frequencies, "sigma": self.sigma})
            return cfg

    return PeriodicEmbedding(n_frequencies=n_frequencies, sigma=sigma, name="periodic")


# %%
def make_optimizer(name: str, learning_rate: float):
    """Resolve an optimizer name to a configured Keras optimizer.

    Supports the optimizers requested for comparison: SGD (with Nesterov
    momentum), RMSprop, Adam, and Nadam.
    """
    from tensorflow import keras

    name = name.lower()
    if name == "sgd":
        return keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
    if name == "rmsprop":
        return keras.optimizers.RMSprop(learning_rate=learning_rate)
    if name == "adam":
        return keras.optimizers.Adam(learning_rate=learning_rate)
    if name == "nadam":
        return keras.optimizers.Nadam(learning_rate=learning_rate)
    raise ValueError(f"Unknown optimizer: {name!r}")


# %%
def build_keras_model(
    n_features: int,
    optimizer: str,
    learning_rate: float,
    dropout: float = 0.4,
    embedding_dim: int = EMBEDDING_DIM,
    hidden1_dim: int = 40,
    hidden2_dim: int = 20,
    dropout1: float | None = None,
    dropout2: float | None = None,
    periodic: bool = False,
    periodic_frequencies: int = 8,
    periodic_sigma: float = 1.0,
):
    """Build and compile the dense network.

    Parameters added for tuning:

    - ``embedding_dim`` — width of the reusable ``embedding`` layer (16/20/32/64).
    - ``hidden1_dim`` — width of the first intermediate Dense layer; default 40.
    - ``hidden2_dim`` — width of the second intermediate Dense layer (the one
      feeding the embedding); default 20.
    - ``dropout`` — dropout rate applied after each intermediate Dense+BN block
      when ``dropout1``/``dropout2`` are not given (backward-compatible default).
    - ``dropout1``/``dropout2`` — per-layer dropout override for the first/second
      intermediate block respectively; falls back to ``dropout`` when ``None``.
    - ``periodic`` — if True, prepend a Periodic (Fourier) numerical-feature
      embedding (Yandex). It expands each feature into learned sin/cos components,
      which makes the MLP markedly more expressive on numeric data.
    - the embedding is followed by a **linear + ReLU** head (Yandex recommendation)
      before the final score.

    The output is ``tanh`` (range [-1, 1]); training uses binary cross-entropy on
    the ``(tanh + 1) / 2`` mapping so the [-1, 1] score behaves as a probability.
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    dropout1 = dropout if dropout1 is None else dropout1
    dropout2 = dropout if dropout2 is None else dropout2

    inp = keras.Input(shape=(n_features,), name="features")

    # Optional periodic numerical embedding front-end.
    x = inp
    if periodic:
        x = make_periodic_layer(periodic_frequencies, periodic_sigma)(inp)

    h1 = layers.Dense(hidden1_dim, activation="relu")(x)
    h1 = layers.BatchNormalization()(h1)
    h1 = layers.Dropout(dropout1)(h1)

    h2 = layers.Dense(hidden2_dim, activation="relu")(h1)
    h2 = layers.BatchNormalization()(h2)
    h2 = layers.Dropout(dropout2)(h2)

    # Skip connection (no trainable params): preserve the (front-end) signal.
    skip = layers.Concatenate()([x, h2])

    # Reusable embedding, then a linear+ReLU head on top (Yandex), then L2-norm.
    emb = layers.Dense(embedding_dim, activation="relu", name=EMBEDDING_LAYER)(skip)
    head = layers.Dense(embedding_dim, activation="relu", name="emb_head")(emb)
    emb_norm = layers.UnitNormalization(name="l2norm")(head)

    # tanh output -> score in [-1, 1]; mapped to a probability by the wrapper.
    score = layers.Dense(1, activation="tanh", name="score")(emb_norm)

    model = keras.Model(inputs=inp, outputs=score)
    model.compile(
        optimizer=make_optimizer(optimizer, learning_rate),
        loss=_tanh_bce,
        metrics=[keras.metrics.AUC(name="pr_auc", curve="PR")],
    )
    return model


# %%
def _tanh_bce(y_true, y_pred):
    """Binary cross-entropy on a tanh output mapped to [0, 1] via (x+1)/2."""
    from tensorflow import keras

    p = (y_pred + 1.0) / 2.0
    return keras.losses.binary_crossentropy(y_true, p)


# %%
def class_weight_dict(y) -> dict:
    """Balanced class weights for the rare-positive problem."""
    y = np.asarray(y).astype(int)
    n = len(y)
    pos = max(int(y.sum()), 1)
    neg = max(int((y == 0).sum()), 1)
    # Inverse-frequency weights normalised so the majority class is ~1.
    return {0: n / (2.0 * neg), 1: n / (2.0 * pos)}
