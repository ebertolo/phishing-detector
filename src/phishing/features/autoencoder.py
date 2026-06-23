"""CPU autoencoder feature transformer (optional / experimental).

Trains an unsupervised MLP to reconstruct the standardised features and uses the
bottleneck (latent) activations as a new representation. Built on scikit-learn's
``MLPRegressor`` so it needs **no GPU and no extra dependencies**.

This is provided so an autoencoder representation can be *validated and compared*
like any other encoding. It is **not** the recommended default: it is slower,
non-deterministic, and far less interpretable than the binning/WOE path that this
project prioritises.
"""

# %%
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


# %%
class AutoencoderEncoder(BaseEstimator, TransformerMixin):
    """Unsupervised MLP autoencoder; outputs the bottleneck latent features.

    Parameters
    ----------
    latent_dim : int
        Width of the bottleneck (number of output features).
    hidden_dim : int
        Width of the symmetric encoder/decoder hidden layers.
    max_iter : int
        Training iterations for the MLP.
    random_state : int
        Seed (note: training is only approximately reproducible).
    denoising : bool
        If True, train as a **denoising** autoencoder: corrupt the input with
        Gaussian noise and learn to reconstruct the clean input. This is the
        common fraud / rare-event smoothing setup.
    output : {"latent", "reconstruction"}
        ``latent`` (default) emits the bottleneck embedding ``ae_*``;
        ``reconstruction`` emits the smoothed, denoised reconstruction of the
        original feature columns (suffixed ``_dn``) — useful as a smoothing step.
    noise_std : float
        Std of the Gaussian corruption used during denoising training.
    """

    def __init__(
        self,
        latent_dim: int = 4,
        hidden_dim: int = 16,
        max_iter: int = 200,
        random_state: int = 42,
        denoising: bool = False,
        output: str = "latent",
        noise_std: float = 0.3,
    ) -> None:
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.max_iter = max_iter
        self.random_state = random_state
        self.denoising = denoising
        self.output = output
        self.noise_std = noise_std

    # %%
    def fit(self, X: pd.DataFrame, y=None) -> "AutoencoderEncoder":
        X = pd.DataFrame(X)
        self.feature_names_in_ = list(X.columns)
        self.scaler_ = StandardScaler()
        Xs = self.scaler_.fit_transform(X)

        # Symmetric architecture; the middle layer is the latent bottleneck.
        self.net_ = MLPRegressor(
            hidden_layer_sizes=(self.hidden_dim, self.latent_dim, self.hidden_dim),
            activation="relu",
            solver="adam",
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        # Plain AE reconstructs the input from itself; denoising AE reconstructs
        # the clean input from a noise-corrupted version.
        if self.denoising:
            rng = np.random.default_rng(self.random_state)
            X_in = Xs + rng.normal(0.0, self.noise_std, size=Xs.shape)
        else:
            X_in = Xs
        self.net_.fit(X_in, Xs)
        return self

    # %%
    def _encode(self, Xs: np.ndarray) -> np.ndarray:
        """Forward-pass up to the bottleneck layer to get latent activations."""
        activation = Xs
        # Layers: input -> hidden -> latent -> hidden -> output. Stop at latent
        # (index 1 of the weight/bias lists, i.e. after the second matmul).
        for i in range(2):
            activation = activation @ self.net_.coefs_[i] + self.net_.intercepts_[i]
            activation = np.maximum(activation, 0)  # ReLU
        return activation

    # %%
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X)
        Xs = self.scaler_.transform(X)
        if self.output == "reconstruction":
            # Smoothed/denoised reconstruction in the original feature space.
            recon_scaled = self.net_.predict(Xs)
            recon = self.scaler_.inverse_transform(recon_scaled)
            cols = [f"{c}_dn" for c in self.feature_names_in_]
            return pd.DataFrame(recon, columns=cols, index=X.index)
        latent = self._encode(Xs)
        cols = [f"ae_{i}" for i in range(latent.shape[1])]
        return pd.DataFrame(latent, columns=cols, index=X.index)

    # %%
    def get_feature_names_out(self, input_features=None):
        if self.output == "reconstruction":
            return np.array([f"{c}_dn" for c in self.feature_names_in_])
        return np.array([f"ae_{i}" for i in range(self.latent_dim)])
