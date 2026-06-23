"""Unsupervised 2-group clustering used as a classifier.

Treats each feature as a dimension of a vector space and splits the data into two
clusters (KMeans or a 2-component Gaussian mixture). The clustering itself is
unsupervised; labels are used *only* to decide which of the two groups is the
"phishing" group (the cluster with the higher positive rate) and to expose a
probability for that group. This is a deliberately simple, diagnostic model —
with heavily overlapping classes its PR-AUC is expected to be modest, and the
value is seeing how well an unsupervised split aligns with the label.
"""

# %%
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from ._common import make_pipeline

NAME = "cluster"


# %%
class _ClusterClassifier(BaseEstimator, ClassifierMixin):
    """Two-group clustering aligned to the label, exposing a phishing probability."""

    def __init__(self, algo: str = "kmeans", random_state: int = 42) -> None:
        self.algo = algo
        self.random_state = random_state

    # %%
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)
        self.classes_ = np.array([0, 1])
        self.scaler_ = StandardScaler()
        Xs = self.scaler_.fit_transform(X)

        if self.algo == "kmeans":
            self.cluster_ = KMeans(n_clusters=2, n_init=10, random_state=self.random_state)
            assign = self.cluster_.fit_predict(Xs)
        elif self.algo == "gmm":
            self.cluster_ = GaussianMixture(n_components=2, random_state=self.random_state)
            self.cluster_.fit(Xs)
            assign = self.cluster_.predict(Xs)
        else:
            raise ValueError(f"Unknown algo: {self.algo!r}")

        # Which cluster index corresponds to phishing (higher positive rate)?
        rate0 = y[assign == 0].mean() if (assign == 0).any() else 0.0
        rate1 = y[assign == 1].mean() if (assign == 1).any() else 0.0
        self.phish_cluster_ = int(rate1 >= rate0)
        return self

    # %%
    def predict_proba(self, X):
        Xs = self.scaler_.transform(np.asarray(X, dtype=float))
        if self.algo == "gmm":
            p = self.cluster_.predict_proba(Xs)[:, self.phish_cluster_]
        else:
            # KMeans: convert distance-to-centroids into a soft phishing score.
            d = self.cluster_.transform(Xs)  # distance to each centroid
            other = 1 - self.phish_cluster_
            # Closer to the phishing centroid -> higher probability.
            denom = d[:, self.phish_cluster_] + d[:, other] + 1e-9
            p = d[:, other] / denom
        p = np.clip(p, 0.0, 1.0)
        return np.column_stack([1 - p, p])

    # %%
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# %%
def build(feature_mode: str = "engineered", y=None, embedding_kwargs: dict | None = None):
    """Unfitted clustering-classifier pipeline for the given feature mode."""
    return make_pipeline(_ClusterClassifier(), feature_mode, embedding_kwargs)


# %%
def param_grid() -> dict:
    return {
        "model__algo": ["kmeans", "gmm"],
    }
