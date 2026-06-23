"""Deterministic feature engineering for the integer count features.

Analysis of the real dataset showed the raw counts carry very little individual
signal (max mutual information ~0.0045) because they are heavy-tailed and
zero-inflated. This transformer derives a rich set of engineered features that
extract the signal the raw counts hide:

- **presence / graded flags** (``has_*``, ``many_*``, ``short_email`` …) for the
  zero-inflated and thresholdable features;
- **log transforms** to tame the heavy tails (max num_words ~2.3M);
- **densities / ratios** that normalise counts by message length or by each
  other (lexical diversity, link/email/urgency densities, per-link/per-domain
  ratios, content-word ratios);
- **pairwise interactions** (links×urgency, domains×errors …);
- a leakage-safe **log+z-score** of num_email_addresses (top-MI feature) and
  **percentile-95 density flags** whose thresholds are learned on the train data.

The transformation is **row-wise and deterministic** apart from the few
train-learned statistics (logz mean/std and the p95 density thresholds), which
are fit on the training data only — so the whole transform is leakage-safe.

De-duplication policy: a feature is emitted **once**. Where a name from the
extended list has the *same formula* as an existing feature (e.g. ``has_emails``,
``stopword_ratio``, ``lexical_diversity`` — all using the ``num_words + eps``
denominator), only the original is kept. Genuinely different formulas (e.g. the
``/ max(x, 1)`` per-link ratios, interactions, graded flags) are added.
"""

# %%
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ..core.data import FEATURES

# Zero-inflated features that get a binary presence flag, mapped to a clean name.
_FLAG_FEATURES = {
    "num_links": "has_links",
    "num_unique_domains": "has_domains",
    "num_email_addresses": "has_emails",
    "num_urgent_keywords": "has_urgent",
}


# %%
class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Add flags, logs, densities, ratios and interactions to the raw counts.

    Parameters
    ----------
    keep_raw : bool
        If True (default) the original count columns are kept alongside the
        engineered ones; if False only the engineered columns are returned.
    eps : float
        Small constant added to ``num_words``-based denominators to avoid
        division by zero.
    add_logz_email : bool
        Whether to add the leakage-safe ``logz_num_email_addresses`` feature.
    density_pct : float
        Percentile (0-1) used to learn the ``high_*_density`` thresholds on the
        training data (default 0.95 = top 5%).

    Notes
    -----
    Operates on whatever subset of the canonical ``FEATURES`` is present, so it
    tolerates schema variations. Learns only train-side statistics (logz stats
    and the p95 density thresholds), so it is leakage-safe.
    """

    def __init__(
        self,
        keep_raw: bool = True,
        eps: float = 1.0,
        add_logz_email: bool = True,
        density_pct: float = 0.95,
    ) -> None:
        self.keep_raw = keep_raw
        self.eps = eps
        self.add_logz_email = add_logz_email
        self.density_pct = density_pct

    # %%
    def fit(self, X: pd.DataFrame, y=None) -> "FeatureEngineer":
        X = pd.DataFrame(X)
        self.feature_names_in_ = list(X.columns)
        self.present_ = [c for c in FEATURES if c in X.columns]

        # logz email stats (train only) — leakage-safe z-score.
        self._logz_email_stats = None
        if self.add_logz_email and "num_email_addresses" in self.present_:
            logged = np.log1p(X["num_email_addresses"].astype(float))
            mean = float(logged.mean())
            std = float(logged.std())
            self._logz_email_stats = (mean, std if std > 0 else 1.0)

        # Percentile-95 thresholds for the density flags (train only).
        self._density_thresholds_ = {}
        derived = self._densities(X)
        for name in ("link_density", "urgency_ratio", "spelling_error_ratio"):
            if name in derived:
                self._density_thresholds_[name] = float(
                    np.quantile(derived[name], self.density_pct)
                )
        return self

    # %%
    def _densities(self, X: pd.DataFrame) -> dict:
        """num_words-normalised densities (used in both fit and transform)."""
        cols = set(self.present_)
        eps = self.eps
        d = {}

        def c(name):
            return X[name].astype(float)

        if "num_words" in cols:
            w = np.maximum(c("num_words") + eps, 1.0)  # match transform's w
            if "num_links" in cols:
                d["link_density"] = c("num_links") / w
            if "num_urgent_keywords" in cols:
                d["urgency_ratio"] = c("num_urgent_keywords") / w
            if "num_spelling_errors" in cols:
                d["spelling_error_ratio"] = c("num_spelling_errors") / w
        return d

    # %%
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X).copy()
        eps = self.eps
        cols = set(self.present_)
        out = pd.DataFrame(index=X.index)

        def c(name):
            return X[name].astype(float)

        def has(*names):
            return all(n in cols for n in names)

        def mx(name):
            # Denominator floored at 1.0 so per-X ratios never divide by zero.
            return np.maximum(c(name), 1.0)

        # num_words denominator: + eps AND floored at 1.0 so it is safe even if
        # eps is set to 0 or a row has num_words == 0.
        w = np.maximum(c("num_words") + eps, 1.0) if "num_words" in cols else None

        # ---- presence flags (one per feature; same formula -> single name) ---
        for src, flag in _FLAG_FEATURES.items():
            if src in cols:
                out[flag] = (c(src) > 0).astype(int)

        # ---- log transforms (log_num_* canonical names) ----------------------
        for col in self.present_:
            out[f"log_{col}"] = np.log1p(c(col))

        # ---- densities / ratios normalised by message length -----------------
        # (num_words + eps denominator; each emitted once)
        if w is not None:
            if has("num_unique_words"):
                out["lexical_diversity"] = c("num_unique_words") / w
                out["word_repetition_ratio"] = 1 - (c("num_unique_words") / w)
            if has("num_stopwords"):
                out["stopword_ratio"] = c("num_stopwords") / w
                out["content_word_ratio"] = (c("num_words") - c("num_stopwords")) / w
            if has("num_links"):
                out["link_density"] = c("num_links") / w
            if has("num_email_addresses"):
                out["email_density"] = c("num_email_addresses") / w
            if has("num_spelling_errors"):
                out["spelling_error_ratio"] = c("num_spelling_errors") / w
            if has("num_urgent_keywords"):
                out["urgency_ratio"] = c("num_urgent_keywords") / w

        # ---- per-link / per-domain / per-unique-word ratios (/ max(x, 1)) ----
        if has("num_unique_domains", "num_links"):
            out["domain_per_link_ratio"] = c("num_unique_domains") / mx("num_links")
            out["links_per_domain_ratio"] = c("num_links") / mx("num_unique_domains")
        if has("num_email_addresses", "num_links"):
            out["emails_per_link_ratio"] = c("num_email_addresses") / mx("num_links")
        if has("num_email_addresses", "num_unique_domains"):
            out["emails_per_domain_ratio"] = c("num_email_addresses") / mx("num_unique_domains")
        if has("num_spelling_errors", "num_links"):
            out["spelling_errors_per_link"] = c("num_spelling_errors") / mx("num_links")
        if has("num_urgent_keywords", "num_links"):
            out["urgent_keywords_per_link"] = c("num_urgent_keywords") / mx("num_links")
        if has("num_spelling_errors", "num_unique_words"):
            out["spelling_errors_per_unique_word"] = c("num_spelling_errors") / mx("num_unique_words")
        if has("num_urgent_keywords", "num_unique_words"):
            out["urgency_per_unique_word"] = c("num_urgent_keywords") / mx("num_unique_words")
        if has("num_unique_words", "num_stopwords"):
            out["text_complexity"] = c("num_unique_words") / mx("num_stopwords")

        # ---- content-word (non-stopword) intensities -------------------------
        if has("num_words", "num_stopwords"):
            content = np.maximum(c("num_words") - c("num_stopwords"), 1.0)
            if has("num_links"):
                out["link_to_content_ratio"] = c("num_links") / content
            if has("num_email_addresses"):
                out["email_to_content_ratio"] = c("num_email_addresses") / content
            if has("num_spelling_errors"):
                out["error_to_content_ratio"] = c("num_spelling_errors") / content
            if has("num_urgent_keywords"):
                out["urgency_to_content_ratio"] = c("num_urgent_keywords") / content

        # ---- graded presence flags -------------------------------------------
        if has("num_links"):
            out["has_multiple_links"] = (c("num_links") > 1).astype(int)
        if has("num_unique_domains"):
            out["has_multiple_domains"] = (c("num_unique_domains") > 1).astype(int)
        if has("num_spelling_errors"):
            out["has_spelling_errors"] = (c("num_spelling_errors") > 0).astype(int)
            out["many_spelling_errors"] = (c("num_spelling_errors") >= 3).astype(int)
        if has("num_urgent_keywords"):
            out["many_urgent_keywords"] = (c("num_urgent_keywords") >= 2).astype(int)
        if has("num_words"):
            out["short_email"] = (c("num_words") < 50).astype(int)
            out["very_short_email"] = (c("num_words") < 20).astype(int)
            out["long_email"] = (c("num_words") > 300).astype(int)

        # ---- percentile-95 density flags (train-learned thresholds) ----------
        for dname, flag in (
            ("link_density", "high_link_density"),
            ("urgency_ratio", "high_urgency_density"),
            ("spelling_error_ratio", "high_error_density"),
        ):
            thr = self._density_thresholds_.get(dname)
            if thr is not None and dname in out.columns:
                out[flag] = (out[dname] > thr).astype(int)

        # ---- pairwise interactions -------------------------------------------
        if has("num_links", "num_urgent_keywords"):
            out["links_x_urgency"] = c("num_links") * c("num_urgent_keywords")
        if has("num_links", "num_spelling_errors"):
            out["links_x_errors"] = c("num_links") * c("num_spelling_errors")
        if has("num_unique_domains", "num_urgent_keywords"):
            out["domains_x_urgency"] = c("num_unique_domains") * c("num_urgent_keywords")
        if has("num_unique_domains", "num_spelling_errors"):
            out["domains_x_errors"] = c("num_unique_domains") * c("num_spelling_errors")
        if has("num_spelling_errors", "num_urgent_keywords"):
            out["errors_x_urgency"] = c("num_spelling_errors") * c("num_urgent_keywords")
        if has("num_email_addresses", "num_links"):
            out["emails_x_links"] = c("num_email_addresses") * c("num_links")
        if has("num_email_addresses", "num_unique_domains"):
            out["emails_x_domains"] = c("num_email_addresses") * c("num_unique_domains")

        # ---- leakage-safe log+z of num_email_addresses -----------------------
        if self._logz_email_stats is not None and "num_email_addresses" in cols:
            mean, std = self._logz_email_stats
            out["logz_num_email_addresses"] = (np.log1p(c("num_email_addresses")) - mean) / std

        if self.keep_raw:
            return pd.concat([X.reset_index(drop=True), out.reset_index(drop=True)], axis=1)
        return out

    # %%
    def get_feature_names_out(self, input_features=None):
        # Built lazily; callers typically read columns off the transformed frame.
        raise NotImplementedError(
            "Use the columns of the transformed DataFrame for feature names."
        )
