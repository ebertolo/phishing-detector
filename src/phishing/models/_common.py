"""Shared helpers for per-algorithm model files.

Keeps each algorithm module small: it provides the optional WOE front-end and a
class-imbalance helper so every model file follows the same shape.
"""

# %%
from __future__ import annotations

import numpy as np
from sklearn.pipeline import Pipeline

from ..features.autoencoder import AutoencoderEncoder
from ..features.binning import OptimalBinningWOE
from ..features.engineering import FeatureEngineer
from ..features.interactions import InteractionFeatures
from ..features.kbins import QuantileBinner
from ..features.smoothing import FeatureSmoothing
from ..features.target_encoding import CrossFitTargetEncoder

# Base encoding strategies for validation/comparison.
_ENCODINGS = ["raw", "binned_woe", "quantile", "target", "autoencoder"]

# Post-engineering steps appended via an "engineered_<post>" mode: a NN embedding,
# winsor+quantile smoothing, or a denoising-autoencoder reconstruction.
_POST_STEPS = ["nnembed", "smooth", "denoise"]

# Selectable feature modes:
# - an "engineered" prefix prepends the deterministic FeatureEngineer
#   (flags + log + ratios); "engineered" alone is the recommended default.
# - an "ix_" prefix prepends explicit pairwise InteractionFeatures.
# - an "engineered_<post>" mode appends a post-engineering step (nnembed / smooth
#   / denoise). Both prefixes compose, e.g. "ix_engineered".
FEATURE_MODES = (
    _ENCODINGS
    + ["engineered", "engineered_noemail"]
    + [f"engineered_{e}" for e in _ENCODINGS if e != "raw"]
    + [f"engineered_{p}" for p in _POST_STEPS]
    + ["ix_raw", "ix_engineered"]
)


# %%
def scale_pos_weight(y) -> float:
    """negatives / positives ratio for boosters' ``scale_pos_weight``."""
    y = np.asarray(y).astype(int)
    pos = max(int(y.sum()), 1)
    neg = int((y == 0).sum())
    return neg / pos


# %%
def gpu_available() -> bool:
    """Detect a usable NVIDIA GPU (e.g. a Colab GPU runtime).

    Cheap, side-effect-free check via ``nvidia-smi`` so booster modules can
    opt into GPU training without importing torch/tensorflow just to probe.
    Honors the ``PHISHING_FORCE_CPU=1`` env var to force CPU even if a GPU is
    present (useful for reproducing CPU-only results on a GPU machine).
    """
    import os
    import shutil
    import subprocess

    if os.environ.get("PHISHING_FORCE_CPU") == "1":
        return False
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0 and "GPU" in result.stdout
    except (OSError, subprocess.SubprocessError):
        return False


# %%
def lightgbm_cuda_available() -> bool:
    """Whether the installed LightGBM wheel was built with CUDA support.

    The default PyPI wheel ships **without** CUDA (only the generic OpenCL
    backend, ``device="gpu"``). On a machine with both an NVIDIA GPU and an
    integrated GPU, OpenCL can silently pick the integrated one — slower than
    CPU and not the device the user asked for. Detect this by attempting a
    1-row CUDA fit; only use LightGBM's GPU device when it is the real CUDA
    backend. Cached per-process since the check launches a tiny fit.
    """
    if gpu_available() is False:
        return False
    if getattr(lightgbm_cuda_available, "_cache", None) is not None:
        return lightgbm_cuda_available._cache
    try:
        import lightgbm as lgb

        m = lgb.LGBMClassifier(device="cuda", n_estimators=1, verbose=-1)
        m.fit([[0.0, 1.0], [1.0, 0.0]], [0, 1])
        lightgbm_cuda_available._cache = True
    except Exception:
        lightgbm_cuda_available._cache = False
    return lightgbm_cuda_available._cache


# %%
def _encoding_step(encoding: str):
    """Return the (name, transformer) step for a base encoding, or None for raw.

    All transformers are leakage-safe: supervised ones fit inside each CV fold,
    target encoding uses out-of-fold statistics, and unsupervised ones learn from
    the training rows only.
    """
    if encoding == "raw":
        return None
    if encoding == "binned_woe":
        return ("woe", OptimalBinningWOE())
    if encoding == "quantile":
        return ("quantile", QuantileBinner())
    if encoding == "target":
        return ("target", CrossFitTargetEncoder())
    if encoding == "autoencoder":
        return ("autoencoder", AutoencoderEncoder())
    raise ValueError(f"Unknown encoding: {encoding!r}")


# %%
def _post_step(post: str, embedding_kwargs: dict | None = None):
    """Return the (name, transformer) for a post-engineering step.

    Imported lazily where TensorFlow is involved (NN embedding) so the rest of
    the module loads without TensorFlow. ``embedding_kwargs`` (e.g. ``embedding_dim``,
    ``hidden1_dim``, ``hidden2_dim``, ``dropout1``, ``dropout2``, ``patience``)
    overrides ``NNEmbedding``'s defaults when ``post == "nnembed"``; ignored
    otherwise.
    """
    if post == "smooth":
        return ("smooth", FeatureSmoothing(method="winsor_quantile"))
    if post == "denoise":
        return ("denoise", AutoencoderEncoder(denoising=True, output="reconstruction"))
    if post == "nnembed":
        from ..features.nn_embedding import NNEmbedding

        return ("nnembed", NNEmbedding(keep_raw=True, **(embedding_kwargs or {})))
    raise ValueError(f"Unknown post step: {post!r}")


# %%
def _parse_mode(feature_mode: str) -> tuple[bool, bool, bool, str, str | None]:
    """Split a mode into (interactions, engineering, logz_email, encoding, post).

    Prefixes compose: ``"ix_"`` -> interactions, ``"engineered"`` ->
    FeatureEngineer. ``"engineered_noemail"`` toggles off the logz email feature.
    ``"engineered_<post>"`` (post in nnembed/smooth/denoise) appends a
    post-engineering step instead of an encoding. Examples: ``"engineered"`` ->
    (F, T, T, "raw", None); ``"engineered_smooth"`` -> (F, T, T, "raw", "smooth");
    ``"engineered_woe"`` -> (F, T, T, "woe", None); a plain encoding ->
    (F, F, T, enc, None).
    """
    use_interactions = False
    if feature_mode.startswith("ix_"):
        use_interactions = True
        feature_mode = feature_mode[len("ix_"):]
    if feature_mode == "engineered":
        return use_interactions, True, True, "raw", None
    if feature_mode == "engineered_noemail":
        return use_interactions, True, False, "raw", None
    if feature_mode.startswith("engineered_"):
        rest = feature_mode[len("engineered_"):]
        if rest in _POST_STEPS:
            return use_interactions, True, True, "raw", rest
        return use_interactions, True, True, rest, None
    return use_interactions, False, True, feature_mode, None


# %%
def make_pipeline(
    estimator, feature_mode: str, embedding_kwargs: dict | None = None
) -> Pipeline:
    """Wrap an estimator with optional engineering / interactions / post / encoding.

    Steps, in order: explicit interactions (``ix_``), the deterministic
    ``FeatureEngineer`` (``engineered``), an optional post-engineering step
    (``nnembed`` / ``smooth`` / ``denoise``), then the encoding front-end (fit
    inside each CV fold, leakage-safe). See ``FEATURE_MODES``.

    ``embedding_kwargs`` overrides the NN embedding's architecture (width/dropout
    per layer, patience, ...) when ``feature_mode`` ends in ``nnembed``; ignored
    for every other mode.
    """
    use_interactions, use_engineering, logz_email, encoding, post = _parse_mode(feature_mode)
    steps = []
    if use_engineering:
        steps.append(
            ("engineer", FeatureEngineer(keep_raw=True, add_logz_email=logz_email))
        )
    if use_interactions:
        steps.append(("interactions", InteractionFeatures(keep_raw=True)))
    if post is not None:
        steps.append(_post_step(post, embedding_kwargs))
    enc = _encoding_step(encoding)
    if enc is not None:
        steps.append(enc)
    steps.append(("model", estimator))
    return Pipeline(steps)
