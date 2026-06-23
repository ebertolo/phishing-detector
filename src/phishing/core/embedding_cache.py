"""Cache for trained NN embeddings, keyed by architecture + input data.

Training the NN embedding (``phishing.features.nn_embedding.NNEmbedding``) is the
single most expensive fixed cost in ``scripts/best_model_report.py`` and similar
flows — several minutes for the full dataset regardless of which boosters are run
afterwards (see ``docs/EXPERIMENT_JOURNEY.md`` §5). When nothing about the
embedding's configuration or its input changed, retraining it is wasted time.
This module persists the fitted ``NNEmbedding`` object (joblib, which round-trips
its Keras sub-model via the object's own ``__getstate__``/``__setstate__`` —
i.e. native ``.keras`` serialisation under the hood) alongside a small JSON
metadata file, keyed by every hyperparameter that affects the trained weights
plus the input feature columns and training-set size.

No Streamlit/FastAPI imports — usable from the CLIs, the runner, and a future API.
"""

# %%
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

CACHE_DIR = Path("embeddings")
_MODEL_FILE = "embedding.joblib"
_META_FILE = "metadata.json"


# %%
@dataclass
class EmbeddingCacheMeta:
    """Self-describing record persisted alongside each cached embedding."""

    cache_key: str
    embedding_dim: int
    hidden1_dim: int
    hidden2_dim: int
    dropout1: float
    dropout2: float
    patience: int
    optimizer: str
    learning_rate: float
    periodic: bool
    cosine_schedule: bool
    feature_columns: list[str]
    n_train_rows: int
    train_pr_auc: float
    val_pr_auc: float
    n_epochs_trained: int
    created_at: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


# %%
def make_embedding_cache_key(
    embedding_dim: int,
    hidden1_dim: int,
    hidden2_dim: int,
    dropout1: float,
    dropout2: float,
    patience: int,
    optimizer: str,
    learning_rate: float,
    periodic: bool,
    cosine_schedule: bool,
    feature_columns: list[str],
    n_train_rows: int,
) -> str:
    """Deterministic key identifying an embedding's exact training configuration.

    Includes every hyperparameter that changes the trained weights (architecture —
    every layer's width and dropout — optimizer, regularisation, schedule) plus
    the input feature columns and the training-set row count, so a different
    feature-engineering version, a different sample size, or a different layer
    width/dropout correctly misses the cache instead of reusing an embedding
    trained with a different architecture or on different data.
    """
    payload = "|".join(
        [
            str(embedding_dim),
            str(hidden1_dim),
            str(hidden2_dim),
            f"{dropout1:.6f}",
            f"{dropout2:.6f}",
            str(patience),
            optimizer,
            f"{learning_rate:.8f}",
            str(periodic),
            str(cosine_schedule),
            ",".join(sorted(feature_columns)),
            str(n_train_rows),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# %%
def load_cached_embedding(
    cache_key: str, cache_dir: Path | str | None = None
) -> tuple[Any, EmbeddingCacheMeta] | None:
    """Return ``(embedding, metadata)`` for ``cache_key``, or ``None`` on a miss.

    A miss covers: no cache directory for this key, a corrupted/partial write, or
    a joblib load failure (e.g. after a library upgrade) — any of these fall back
    to retraining rather than raising, since the cache is purely an optimisation.
    """
    version_dir = Path(cache_dir if cache_dir is not None else CACHE_DIR) / cache_key
    model_path = version_dir / _MODEL_FILE
    meta_path = version_dir / _META_FILE
    if not (model_path.exists() and meta_path.exists()):
        return None
    try:
        meta = EmbeddingCacheMeta(**json.loads(meta_path.read_text(encoding="utf-8")))
        embedding = joblib.load(model_path)
    except Exception:
        return None
    return embedding, meta


# %%
def save_cached_embedding(
    embedding: Any,
    embedding_dim: int,
    hidden1_dim: int,
    hidden2_dim: int,
    dropout1: float,
    dropout2: float,
    patience: int,
    optimizer: str,
    learning_rate: float,
    periodic: bool,
    cosine_schedule: bool,
    feature_columns: list[str],
    n_train_rows: int,
    train_pr_auc: float,
    val_pr_auc: float,
    n_epochs_trained: int,
    cache_dir: Path | str | None = None,
) -> Path:
    """Persist a fitted embedding + its metadata, returning the version directory."""
    cache_key = make_embedding_cache_key(
        embedding_dim, hidden1_dim, hidden2_dim, dropout1, dropout2, patience,
        optimizer, learning_rate, periodic, cosine_schedule, feature_columns,
        n_train_rows,
    )
    version_dir = Path(cache_dir if cache_dir is not None else CACHE_DIR) / cache_key
    version_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(embedding, version_dir / _MODEL_FILE)
    meta = EmbeddingCacheMeta(
        cache_key=cache_key,
        embedding_dim=embedding_dim,
        hidden1_dim=hidden1_dim,
        hidden2_dim=hidden2_dim,
        dropout1=dropout1,
        dropout2=dropout2,
        patience=patience,
        optimizer=optimizer,
        learning_rate=learning_rate,
        periodic=periodic,
        cosine_schedule=cosine_schedule,
        feature_columns=list(feature_columns),
        n_train_rows=n_train_rows,
        train_pr_auc=train_pr_auc,
        val_pr_auc=val_pr_auc,
        n_epochs_trained=n_epochs_trained,
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    (version_dir / _META_FILE).write_text(meta.to_json(), encoding="utf-8")
    return version_dir
