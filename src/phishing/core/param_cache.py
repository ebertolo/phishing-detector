"""Cache for hyperparameter-search winners, keyed by search configuration.

``RandomizedSearchCV``/``GridSearchCV`` is the most expensive step in a run (see
``docs/EXPERIMENT_JOURNEY.md`` — tuning is the single biggest lever after feature
engineering, but it is also the slowest). Once a search has found the best
parameters for a given (model, feature set, search config) combination, repeating
it is wasted time. This module persists each winner to a structured JSON file
keyed by a deterministic hash of everything that could change the answer, so a
later run with the same configuration can skip the search and fit once with the
cached parameters — and a run with a *different* configuration (different
features, different search budget, ...) correctly misses the cache and searches
again.

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

CACHE_DIR = Path("best_params")


# %%
@dataclass
class CachedParams:
    """One cached hyperparameter-search winner, self-describing on disk."""

    model_name: str
    feature_mode: str
    search_method: str
    n_iter: int
    n_splits: int
    feature_columns: list[str]
    best_params: dict[str, Any]
    cv_pr_auc: float
    created_at: str
    cache_key: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


# %%
def make_cache_key(
    model_name: str,
    feature_mode: str,
    search_method: str,
    n_iter: int,
    n_splits: int,
    feature_columns: list[str],
    embedding_kwargs: dict | None = None,
) -> str:
    """Deterministic key identifying a search configuration.

    Two runs hash to the same key — and so can share a cached winner — only when
    the model, feature mode, search method/budget, CV folds, the exact set of
    training columns, and the embedding architecture overrides (when relevant,
    e.g. ``feature_mode="engineered_nnembed"``) all match. Changing any of these
    (e.g. a different feature engineering version, a wider random-search budget,
    a different embedding width) naturally invalidates the cache instead of
    silently reusing parameters tuned for a different problem.
    """
    emb_payload = (
        ",".join(f"{k}={embedding_kwargs[k]}" for k in sorted(embedding_kwargs))
        if embedding_kwargs else ""
    )
    payload = "|".join(
        [
            model_name,
            feature_mode,
            search_method,
            str(n_iter if search_method == "random" else "grid"),
            str(n_splits),
            ",".join(sorted(feature_columns)),
            emb_payload,
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# %%
def load_cached_params(
    cache_key: str, cache_dir: Path | str | None = None
) -> CachedParams | None:
    """Return the cached winner for ``cache_key``, or ``None`` if absent/unreadable."""
    path = Path(cache_dir if cache_dir is not None else CACHE_DIR) / f"{cache_key}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return CachedParams(**data)
    except (json.JSONDecodeError, TypeError, KeyError):
        return None


# %%
def save_cached_params(
    model_name: str,
    feature_mode: str,
    search_method: str,
    n_iter: int,
    n_splits: int,
    feature_columns: list[str],
    best_params: dict[str, Any],
    cv_pr_auc: float,
    cache_dir: Path | str | None = None,
    embedding_kwargs: dict | None = None,
) -> Path:
    """Persist a search winner, returning the JSON file path written."""
    cache_key = make_cache_key(
        model_name, feature_mode, search_method, n_iter, n_splits, feature_columns,
        embedding_kwargs,
    )
    record = CachedParams(
        model_name=model_name,
        feature_mode=feature_mode,
        search_method=search_method,
        n_iter=n_iter,
        n_splits=n_splits,
        feature_columns=list(feature_columns),
        best_params=best_params,
        cv_pr_auc=cv_pr_auc,
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        cache_key=cache_key,
    )
    cache_dir = Path(cache_dir if cache_dir is not None else CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{cache_key}.json"
    path.write_text(record.to_json(), encoding="utf-8")
    return path
