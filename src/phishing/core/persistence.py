"""Versioned model persistence.

Each saved version is a folder ``models/<name>__<timestamp>/`` containing:

- ``model.joblib``  : the fitted pipeline / estimator (joblib artifact).
- ``metadata.json`` : algorithm description, training timestamp, feature list,
                      chosen threshold, and the validation metrics achieved.

This lets a version be identified and compared without retraining, and lets the
UI list and select versions for inference or validation.
"""

# %%
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

MODELS_DIR = Path("models")
_MODEL_FILE = "model.joblib"
_META_FILE = "metadata.json"


# %%
@dataclass
class ModelMetadata:
    """Self-describing record persisted alongside each model version."""

    name: str
    algorithm: str
    created_at: str
    feature_names: list[str]
    threshold: float
    threshold_mode: str
    feature_mode: str           # "raw" | "binned_woe"
    metrics: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2, default=str)


# %%
def _slugify(name: str) -> str:
    """Filesystem-safe slug for a model name."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


# %%
def save_model(
    model: Any,
    metadata: ModelMetadata,
    models_dir: Path | str = MODELS_DIR,
) -> Path:
    """Persist ``model`` plus ``metadata`` to a fresh versioned folder.

    Returns the version directory created.
    """
    models_dir = Path(models_dir)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    version_dir = models_dir / f"{_slugify(metadata.name)}__{ts}"
    version_dir.mkdir(parents=True, exist_ok=False)

    metadata.created_at = metadata.created_at or ts
    joblib.dump(model, version_dir / _MODEL_FILE)
    (version_dir / _META_FILE).write_text(metadata.to_json(), encoding="utf-8")
    return version_dir


# %%
def list_versions(models_dir: Path | str = MODELS_DIR) -> list[dict[str, Any]]:
    """List saved versions (newest first) with their metadata.

    Each entry: ``{"path": Path, "metadata": dict}``. Folders missing a
    metadata file are skipped.
    """
    models_dir = Path(models_dir)
    if not models_dir.exists():
        return []
    versions: list[dict[str, Any]] = []
    for d in sorted(models_dir.iterdir(), reverse=True):
        meta_path = d / _META_FILE
        if d.is_dir() and meta_path.exists():
            versions.append(
                {"path": d, "metadata": json.loads(meta_path.read_text(encoding="utf-8"))}
            )
    return versions


# %%
def load_model(version_dir: Path | str) -> tuple[Any, dict[str, Any]]:
    """Load a saved version, returning ``(model, metadata_dict)``."""
    version_dir = Path(version_dir)
    model = joblib.load(version_dir / _MODEL_FILE)
    metadata = json.loads((version_dir / _META_FILE).read_text(encoding="utf-8"))
    return model, metadata
