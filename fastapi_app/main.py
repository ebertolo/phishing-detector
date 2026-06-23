"""FastAPI application entry point.

Serves the best-trained phishing-detection model over HTTP, reusing the same
``ModelWrapper``/``persistence`` core used by the Streamlit app and the CLIs
(``src/phishing/core``) — no modeling logic is duplicated here.

Run locally
-----------
::

    uv run uvicorn fastapi_app.main:app --reload --port 8000

Then open http://localhost:8000/docs for the interactive Swagger UI (public,
no token needed) or http://localhost:8000/redoc for ReDoc. See
``fastapi_app/README.md`` for the full walkthrough, including authentication.
"""

# %%
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from phishing.core.persistence import list_versions
from phishing.core.wrapper import ModelWrapper

from .routers import health, predict

MODEL_VERSION_DIR_ENV_VAR = "MODEL_VERSION_DIR"

DESCRIPTION = """
Predicts the likelihood of an email/message being phishing from a small set
of count-based features (word/link/domain/email/spelling-error/urgent-keyword
counts). The primary output is a continuous probability
(`phishing_likelihood`), not a forced binary label — see `POST /predict` for
details.

**Authentication.** `POST /predict` requires an `Authorization: Bearer
<token>` header (see `GET /health` or the "Authorize" button above for how to
send it). `GET /health`, `/docs` and `/redoc` are public.
"""


# %%
def _resolve_model_version_dir() -> Path:
    """Pick the model version to serve: an explicit override, or the newest."""
    override = os.environ.get(MODEL_VERSION_DIR_ENV_VAR)
    if override:
        return Path(override)
    versions = list_versions()
    if not versions:
        raise RuntimeError(
            "No saved model versions found in ./models. Train and save one "
            "first — e.g. via the Streamlit 'Train & Compare' page, "
            "`uv run python scripts/cli.py train --csv <csv> --model <name>`, "
            "or `uv run python scripts/run_experiments.py --csv <csv> "
            "--save-best` — then restart this API. To pin a specific "
            f"version, set the {MODEL_VERSION_DIR_ENV_VAR} environment variable."
        )
    return versions[0]["path"]


# %%
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup; keep it in ``app.state`` for reuse."""
    version_dir = _resolve_model_version_dir()
    app.state.model_wrapper = ModelWrapper.load(version_dir)
    app.state.model_version_dir = version_dir.name
    yield
    app.state.model_wrapper = None


# %%
app = FastAPI(
    title="Phishing Detection API",
    description=DESCRIPTION,
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(predict.router)
