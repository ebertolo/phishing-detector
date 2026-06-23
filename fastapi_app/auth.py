"""Fixed-token bearer authentication for protected endpoints.

How to authenticate
--------------------
Every protected endpoint (currently ``POST /predict``) requires an
``Authorization: Bearer <token>`` header. The token is a single, fixed value
read from the ``API_TOKEN`` environment variable (default
``"changeme-dev-token"`` for local development — override it before deploying
anywhere reachable). There is no login/token-issuance endpoint because the
token never changes at runtime; you already have it, you just send it.

Example::

    curl -X POST http://localhost:8000/predict \\
        -H "Authorization: Bearer changeme-dev-token" \\
        -H "Content-Type: application/json" \\
        -d '{"samples": [...]}'

In the Swagger UI (``/docs``), click "Authorize" and paste the token — every
subsequent "Try it out" call on a protected endpoint will include it
automatically.
"""

# %%
from __future__ import annotations

import os

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

API_TOKEN_ENV_VAR = "API_TOKEN"
DEFAULT_DEV_TOKEN = "changeme-dev-token"

# HTTPBearer registers the "Authorization: Bearer ..." scheme with FastAPI's
# OpenAPI generation, which is what makes the Swagger UI show the padlock
# icon on protected endpoints and the "Authorize" button.
_security = HTTPBearer(
    description="Send the API token configured via the API_TOKEN environment "
    "variable as a Bearer token. Default for local development: "
    f"'{DEFAULT_DEV_TOKEN}'.",
)


# %%
def _expected_token() -> str:
    """Read the current expected token from the environment.

    Read on every call (not cached at import time) so tests can override
    ``API_TOKEN`` per-case via ``monkeypatch.setenv``.
    """
    return os.environ.get(API_TOKEN_ENV_VAR, DEFAULT_DEV_TOKEN)


# %%
def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(_security),
) -> str:
    """FastAPI dependency that validates the Bearer token on a protected route.

    Send the configured token as ``Authorization: Bearer <token>`` on every
    protected request. Set the ``API_TOKEN`` environment variable to override
    the development default (``changeme-dev-token``) before deploying.
    Raises **401 Unauthorized** when the token is missing or does not match.
    """
    if credentials.credentials != _expected_token():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API token. Send 'Authorization: Bearer <token>' "
            "with the value of the API_TOKEN environment variable.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials
