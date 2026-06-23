"""Public health-check endpoint."""

# %%
from __future__ import annotations

from fastapi import APIRouter, Request

from ..schemas import HealthResponse

router = APIRouter(tags=["health"])


# %%
@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service and model status",
)
def health(request: Request) -> HealthResponse:
    """Report whether the service is up and which model version is loaded.

    This endpoint requires **no authentication** — unlike ``/predict``, it is
    meant for uptime checks and load balancers. The other endpoints in this
    API are protected: they expect an ``Authorization: Bearer <token>``
    header, where ``<token>`` is the value of the ``API_TOKEN`` environment
    variable (see ``fastapi_app/auth.py`` and the "Authorize" button at the
    top of this Swagger page for details).
    """
    wrapper = getattr(request.app.state, "model_wrapper", None)
    version_dir = getattr(request.app.state, "model_version_dir", None)
    if wrapper is None:
        return HealthResponse(status="degraded", model_loaded=False)
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_name=wrapper.name,
        model_version=version_dir,
    )
