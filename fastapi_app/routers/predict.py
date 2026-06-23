"""Prediction endpoint — the only protected route in this API."""

# %%
from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Request, status

from phishing.core.data import FEATURES

from ..auth import verify_token
from ..schemas import PredictionItem, PredictRequest, PredictResponse

router = APIRouter(tags=["predict"])


# %%
@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Predict the likelihood of phishing for one or more samples",
    dependencies=[Depends(verify_token)],
)
def predict(request: Request, payload: PredictRequest) -> PredictResponse:
    """Predict the likelihood of each sample being phishing.

    **Authentication required.** Send ``Authorization: Bearer <token>`` with
    the value of the ``API_TOKEN`` environment variable (default
    ``changeme-dev-token`` for local development) — see ``GET /health`` or
    ``fastapi_app/README.md`` for the full authentication walkthrough.

    **Request.** A JSON body with a ``samples`` list; each sample carries the
    eight raw count features the model was trained on (word/link/domain/
    email/spelling-error/urgent-keyword counts — see the schema below for
    each field's meaning). Feature engineering (presence flags, log
    transforms, ratios, and the frozen NN embedding when the loaded model
    uses one) is applied internally by the saved model pipeline; the caller
    only supplies the raw counts.

    **Response.** For every sample, the **primary output is
    `phishing_likelihood`** — a continuous probability in [0, 1] estimating
    how likely the sample is to be phishing, not a forced binary label. The
    `is_phishing` flag is a convenience threshold cut for simple filtering;
    `threshold_used` reports exactly where that cut was made so callers can
    re-derive a different operating point from the likelihood if needed.
    """
    wrapper = getattr(request.app.state, "model_wrapper", None)
    if wrapper is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No model is currently loaded. See GET /health.",
        )

    X = pd.DataFrame(
        [[getattr(s, col) for col in FEATURES] for s in payload.samples],
        columns=FEATURES,
    )
    proba = wrapper.predict_proba(X)
    threshold = wrapper.threshold

    predictions = [
        PredictionItem(
            phishing_likelihood=float(p),
            is_phishing=bool(p >= threshold),
            threshold_used=float(threshold),
        )
        for p in proba
    ]
    version_dir = getattr(request.app.state, "model_version_dir", None)
    return PredictResponse(
        predictions=predictions,
        model_name=wrapper.name,
        model_version=str(version_dir) if version_dir else "unknown",
    )
