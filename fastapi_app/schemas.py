"""Pydantic request/response models for the phishing-detection API.

Field descriptions are written to render directly in the Swagger UI
(``/docs``), so each one explains the feature in plain terms rather than
just naming it.
"""

# %%
from __future__ import annotations

from pydantic import BaseModel, Field


# %%
class PhishingFeatures(BaseModel):
    """The eight raw count features the model was trained on.

    These mirror ``phishing.core.data.FEATURES`` exactly — the API does not
    accept pre-engineered columns; the saved model pipeline derives its own
    engineered features (presence flags, log transforms, ratios, and the
    frozen NN embedding when applicable) internally from these raw counts.
    """

    num_words: int = Field(..., ge=0, description="Total word count in the email/message body.")
    num_unique_words: int = Field(..., ge=0, description="Count of distinct words (vocabulary size).")
    num_stopwords: int = Field(..., ge=0, description="Count of common stopwords (e.g. \"the\", \"is\").")
    num_links: int = Field(..., ge=0, description="Count of hyperlinks present in the message.")
    num_unique_domains: int = Field(..., ge=0, description="Count of distinct domains among the links.")
    num_email_addresses: int = Field(..., ge=0, description="Count of email addresses mentioned in the body.")
    num_spelling_errors: int = Field(..., ge=0, description="Count of detected spelling errors.")
    num_urgent_keywords: int = Field(..., ge=0, description="Count of urgency-signalling keywords (e.g. \"verify now\", \"act immediately\").")

    model_config = {
        "json_schema_extra": {
            "example": {
                "num_words": 120,
                "num_unique_words": 80,
                "num_stopwords": 40,
                "num_links": 3,
                "num_unique_domains": 2,
                "num_email_addresses": 1,
                "num_spelling_errors": 2,
                "num_urgent_keywords": 1,
            }
        }
    }


# %%
class PredictRequest(BaseModel):
    """A batch of samples to score in a single call."""

    samples: list[PhishingFeatures] = Field(
        ..., min_length=1,
        description="One or more emails/messages, each described by the eight raw count features.",
    )


# %%
class PredictionItem(BaseModel):
    """The model's output for a single sample.

    ``phishing_likelihood`` is the primary output of this API: a continuous
    probability in [0, 1] estimating how likely the sample is to be phishing,
    rather than a forced binary classification. ``is_phishing`` is a
    convenience flag derived from that likelihood at the model's tuned
    decision threshold — useful for simple pass/fail filtering, but the
    likelihood is what should drive any ranking or triage decision.
    """

    phishing_likelihood: float = Field(
        ..., ge=0.0, le=1.0,
        description="Primary output: the model's estimated probability that this sample is phishing.",
    )
    is_phishing: bool = Field(
        ..., description="Convenience label: phishing_likelihood >= threshold_used.",
    )
    threshold_used: float = Field(
        ..., description="The decision threshold the loaded model version was tuned to (see metadata.json).",
    )


# %%
class PredictResponse(BaseModel):
    """Response envelope for a prediction batch."""

    predictions: list[PredictionItem] = Field(
        ..., description="One prediction per input sample, in the same order as the request.",
    )
    model_name: str = Field(..., description="Name of the algorithm/ensemble serving this response.")
    model_version: str = Field(..., description="Identifier of the loaded model version directory.")


# %%
class HealthResponse(BaseModel):
    """Service liveness and currently-loaded model summary."""

    status: str = Field(..., description="\"ok\" when the service is up and a model is loaded.")
    model_loaded: bool = Field(..., description="Whether a model version was successfully loaded at startup.")
    model_name: str | None = Field(None, description="Name of the loaded model, if any.")
    model_version: str | None = Field(None, description="Loaded model version directory name, if any.")
