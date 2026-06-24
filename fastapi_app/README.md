# Phishing Detection API

A thin FastAPI service that serves the best-trained phishing-detection model
over HTTP. It reuses the same `ModelWrapper`/`persistence` core that the
Streamlit app and the CLIs use (`src/phishing/core`) — this package contains
no modeling logic, only request/response handling.

## Prerequisite — train and save a model version

The API loads a saved model version at startup; it does not train anything
itself. If `./models/` is empty, the API fails to start with a clear error.
Save a version first using any of:

- Streamlit's **Train & Compare** page → "Save selected model" (see the
  project root [README.md](../README.md#quickstart-uv-is-the-project-standard--no-pip)).
- `uv run python scripts/cli.py train --csv data/email_phishing_data.csv --model lightgbm`
- `uv run python scripts/run_experiments.py --csv data/email_phishing_data.csv --save-best`

## Run locally

```bash
uv run uvicorn fastapi_app.main:app --reload --port 8000
```

- Swagger UI (interactive docs, **no token needed**): http://localhost:8000/docs
- ReDoc (read-only docs, **no token needed**): http://localhost:8000/redoc
- Health check (**no token needed**): http://localhost:8000/health

By default the API loads the **most recently saved** model version (same
logic as `scripts/cli.py infer`). To pin a specific version instead, set
`MODEL_VERSION_DIR` before launching:

```bash
# bash / zsh (Linux / macOS / Git Bash)
export MODEL_VERSION_DIR=models/lightgbm__20260619T230000Z
uv run uvicorn fastapi_app.main:app --port 8000
```

```powershell
# PowerShell (Windows)
$env:MODEL_VERSION_DIR = "models/lightgbm__20260619T230000Z"
uv run uvicorn fastapi_app.main:app --port 8000
```

Alternatively, create a `.env` file in the project root and source it before
running (the app reads the variable via `os.environ` — no dotenv library is
needed, just export it in your shell before `uv run`):

```bash
# .env (source manually: source .env  or  . .env)
export MODEL_VERSION_DIR=models/lightgbm__20260619T230000Z
export API_TOKEN=changeme-dev-token
```

## Authentication

`POST /predict` is the only protected endpoint. It expects an
`Authorization: Bearer <token>` header, where `<token>` is the value of the
`API_TOKEN` environment variable.

```bash
# Default for local development (do not use this value anywhere reachable):
export API_TOKEN=changeme-dev-token
```

In the Swagger UI, click **Authorize** at the top of the page and paste the
token — every subsequent "Try it out" call on `/predict` will include it
automatically. `GET /health`, `/docs` and `/redoc` need no authentication.

### Example: calling `/predict` with curl

```bash
curl -X POST http://localhost:8000/predict \
    -H "Authorization: Bearer changeme-dev-token" \
    -H "Content-Type: application/json" \
    -d '{
      "samples": [
        {
          "num_words": 120,
          "num_unique_words": 80,
          "num_stopwords": 40,
          "num_links": 3,
          "num_unique_domains": 2,
          "num_email_addresses": 1,
          "num_spelling_errors": 2,
          "num_urgent_keywords": 1
        }
      ]
    }'
```

Response (`phishing_likelihood` is the primary output — a continuous
probability, not a forced 0/1 label):

```json
{
  "predictions": [
    {
      "phishing_likelihood": 0.0734,
      "is_phishing": false,
      "threshold_used": 0.29
    }
  ],
  "model_name": "blend",
  "model_version": "blend__20260623T101530Z"
}
```

## Tests

```bash
uv run pytest fastapi_app/tests/
```

Tests train a small model on synthetic data, save it to a temporary
`models/` directory, point `MODEL_VERSION_DIR` at it, and exercise
`/health`, `/predict`, and the auth dependency in isolation — no real,
pre-trained model is required to run the test suite.

## Endpoints summary

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/health` | none | Service + loaded-model status |
| POST | `/predict` | Bearer token | Predict phishing likelihood for a batch of samples |
| GET | `/docs` | none | Swagger UI |
| GET | `/redoc` | none | ReDoc |
