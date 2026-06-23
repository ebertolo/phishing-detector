# Python 3.12 image with uv as the package manager (no pip used directly).
# 3.12 is required because optbinning/ortools have no 3.13 wheels yet.
FROM python:3.12-slim

# System libs required by LightGBM / XGBoost.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv into the image (the project standard — pip is never used directly).
# The base python:3.12-slim image does not ship uv, so we copy the static uv/uvx
# binaries from Astral's official image into /bin. Pinned to a fixed version
# (not :latest) so the build is reproducible.
COPY --from=ghcr.io/astral-sh/uv:0.6.14 /uv /uvx /bin/

WORKDIR /app

# Resolve dependencies first for better layer caching. Copy the lockfile so the
# build is reproducible (uv installs the exact pinned versions, not a fresh
# resolution); --locked fails the build if pyproject.toml and uv.lock disagree.
COPY pyproject.toml uv.lock ./
COPY src ./src
RUN uv sync --no-dev --locked

# Application code for every service that shares this image:
#   app       -> app/ (Streamlit)            mlflow-ui -> (no extra code)
#   api       -> fastapi_app/ (FastAPI)      CLIs      -> scripts/
COPY app ./app
COPY fastapi_app ./fastapi_app
COPY scripts ./scripts

EXPOSE 8501
EXPOSE 5000
EXPOSE 8000

# Run the Streamlit app as the primary process.
CMD ["uv", "run", "streamlit", "run", "app/streamlit_app.py", \
     "--server.address=0.0.0.0", "--server.port=8501"]
