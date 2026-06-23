# Python 3.12 image with uv as the package manager (no pip used directly).
# 3.12 is required because optbinning/ortools have no 3.13 wheels yet.
FROM python:3.12-slim

# System libs required by LightGBM / XGBoost.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv (the project standard).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Resolve dependencies first for better layer caching.
COPY pyproject.toml ./
COPY src ./src
RUN uv sync --no-dev

# Application code.
COPY app ./app

EXPOSE 8501
EXPOSE 5000

# Run the Streamlit app as the primary process.
CMD ["uv", "run", "streamlit", "run", "app/streamlit_app.py", \
     "--server.address=0.0.0.0", "--server.port=8501"]
