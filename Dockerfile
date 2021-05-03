FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    NEWSROOM_ARTIFACTS_DIR=/app/artifacts

WORKDIR /app

RUN useradd --create-home --uid 1000 newsroom

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY configs ./configs

RUN pip install --upgrade pip && pip install .

# Train a small default model and index at build time so the API is usable immediately.
RUN newsroom generate-corpus --n-docs 1200 --output /app/artifacts/corpus.csv \
    && newsroom train --input /app/artifacts/corpus.csv --output /app/artifacts/classifier.joblib --no-grid-search \
    && newsroom build-index --input /app/artifacts/corpus.csv --output /app/artifacts/index \
    && chown -R newsroom:newsroom /app/artifacts

USER newsroom
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import httpx,sys; sys.exit(0 if httpx.get('http://127.0.0.1:8000/health').status_code==200 else 1)"

CMD ["newsroom", "serve", "--host", "0.0.0.0", "--port", "8000", "--artifacts", "/app/artifacts"]
