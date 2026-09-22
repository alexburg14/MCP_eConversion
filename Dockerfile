# e-Converse: FastAPI chat over the e-conversion corpus. Build from the repo root:
#   docker build --build-arg GIT_SHA=$(git rev-parse --short HEAD) \
#                --build-arg BUILD_TIME=$(date -u +%FT%TZ) -t econverse .
#   docker run -p 8501:8501 -v $PWD/data:/app/data -v $PWD/logs:/app/logs --env-file .env econverse
FROM python:3.12-slim

# Build provenance, shown in the stats panel and every telemetry line.
ARG GIT_SHA=unknown
ARG BUILD_TIME=
ENV GIT_SHA=${GIT_SHA} \
    BUILD_TIME=${BUILD_TIME} \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hf \
    LOG_DIR=/app/logs \
    PORT=8501
LABEL org.opencontainers.image.revision=${GIT_SHA} \
      org.opencontainers.image.created=${BUILD_TIME}

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# CPU torch first so sentence-transformers does not pull the CUDA build.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN mkdir -p /app/data /app/logs /app/.hf

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
  CMD curl -fsS "http://127.0.0.1:${PORT}${ROOT_PATH:-}/api/health" > /dev/null || exit 1

ENTRYPOINT ["/entrypoint.sh"]
