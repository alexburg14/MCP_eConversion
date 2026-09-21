#!/bin/bash
# Build any missing cache, then serve. Caches live on the mounted /app/data
# volume, so a fresh container with an existing volume starts immediately.
set -e
cd /app

build_if_missing() {
    local cache="$1" script="$2"
    if [ ! -f "/app/data/cache/$cache" ]; then
        echo "[entrypoint] $cache missing — running $script"
        python "src/scripts/$script"
    fi
}
mkdir -p /app/data/cache /app/data/feedback /app/logs
build_if_missing abstracts_cache.json build_abstracts_cache.py
build_if_missing pis_cache.json build_pis_cache.py
build_if_missing embeddings_cache.npz build_embeddings_cache.py
build_if_missing collaboration_graph.json build_graph_cache.py
build_if_missing proposal_summary.md extract_proposal_summary.py

# ROOT_PATH (e.g. /nomad-oasis/api/everse) mounts the app under a prefix when
# the reverse proxy forwards the full path. Sessions are in-memory: one worker.
echo "[entrypoint] Starting e-Converse on :${PORT:-8501} (root path '${ROOT_PATH:-/}')"
exec uvicorn main:app --app-dir src --host 0.0.0.0 --port "${PORT:-8501}" --workers 1 \
    --proxy-headers --forwarded-allow-ips='*' --timeout-keep-alive 75
