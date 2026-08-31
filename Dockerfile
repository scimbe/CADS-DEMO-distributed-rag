# Single-container packaging for the marketplace#33 distributed-RAG demo's HTTP service
# (app.py). Mirrors manifests/litellm-proof/heartbeat-proxy/Dockerfile's pattern in
# CADS-agent-marketplace: python:3.12-slim, install requirements.txt, copy the app in,
# run it directly (no in-container build step, no compiled toolchain needed -- fastembed
# and lancedb both ship prebuilt wheels for this platform).
FROM python:3.12-slim

WORKDIR /app

# System deps: none beyond what python:3.12-slim already has -- fastembed's onnxruntime
# and lancedb both install as manylinux wheels, no compiler needed.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY rag/ ./rag/
COPY scripts/ ./scripts/
COPY static/ ./static/
COPY app.py .

# Everything that must survive a container restart -- the document-corpus SQLite store,
# the interaction-memory LanceDB table, and the one-time embedding-model download --
# lives under /app/data. Mount this as a volume (see docker-compose.yml) or the corpus
# and the downloaded model are lost every restart.
ENV RAG_DB_PATH=/app/data/rag.sqlite3 \
    RAG_MEMORY_PATH=/app/data/memory.lance \
    RAG_FASTEMBED_CACHE_DIR=/app/data/fastembed_cache \
    RAG_HOST=0.0.0.0 \
    RAG_PORT=8080
RUN mkdir -p /app/data

EXPOSE 8080

CMD ["python", "app.py"]
