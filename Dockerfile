# syntax=docker/dockerfile:1
# Use the same Python version as "requires-python" in your pyproject.toml
ARG PYTHON_VERSION=3.13
FROM python:${PYTHON_VERSION}-slim

# uv: fast, reproducible installs from uv.lock
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# libgomp is needed by faiss on slim images
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Run as a non-root user (uid 1000 also matches Hugging Face Spaces)
RUN useradd --create-home --uid 1000 app
WORKDIR /app
RUN chown app:app /app
USER app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    FASTEMBED_CACHE_PATH=/home/app/models \
    EMBED_CACHE_DB=/app/data/embeddings.db \
    PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_MAX_UPLOAD_SIZE=25

# 1) Dependencies first: this layer is cached until pyproject/uv.lock change
COPY --chown=app:app pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# 2) Then the app code (changes often, so it goes last)
COPY --chown=app:app README.md main.py ./
COPY --chown=app:app src ./src
RUN uv sync --frozen --no-dev

# 3) Bake the embedding model into the image so startup never waits on a download
RUN mkdir -p /app/data \
 && python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import os,urllib.request; urllib.request.urlopen('http://localhost:%s/_stcore/health' % os.environ.get('PORT','8501'))"

CMD ["sh", "-c", "exec streamlit run main.py --server.port=${PORT} --server.address=0.0.0.0"]