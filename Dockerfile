# Rig Assist API image. Multi-stage: a builder installs dependencies with uv, the final
# image gets only the virtual environment (no uv, no build tools, no source tree, no tests).
# Images are pinned by tag and digest: a tag can be moved, a digest can't.

# ---- builder ---------------------------------------------------------------
FROM python:3.12.15-slim-trixie@sha256:a0f57064cbdcd6285fa45f51237ab0b35ea8ebaed06a1f5973e89e19dfb86704 AS builder
COPY --from=ghcr.io/astral-sh/uv:0.13.3@sha256:bee0a2a458fd956c5df7b8b4c5fed2257b0af34866e3da7b51db0be3acb375d1 /uv /bin/

# Compile .pyc at build time (faster start), copy instead of hard-linking from the cache
# mount, and use the image's Python rather than downloading one.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# 1. Dependencies only, from the lockfile. This layer is reused until uv.lock changes,
#    so editing code doesn't reinstall numpy, pdfplumber, ...
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev --no-editable

# 2. The project itself, installed non-editable so the final image needs no source tree.
COPY pyproject.toml uv.lock ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

# 3. tiktoken downloads its tokenizer file on first use; the chunker loads it at import.
#    Fetch it now so the container starts without internet access.
ENV TIKTOKEN_CACHE_DIR=/app/tiktoken
RUN /app/.venv/bin/python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

# ---- runtime ---------------------------------------------------------------
FROM python:3.12.15-slim-trixie@sha256:a0f57064cbdcd6285fa45f51237ab0b35ea8ebaed06a1f5973e89e19dfb86704

# Non-root user: if the app is ever compromised, it can't modify the image or the system.
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --no-create-home app

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/tiktoken /app/tiktoken

# Data lives in /app/data (a volume in docker compose); settings resolve data/... from /app.
RUN mkdir -p /app/data && chown app:app /app/data

ENV PATH="/app/.venv/bin:$PATH" \
    TIKTOKEN_CACHE_DIR=/app/tiktoken \
    PYTHONUNBUFFERED=1

USER app
EXPOSE 8000

# Liveness: the slim image has no curl, so ask Python. Docker marks the container unhealthy
# if /health stops answering.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]

CMD ["uvicorn", "rig_assist.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
