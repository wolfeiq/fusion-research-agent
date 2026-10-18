# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:0.11.18 AS uv
FROM python:3.12-slim-bookworm AS base
COPY --from=uv /uv /uvx /usr/local/bin/
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=0 UV_LINK_MODE=copy UV_NO_CACHE=1 \
    PYTHONPATH=/app/flower-app HOME=/tmp
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 fusion && useradd --uid 10001 --gid fusion --no-create-home fusion
# .dockerignore is an allowlist. Runtime secrets, saved local runs and the
# original experiments are never sent to the builder or baked into the image.
COPY . /app/
WORKDIR /app/flower-app
RUN uv sync --locked --extra research --no-dev --no-editable \
    && mkdir -p .dashboard && chown fusion:fusion .dashboard
ENV PATH=/app/flower-app/.venv/bin:$PATH
USER 10001:10001
EXPOSE 8787
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["/app/flower-app/.venv/bin/python", "/app/deploy/healthcheck.py"]
CMD ["/app/flower-app/.venv/bin/python", "/app/deploy/entrypoint.py"]

# This stage is optional. The process stays in the same application container,
# using an isolated venv and per-trial source snapshot, not a separate service.
FROM base AS shadow
USER root
WORKDIR /app
RUN uv venv --python /usr/local/bin/python .venv \
    && uv pip install --python .venv torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu \
    && uv pip install --python .venv --constraint deploy/constraints-rl-cpu.txt ".[rl]" \
    && uv pip check --python .venv
WORKDIR /app/flower-app
USER 10001:10001

# Default is the smaller research image. Select shadow explicitly to enable
# actual CPU TORAX trials after validating its runtime on the target VM.
FROM base AS research
