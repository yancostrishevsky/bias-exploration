# syntax=docker/dockerfile:1

FROM python:3.11-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Build deps for compiling wheels (if any)
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
    && rm -rf /var/lib/apt/lists/*

# Build wheels for the project (and its deps)
COPY pyproject.toml README.md /app/
COPY ai_bias_search /app/ai_bias_search

RUN python -m pip install --upgrade pip \
  && python -m pip wheel --wheel-dir /wheels .


FROM python:3.11-slim AS runtime

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Runtime deps + install "just" (APT package usually not available on Debian slim)
ARG JUST_VERSION=1.34.0
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      tar \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL \
        "https://github.com/casey/just/releases/download/${JUST_VERSION}/just-${JUST_VERSION}-x86_64-unknown-linux-musl.tar.gz" \
      | tar -xz -C /usr/local/bin just \
    && chmod +x /usr/local/bin/just \
    && just --version

# Non-root user
RUN addgroup --system app && adduser --system --ingroup app app

# Install wheels
COPY --from=builder /wheels /wheels
RUN python -m pip install --no-cache-dir /wheels/* \
  && rm -rf /wheels

# Copy the rest of the repo (scripts/configs/templates/etc.)
COPY . /app

# Create writable dirs + permissions
RUN mkdir -p /app/data /app/results \
  && chown -R app:app /app

# Ensure entrypoint is executable (do this before USER)
RUN chmod +x /app/scripts/entrypoint.sh

USER app

ENTRYPOINT ["/app/scripts/entrypoint.sh"]
CMD ["python", "-m", "ai_bias_search.cli"]
