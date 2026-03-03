FROM python:3.11-slim

# Metadata
LABEL org.opencontainers.image.title="Bakup Backend"
LABEL org.opencontainers.image.version="0.1.0"
LABEL org.opencontainers.image.description="Local-first incident intelligence — backend API"

# Prevent interactive prompts during package install
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies for llama-cpp-python (CPU build) and sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies before copying source so Docker can cache this layer.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# ── Security: run as non-root ─────────────────────────────────────────────────
RUN useradd --create-home --shell /bin/bash bakup
USER bakup

# ── Runtime ───────────────────────────────────────────────────────────────────
# BAKUP_ACCESS_KEY and BAKUP_PROJECT_PATH must be supplied at container run time.
# The app will exit immediately if either is missing or the key is invalid.
EXPOSE 8000

# tini handles signal forwarding and zombie reaping cleanly
ENTRYPOINT ["python", "main.py"]
