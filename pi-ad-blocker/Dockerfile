# Pi Ad Blocker — Multi-stage Docker build
# Target: Raspberry Pi 5 (arm64) or x86_64 testing

# ---- Stage 1: Build ----
FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir onnx onnxruntime-tools

# ---- Stage 2: Runtime ----
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="Pi Ad Blocker"
LABEL org.opencontainers.image.description="Raspberry Pi Advertisement Blocking System"
LABEL org.opencontainers.image.version="1.0.0"

# Install runtime system deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only what's needed
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/
COPY data/ data/

# Use arm64 base for Raspberry Pi
FROM --platform=linux/arm64 python:3.11-slim AS pi-runtime

LABEL org.opencontainers.image.title="Pi Ad Blocker (RPi)"
LABEL org.opencontainers.image.description="Raspberry Pi 5 optimized build"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libgomp1 \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/
COPY data/ data/

# Default command
ENTRYPOINT ["python", "-m", "pi_ad_blocker.cli"]
CMD ["--help"]