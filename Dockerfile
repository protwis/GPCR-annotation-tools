# ── Stage 1: Build ──────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir build \
    && python -m build --wheel

# ── Stage 2: Runtime ────────────────────────────────────────────────────
FROM python:3.12-slim

LABEL org.opencontainers.image.title="GPCR Annotation Tools"
LABEL org.opencontainers.image.description="Interactive CSV generator for GPCR structure annotation review"
LABEL org.opencontainers.image.source="https://github.com/protwis/GPCR-annotation-tools"

WORKDIR /app

# Install the wheel from the builder stage
COPY --from=builder /build/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl \
    && rm -rf /tmp/*.whl

# Create default mount points
RUN mkdir -p /data /output

# Set environment variables for the application
ENV GPCR_DATA_DIR=/data
ENV GPCR_OUTPUT_DIR=/output

# Default entry point: run the CSV generator
ENTRYPOINT ["gpcr-csv-generator"]
