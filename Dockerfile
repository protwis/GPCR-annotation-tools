# ── Stage 1: Build ──────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build
COPY pyproject.toml README.md LICENSE ./
COPY src/ src/
RUN pip install --no-cache-dir build \
    && python -m build --wheel

# ── Stage 2: Runtime ────────────────────────────────────────────────────
FROM python:3.12-slim

LABEL org.opencontainers.image.title="GPCR Annotation Tools"
LABEL org.opencontainers.image.description="Human-in-the-loop curation suite for GPCR structure annotations"
LABEL org.opencontainers.image.source="https://github.com/protwis/GPCR-annotation-tools"

WORKDIR /app

COPY --from=builder /build/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl \
    && rm -rf /tmp/*.whl

ENV GPCR_WORKSPACE=/workspace

ENTRYPOINT ["gpcr-tools"]
CMD ["curate"]
