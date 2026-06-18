# ── Stage 1: Build ──────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build
COPY pyproject.toml README.md LICENSE ./
COPY src/ src/
RUN pip install --no-cache-dir build \
    && python -m build --wheel

# ── Stage 2: Test ───────────────────────────────────────────────────────
# Hermetic pytest environment. Built with `--target test` (compose `test`
# service). Live src/ and tests/ are bind-mounted at run time so edits are
# picked up without a rebuild (editable install). Kept BEFORE the runtime
# stage so a plain `docker build .` still yields the production image.
FROM python:3.12-slim AS test

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src/ src/
COPY tests/ tests/
RUN pip install --no-cache-dir -e ".[dev]"
CMD ["pytest", "tests", "-q"]

# ── Stage 3: Runtime ────────────────────────────────────────────────────
FROM python:3.12-slim

LABEL org.opencontainers.image.title="GPCR Annotation Tools"
LABEL org.opencontainers.image.description="Human-in-the-loop curation suite for GPCR structure annotations"
LABEL org.opencontainers.image.source="https://github.com/protwis/GPCR-annotation-tools"

WORKDIR /app

# Ghostscript ('gs') is required to compress PDFs over the size limit before
# sending them to the model; without it large PDFs fail at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ghostscript \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/dist/*.whl /tmp/
# Install with the annotate + papers extras so the full pipeline works, not just
# curate: google-genai for annotation, PyMuPDF/lxml for paper parsing. Without
# them the runtime image can fetch and curate but cannot annotate or read PDFs.
RUN pip install --no-cache-dir "$(ls /tmp/*.whl)[annotate,papers]" \
    && rm -rf /tmp/*.whl

ENV GPCR_WORKSPACE=/workspace

# Bake the source commit into the image so each annotation's provenance can name
# the exact code that produced it. The .git directory and git binary are absent
# here, so this is the only way the runtime can know its version. Defaults to
# "unknown" when built without --build-arg GIT_SHA=...
ARG GIT_SHA=unknown
ENV GPCR_CODE_VERSION=$GIT_SHA

ENTRYPOINT ["gpcr-tools"]
CMD ["curate"]
