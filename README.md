# GPCR Annotation Tools

**An end-to-end, AI-assisted annotation and human-in-the-loop curation suite for GPCR structural biology.**

GPCR Annotation Tools automates the extraction of structured metadata from GPCR crystal and cryo-EM structures deposited in the PDB. It combines automated data enrichment, multi-run AI annotation with structured output, algorithmic cross-validation, and an interactive expert review dashboard to produce database-ready CSVs with full decision provenance.

---

## Pipeline at a Glance

```text
                        PDB IDs (targets.txt)
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  1. gpcr-tools fetch          Download RCSB metadata + enrich       │
│                                (UniProt, PubChem, CrossRef, SMILES) │
├─────────────────────────────────────────────────────────────────────┤
│  2. gpcr-tools fetch-papers   Download open-access PDFs             │
│                                (Unpaywall → PMC OA → abstract       │
│                                 fallback + manual watch mode)       │
├─────────────────────────────────────────────────────────────────────┤
│  3. gpcr-tools annotate       AI annotation via Gemini              │
│                                (10 independent runs per PDB,        │
│                                 structured output via tool calling) │
├─────────────────────────────────────────────────────────────────────┤
│  4. gpcr-tools aggregate      Majority-vote consensus + validation  │
│                                (7-validator chain against PDB/      │
│                                 UniProt/PubChem ground truth)       │
├─────────────────────────────────────────────────────────────────────┤
│  5. gpcr-tools curate         Interactive expert review dashboard   │
│                                (Rich terminal UI + audit trail)     │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
                          output/csv/
                    (database-ready CSVs)
```

Each step is **resumable** and **idempotent** — re-running any command skips already-completed work unless `--force` is passed.

---

## Key Features

### Data Enrichment (pre-annotation)

- **RCSB GraphQL integration** — Downloads comprehensive PDB metadata including polymer/nonpolymer entities, assemblies, citations, and experimental details.
- **Multi-source enrichment** — Automatically resolves UniProt entry names, PubChem CIDs + synonyms, SMILES/InChIKey descriptors, and sibling PDB structures sharing the same publication.
- **Persistent caching** — All external API responses are cached locally with atomic writes, eliminating redundant network calls across pipeline runs.
- **Tiered paper acquisition** — Fetches open-access PDFs via Unpaywall and NCBI PMC OA, with PubMed abstract fallback for paywalled papers and a live filesystem watcher for manual drops.

### AI Annotation

- **Multi-run consensus** — Each PDB is annotated 10 times independently (configurable via `--runs`), producing a statistically robust basis for majority voting.
- **Structured output via tool calling** — Gemini returns annotations in a strict JSON schema enforced by function calling, not free-form text. Every field (receptor identity, ligand roles, signaling partners, state classification) is constrained to defined types and enumerations.
- **Context-rich prompts** — The AI receives not just the paper PDF but also pre-enriched PDB metadata, a chain inventory reminder, and sibling structure warnings — reducing hallucination by grounding the model in API-verified facts.
- **Flexible model selection** — Switch models at runtime via `--model` flag or `GPCR_GEMINI_MODEL` environment variable without code changes.
- **Batch API support** — Large-scale annotation via Gemini Batch API with JSONL submission, polling, and automatic result recovery.
- **Rate-limited client** — Sliding-window rate limiting (1000 RPM) with exponential backoff on 429 responses.

### Post-Annotation Validation

- **7-validator chain** — Each aggregated annotation passes through a chain of cross-validation steps:
  1. **Chimera detection** — Identifies fusion constructs by comparing G-alpha C-terminal tails against UniProt reference sequences.
  2. **Receptor identity verification** — Validates UniProt entry names against the UniProt API.
  3. **Ligand existence check** — Confirms every annotated ligand exists in PDB Chemical Component Dictionary, filtering common buffers and crystallization artifacts.
  4. **Oligomer analysis** — Classifies complexes (monomer / homomer / heteromer), scans 7TM domain completeness per chain, suggests the primary protomer, and auto-corrects chain-ID assignments when API evidence disagrees with AI output.
  5. **Structural integrity** — Cross-checks internal consistency of the annotation structure.
  6. **Ground truth injection** — Overwrites method, resolution, and release date with PDB-authoritative values.
  7. **Controversy detection** — Flags fields where AI runs disagreed, with per-field vote breakdowns.

### Expert Curation

- **Rich terminal dashboard** — An ergonomic review interface built with [Rich](https://github.com/Textualize/rich) for rapid, informed decision-making.
- **Context-aware validation alerts** — Real-time display of ghost chains, hallucinated ligands, UniProt identity clashes, and chimera warnings alongside the data being reviewed.
- **Recursive review engine** — Navigate field-by-field through the annotation tree, with controversy highlights guiding attention to disputed values.
- **Append-only audit trail** — Every human decision (accept / edit / reject) is logged to `audit_trail.jsonl` with timestamps, providing full reproducibility.
- **Resumable sessions** — Curation progress is persisted; interrupted sessions resume exactly where they left off.

---

## Quick Start

### Option 1: Docker (Recommended)

```bash
# Pull the latest image
docker pull ghcr.io/protwis/gpcr-annotation-tools:latest

# Initialize a workspace
mkdir -p ~/gpcr_workspace
docker run --rm \
  -v ~/gpcr_workspace:/workspace \
  ghcr.io/protwis/gpcr-annotation-tools init-workspace

# Add PDB IDs to the target list
echo -e "8TII\n7W55\n9BLW" >> ~/gpcr_workspace/targets.txt

# Run the full pipeline
docker run --rm \
  -v ~/gpcr_workspace:/workspace \
  -e GPCR_GEMINI_API_KEY="$GPCR_GEMINI_API_KEY" \
  -e GPCR_EMAIL_FOR_APIS="you@example.com" \
  ghcr.io/protwis/gpcr-annotation-tools fetch

docker run --rm \
  -v ~/gpcr_workspace:/workspace \
  -e GPCR_EMAIL_FOR_APIS="you@example.com" \
  ghcr.io/protwis/gpcr-annotation-tools fetch-papers --auto-only

docker run --rm \
  -v ~/gpcr_workspace:/workspace \
  -e GPCR_GEMINI_API_KEY="$GPCR_GEMINI_API_KEY" \
  ghcr.io/protwis/gpcr-annotation-tools annotate

docker run --rm \
  -v ~/gpcr_workspace:/workspace \
  ghcr.io/protwis/gpcr-annotation-tools aggregate

docker run -it --rm \
  -v ~/gpcr_workspace:/workspace \
  ghcr.io/protwis/gpcr-annotation-tools curate
```

> **Note:** The `-it` flags are required only for the interactive `curate` command. Pass `--user "$(id -u):$(id -g)"` to avoid root-owned files on the host.

### Option 2: Local Installation

Requires Python 3.11+.

```bash
git clone https://github.com/protwis/GPCR-annotation-tools.git
cd GPCR-annotation-tools

# Install with all optional dependencies
pip install -e ".[dev]"

# Configure
export GPCR_WORKSPACE=~/gpcr_workspace
export GPCR_GEMINI_API_KEY=your-api-key
export GPCR_EMAIL_FOR_APIS=you@example.com

# Initialize and run
gpcr-tools init-workspace

gpcr-tools fetch
gpcr-tools fetch-papers
gpcr-tools annotate
gpcr-tools aggregate
gpcr-tools curate
```

---

## CLI Reference

### `gpcr-tools fetch`

Download PDB metadata from RCSB GraphQL and enrich with UniProt, PubChem, and CrossRef data.

```bash
gpcr-tools fetch                        # Process all targets
gpcr-tools fetch 8TII                   # Single PDB
gpcr-tools fetch --targets ids.txt      # Custom target file
gpcr-tools fetch --force                # Re-fetch existing entries
```

### `gpcr-tools fetch-papers`

Download open-access papers with tiered fallback (Unpaywall → PMC OA → abstract).

```bash
gpcr-tools fetch-papers                 # All targets, with watch mode for paywalled papers
gpcr-tools fetch-papers --auto-only     # Skip watch mode (for CI/scripting)
gpcr-tools fetch-papers 8TII           # Single PDB
```

### `gpcr-tools annotate`

Run Gemini AI annotation with structured output.

```bash
gpcr-tools annotate                                    # Auto-discover pending PDBs
gpcr-tools annotate 8TII --runs 5                      # Single PDB, 5 runs
gpcr-tools annotate --model gemini-2.5-flash            # Use a different model
gpcr-tools annotate --prompt prompts/custom.txt         # Custom prompt template
gpcr-tools annotate --batch                             # Submit via Batch API
gpcr-tools annotate --check-batch                       # Poll batch status
gpcr-tools annotate --recover                           # Re-process raw batch output
```

### `gpcr-tools aggregate`

Aggregate multi-run AI results with majority voting and cross-validation.

```bash
gpcr-tools aggregate                    # All pending PDBs
gpcr-tools aggregate 8TII              # Single PDB
gpcr-tools aggregate --skip-api-checks  # Offline mode (no UniProt/PubChem calls)
gpcr-tools aggregate --force            # Re-process already-aggregated entries
```

### `gpcr-tools curate`

Interactive expert review dashboard.

```bash
gpcr-tools curate                       # Review all pending PDBs
gpcr-tools curate 8TII                  # Target a single PDB
gpcr-tools curate --auto-accept         # Non-interactive mode (CI/testing)
```

---

## Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GPCR_WORKSPACE` | No | Workspace root (default: `/workspace`) |
| `GPCR_GEMINI_API_KEY` | For `annotate` | Google Gemini API key |
| `GPCR_GEMINI_MODEL` | No | Model override (default: `gemini-2.5-pro`) |
| `GPCR_EMAIL_FOR_APIS` | For `fetch-papers` | Email for Unpaywall/NCBI polite access |

<details>
<summary>Advanced: per-directory path overrides</summary>

For non-standard workspace layouts (e.g., separate storage mounts), each subdirectory can be overridden independently:

| Variable | Default |
|----------|---------|
| `GPCR_RAW_PATH` | `{workspace}/raw` |
| `GPCR_ENRICHED_PATH` | `{workspace}/enriched` |
| `GPCR_PAPERS_PATH` | `{workspace}/papers` |
| `GPCR_AI_RESULTS_PATH` | `{workspace}/ai_results` |
| `GPCR_AGGREGATED_PATH` | `{workspace}/aggregated` |
| `GPCR_OUTPUT_PATH` | `{workspace}/output` |
| `GPCR_CACHE_PATH` | `{workspace}/cache` |
| `GPCR_STATE_PATH` | `{workspace}/state` |
| `GPCR_TMP_PATH` | `{workspace}/tmp` |

</details>

---

## Workspace Layout

```text
/workspace/
├── contract/storage_contract.json    # Versioned workspace contract
├── targets.txt                       # PDB IDs to process (one per line)
├── prompts/v5.txt                    # Default annotation prompt template
│
├── raw/pdb_json/                     # RCSB GraphQL responses
├── enriched/                         # Enriched PDB metadata (AI input)
├── papers/                           # Downloaded PDFs and abstracts
├── ai_results/{pdb_id}/run_*.json   # 10 independent AI annotation runs
│
├── aggregated/                       # Voted + validated annotations
│   ├── {pdb_id}.json
│   ├── logs/                         # Per-field voting discrepancy logs
│   └── validation_logs/              # Algorithmic validation reports
│
├── output/
│   ├── csv/                          # Database-ready CSV exports
│   └── audit/audit_trail.jsonl       # Append-only decision provenance
│
├── cache/                            # Persistent API caches
└── state/                            # Operational state (resumability)
```

---

## Output Artifacts

### Database CSVs (`output/csv/`)

Tab-separated, normalized files ready for database ingestion:

| File | Contents |
|------|----------|
| `structures.csv` | PDB ID, receptor UniProt, method, resolution, state, chain, date |
| `ligands.csv` | Ligand names, PubChem IDs, roles, types, SMILES, InChIKey, sequences |
| `g_proteins.csv` | G-protein subunit UniProt IDs and chain assignments |
| `arrestins.csv` | Arrestin UniProt IDs and chains |
| `fusion_proteins.csv` | Fusion protein names |
| `nanobodies.csv`, `antibodies.csv`, `scfv.csv` | Binding partner names |
| `grk.csv`, `ramp.csv`, `other_aux_proteins.csv` | Auxiliary protein names |

### Validation Reports (`aggregated/validation_logs/`)

Per-PDB structured reports containing:
- **Critical warnings** — hallucinated ligands, chimeric fusion proteins, identity clashes
- **Algorithmic conflicts** — AI annotation vs. API ground truth disagreements
- **Oligomer analysis** — complex classification, 7TM completeness, chain corrections

### Provenance Logs

| Log | Purpose |
|-----|---------|
| `output/audit/audit_trail.jsonl` | Every human decision, timestamped and append-only |
| `aggregated/logs/*_voting_log.json` | Per-field majority-vote breakdowns across 10 AI runs |
| `state/processed_log.json` | Curation completion status (enables resumable sessions) |

---

## Architecture

```text
src/gpcr_tools/
├── config.py                  # All constants, URLs, timeouts, thresholds
├── workspace.py               # Workspace initialization & contract validation
├── __main__.py                # CLI entry point
│
├── fetcher/                   # Stage 1: RCSB download + enrichment
│   ├── rcsb_client.py         #   GraphQL query + rate-limited download
│   ├── enricher.py            #   UniProt / PubChem / CrossRef enrichment
│   └── cache.py               #   Atomic JSON cache with version invalidation
│
├── papers/                    # Stage 2: Paper acquisition
│   ├── downloader.py          #   Tiered PDF download (Unpaywall → PMC → abstract)
│   └── watcher.py             #   Filesystem watcher for manual PDF drops
│
├── annotator/                 # Stage 3: Gemini AI annotation
│   ├── gemini_client.py       #   Rate-limited API client
│   ├── prompt_builder.py      #   Context-rich prompt assembly
│   ├── schema.py              #   Structured output schema (tool calling)
│   ├── pdf_compressor.py      #   Ghostscript compression for large PDFs
│   ├── post_processor.py      #   Response normalization
│   └── runner.py              #   Single-call + batch modes with recovery
│
├── aggregator/                # Stage 4: Consensus + validation
│   ├── voting.py              #   Majority-vote engine + controversy detection
│   ├── ground_truth.py        #   PDB/UniProt ground truth injection
│   └── runner.py              #   12-step orchestration with error isolation
│
├── validator/                 # 7-validator cross-check chain
│   ├── chimera.py             #   Fusion protein detection (C-terminal tail matching)
│   ├── receptor_validator.py  #   UniProt identity verification
│   ├── ligand_validator.py    #   PDB-CCD existence check
│   ├── oligomer.py            #   Complex classification + 7TM completeness
│   ├─�� integrity_checker.py   #   Structural consistency validation
│   └── api_clients.py         #   Shared API wrappers with retry + caching
│
└── csv_generator/             # Stage 5: Expert curation
    ├── app.py                 #   Main curation loop
    ├── review_engine.py       #   Recursive review tree
    ├── ui.py                  #   Rich terminal panels
    ├── csv_writer.py          #   Pure data → CSV export
    └── audit.py               #   JSONL audit trail writer
```

### Design Principles

| Principle | Implementation |
|-----------|---------------|
| **Atomic writes** | `tempfile` + `os.replace` + `try/finally` cleanup — no partial outputs |
| **Mutation isolation** | `deepcopy()` boundary before validator invocations |
| **None-safety** | `(data.get(key) or {}).get(child)` — never `.get(key, {})` on external data |
| **Centralized configuration** | All URLs, timeouts, thresholds, and magic strings in `config.py` |
| **Immutable constants** | `frozenset`, `tuple`, `MappingProxyType` for module-level data |
| **Error isolation** | Each PDB wrapped in `try/except` — failures logged, pipeline continues |
| **Timeout-guarded I/O** | Every HTTP call has an explicit timeout; sessions use `urllib3.Retry` |

---

## Development

### Prerequisites

```bash
pip install -e ".[dev]"
```

### Quality Gates

```bash
# Lint + format
ruff check src/ tests/
ruff format src/ tests/

# Type checking
mypy src/

# Tests
pytest tests/ -v
```

### Test Suite

The test suite includes 770+ tests:

- **Unit tests** for every module across all five pipeline stages
- **Integration tests** for the full aggregation pipeline, error isolation, and atomic write safety
- **Real PDB fixture tests** covering 9 canonical GPCR structures (5G53, 8TII, 9AS1, 9BLW, 9EJZ, 9IQS, 9M88, 9NOR, 9O38) with 10 AI runs each
- **Mock HTTP** for external APIs in the default test suite; live network integration tests are gated and skipped unless `GPCR_RUN_LIVE_TESTS=1` is set

### CI/CD

GitHub Actions workflows run on every push and pull request:

- **Ruff** — Enforced linting and formatting
- **mypy** — Static type checking with `ignore_missing_imports = false`
- **pytest** — Test matrix across Python 3.11 and 3.12
- **Docker smoke tests** — Build + exercise `init-workspace`, `curate --help`, and `curate --auto-accept`
- **Automated releases** — Docker image published to GHCR on semantic version tags (`v*`)

---

## License

This project is licensed under the [Apache License 2.0](LICENSE).
