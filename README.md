# GPCR Annotation Tools

**A production-grade, human-in-the-loop curation suite for GPCR structural biology.**

GPCR Annotation Tools provides GPCR structural biology experts with a robust, interactive command-line dashboard to review, validate, and export AI-assisted structural metadata into database-ready CSVs. Designed for high-throughput expert curation, the suite seamlessly integrates automated quality checks, algorithmic conflict resolution, and comprehensive decision provenance.

## Key Features

* **Multi-Run Aggregation:** Ingests 10 independent AI annotation runs per PDB, selects the highest-scoring run, and applies majority-vote consensus with per-field controversy detection.
* **Algorithmic Validation:** A chain of validators cross-checks annotations against PDB and UniProt APIs — detecting chimeric fusion proteins, hallucinated ligands, incorrect receptor identities, and oligomeric assembly misclassifications.
* **Oligomer Analysis:** Classifies complexes (monomer / homomer / heteromer), scans 7TM domain completeness per chain, suggests primary protomers, and auto-corrects chain-ID assignments when API evidence disagrees with AI output.
* **Interactive Expert Review:** A rich, ergonomic terminal UI designed for rapid curation of GPCR complex structural data, signaling partners, and ligands.
* **Integrated Validation Alerts:** Real-time, context-aware alerts for structural discrepancies (e.g., ghost chains, hallucinated ligands, and UniProt identity clashes).
* **Provable Audit Trails:** Every intervention — whether automated correction or human decision — is securely logged to a JSONL audit trail.
* **Docker-Native Deployment:** Zero-configuration setup for curators via a single workspace mount.

## Pipeline Overview

```text
ai_results/         enriched/         PDB / UniProt APIs
  (10 runs)           (metadata)            │
     │                   │                  │
     ▼                   ▼                  ▼
┌─────────────────────────────────────────────────┐
│              gpcr-tools aggregate               │
│                                                 │
│  1. Load & score AI runs                        │
│  2. Majority-vote consensus                     │
│  3. Ground truth injection                      │
│  4. Chimera detection                           │
│  5. Receptor identity validation                │
│  6. Ligand PDB-CCD cross-check                  │
│  7. Oligomer analysis & chain correction        │
│  8. Structural integrity checks                 │
│  9. Atomic output writes                        │
└─────────────┬───────────────────────────────────┘
              │
              ▼
         aggregated/
      (validated JSON)
              │
              ▼
┌─────────────────────────────────────────────────┐
│              gpcr-tools curate                  │
│                                                 │
│  Interactive expert review with validation      │
│  alerts, controversy resolution, and            │
│  decision provenance logging                    │
└─────────────┬───────────────────────────────────┘
              │
              ▼
         output/csv/
    (database-ready CSVs)
```

## Quick Start

### Option 1: Docker (Recommended for Curators)

The tool runs completely inside a container. You only need to mount a single workspace directory.

```bash
# Pull the latest production image
docker pull ghcr.io/protwis/gpcr-annotation-tools:latest

# Initialize a workspace (creates directory structure and contract file)
mkdir -p /path/to/gpcr_workspace
docker run --rm \
  -v /path/to/gpcr_workspace:/workspace \
  ghcr.io/protwis/gpcr-annotation-tools init-workspace

# Aggregate AI runs and validate against PDB/UniProt
docker run --rm \
  -v /path/to/gpcr_workspace:/workspace \
  ghcr.io/protwis/gpcr-annotation-tools aggregate

# Run the interactive curation dashboard
docker run -it --rm \
  -v /path/to/gpcr_workspace:/workspace \
  ghcr.io/protwis/gpcr-annotation-tools curate

# Target a specific PDB entry directly
docker run -it --rm \
  -v /path/to/gpcr_workspace:/workspace \
  ghcr.io/protwis/gpcr-annotation-tools curate 8TII
```

> **Note:** The `-it` (interactive + TTY) flags are required for the interactive review dashboard. The `aggregate` command does not need `-it`.

> **Tip:** To avoid root-owned files on the host, pass `--user "$(id -u):$(id -g)"` to `docker run`.

### Option 2: Local Installation (For Developers)

Requires Python 3.11+.

```bash
# Clone the repository
git clone https://github.com/protwis/GPCR-annotation-tools.git
cd GPCR-annotation-tools

# Install the package and dependencies
pip install -e ".[dev]"

# Point to your workspace directory
export GPCR_WORKSPACE=/path/to/gpcr_workspace

# Initialize the workspace (creates directory tree and contract file)
gpcr-tools init-workspace

# Aggregate all pending PDBs
gpcr-tools aggregate

# Aggregate a single PDB (offline mode, no API calls)
gpcr-tools aggregate 8TII --skip-api-checks

# Re-aggregate already-processed PDBs
gpcr-tools aggregate --force

# Launch the interactive curation dashboard
gpcr-tools curate

# Target a specific PDB
gpcr-tools curate 8Y72
```

## Workspace Layout

All commands operate under a single workspace root (`/workspace` inside Docker, or `GPCR_WORKSPACE` locally). The workspace must be initialized before first use:

```text
/workspace/
├── contract/
│   └── storage_contract.json   # Versioned workspace contract (required)
├── raw/                        # Downloaded source data
├── enriched/                   # Normalized and enriched PDB metadata
├── papers/                     # Paper files and metadata
├── ai_results/                 # Per-PDB AI annotation runs (10 per PDB)
│   ├── 8TII/
│   │   ├── run_1.json
│   │   ├── run_2.json
│   │   └── ...                 # Up to run_10.json
│   └── 8Y72/
│       └── ...
├── aggregated/                 # Voted/validated annotations (curation input)
│   ├── 8TII.json
│   ├── logs/                   # Multi-run voting discrepancy logs
│   │   └── 8TII_voting_log.json
│   └── validation_logs/        # Algorithmic validation reports
│       └── 8TII_validation.json
├── output/
│   ├── csv/                    # Curated database-ready CSVs
│   └── audit/                  # Decision provenance (audit_trail.jsonl)
├── cache/                      # Persistent API caches (UniProt, PDB-CCD)
├── state/                      # Machine-owned operational state
│   ├── processed_log.json      # Tracks completed/skipped PDBs
│   ├── aggregate_log.json      # Tracks aggregation status per PDB
│   └── pipeline_runs/
└── tmp/                        # Ephemeral scratch space
```

### Data Requirements

The **aggregation** step reads from `ai_results/` (10 AI runs per PDB) and `enriched/` (PDB metadata), and writes to `aggregated/`.

The **curation** step reads from `aggregated/` and writes to `output/csv/` and `output/audit/`.

## Output Artifacts

The tool exports curated data into two main categories: strict relational CSVs for database ingestion, and provenance logs for quality assurance.

### Database CSVs (`output/csv/`)

Generated as tab-separated files:

| File | Contents |
| --- | --- |
| `structures.csv` | PDB ID, receptor, method, resolution, state, chain, date |
| `ligands.csv` | Ligand names, PubChem IDs, roles, SMILES, InChIKey, sequences |
| `g_proteins.csv` | G-protein subunit UniProt IDs and chain assignments |
| `arrestins.csv` | Arrestin UniProt IDs and chains |
| `fusion_proteins.csv` | Fusion protein names |
| `nanobodies.csv` | Nanobody names |
| `antibodies.csv` | Antibody and Fab fragment names |
| `grk.csv` | GRK names |
| `ramp.csv` | RAMP/MRAP names |
| `scfv.csv` | scFv names |
| `other_aux_proteins.csv` | Other auxiliary protein names |

### Validation Reports (`aggregated/validation_logs/`)

Each PDB receives a structured validation report containing:
* **Critical warnings** — hallucinated ligands, chimeric fusion proteins, identity clashes
* **Algorithmic conflicts** — disagreements between AI annotation and API ground truth
* **Oligomer analysis** — classification, 7TM completeness, chain corrections, alerts

### Provenance & Audit Logs

* **`output/audit/audit_trail.jsonl`:** A meticulous, append-only log of every decision made by the human expert.
* **`aggregated/logs/*_voting_log.json`:** Per-field majority-vote breakdown showing where AI runs disagreed.
* **`state/processed_log.json`:** Tracks curation status (completed, skipped) to enable resumable sessions.
* **`state/aggregate_log.json`:** Tracks aggregation status per PDB to avoid redundant processing.

## Non-Interactive Mode

For CI pipelines and automated verification, curation can run without interactive prompts:

```bash
gpcr-tools curate --auto-accept
```

This processes all pending PDBs with deterministic accept-all behavior and writes the same output artifacts as the interactive path.

## System Architecture

The suite is engineered for modularity and safety, built upon modern Python packaging standards (PEP 621):

```text
gpcr_tools/
├── aggregator/          # Multi-run aggregation engine
│   ├── ai_results_loader.py    # Load & score 10 AI runs per PDB
│   ├── enriched_loader.py      # Load PDB enriched metadata
│   ├── voting.py               # Majority-vote consensus & controversy detection
│   ├── ground_truth.py         # Inject PDB/UniProt ground truth fields
│   └── runner.py               # 12-step orchestration with error isolation
├── validator/           # Algorithmic validation chain
│   ├── chimera.py              # Fusion protein / chimera detection
│   ├── receptor_validator.py   # Receptor identity cross-check (UniProt API)
│   ├── ligand_validator.py     # Ligand existence check (PDB-CCD)
│   ├── oligomer.py             # Oligomer analysis, 7TM scan, chain override
│   ├── integrity_checker.py    # Structural consistency validation
│   ├── api_clients.py          # UniProt / PDB API wrappers
│   └── cache.py                # Persistent JSON cache layer
├── csv_generator/       # Interactive curation dashboard
│   ├── app.py                  # Main curation loop
│   ├── review_engine.py        # Recursive review tree & controversy resolution
│   ├── ui.py                   # Rich terminal UI panels & displays
│   ├── validation_display.py   # Validation alert rendering
│   ├── logic.py                # Oligomer-aware data transformations
│   ├── csv_writer.py           # Pure data → CSV export (no UI)
│   └── audit.py                # JSONL audit trail writer
├── config.py            # All constants, paths, and magic strings
├── workspace.py         # Workspace initialization & contract validation
└── __main__.py          # CLI entry point (aggregate / curate)
```

### Design Principles

| Principle | Implementation |
|-----------|---------------|
| **Atomic writes** | `tempfile` + `os.replace` + `try/finally` cleanup — no partial output files |
| **Mutation isolation** | `deepcopy()` boundary in runner before passing data to validators |
| **None-safety** | `(data.get(key) or {}).get(child)` everywhere — never `.get(key, {})` |
| **No magic strings** | All cross-module strings as named constants in `config.py` |
| **Immutability** | `frozenset`, `tuple`, `MappingProxyType` for module-level constants |
| **Error isolation** | Each PDB wrapped in `try/except` — failures logged, never crash the batch |

## Development & CI/CD

We enforce strict engineering standards to maintain data integrity.

### Testing & Quality Assurance

```bash
# Run the test suite with coverage
pytest tests/ -v --cov=gpcr_tools --cov-report=term-missing

# Linting & Formatting (Ruff)
ruff check src/ tests/
ruff format src/ tests/

# Static Type Checking
mypy src/
```

The test suite includes:
* **Unit tests** for every aggregator, validator, and csv_generator module
* **Integration tests** for the full aggregate pipeline, error isolation, and atomic write safety
* **Real PDB fixture tests** covering 9 canonical GPCR structures (5G53, 8TII, 9AS1, 9BLW, 9EJZ, 9IQS, 9M88, 9NOR, 9O38) with 10 AI runs each

### Continuous Integration

GitHub Actions workflows automatically run on every push and Pull Request:
* **Code Quality:** Enforces ruff linting and formatting.
* **Type Safety:** Validates signatures via mypy (with `ignore_missing_imports = false`).
* **Test Matrix:** Executes pytest across Python 3.11 and 3.12.
* **Docker Smoke Tests:** Builds the image and exercises `init-workspace`, `curate --help`, and `curate --auto-accept` against a real workspace mount.
* **Automated Releases:** Builds and publishes the Docker image to GHCR upon semantic version tags (`v*`), gated by a smoke-test pass.

## License

This project is licensed under the Apache License 2.0.
