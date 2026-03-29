# GPCR Annotation Tools

**A production-grade, human-in-the-loop curation suite for GPCR structural biology.**

GPCR Annotation Tools provides GPCR structural biology experts with a robust, interactive command-line dashboard to review, validate, and export AI-assisted structural metadata into database-ready CSVs. Designed for high-throughput expert curation, the suite seamlessly integrates automated quality checks, algorithmic conflict resolution, and comprehensive decision provenance.

## Key Features

* **Interactive Expert Review:** A rich, ergonomic terminal UI designed for rapid curation of GPCR complex structural data, signaling partners, and ligands.
* **Integrated Validation Engine:** Real-time, context-aware alerts for structural discrepancies (e.g., ghost chains, hallucinated ligands, and UniProt identity clashes).
* **Algorithmic Resolution:** Built-in logic to surface heteromer resolutions and assess 7TM domain completeness.
* **Provable Audit Trails:** Every human intervention—whether accepting, modifying, or rejecting AI annotations—is securely logged to a JSONL audit trail.
* **Docker-Native Deployment:** Zero-configuration setup for curators via a single workspace mount.

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

# Run the interactive curation dashboard
docker run -it --rm \
  -v /path/to/gpcr_workspace:/workspace \
  ghcr.io/protwis/gpcr-annotation-tools

# Target a specific PDB entry directly
docker run -it --rm \
  -v /path/to/gpcr_workspace:/workspace \
  ghcr.io/protwis/gpcr-annotation-tools curate 8TII
```

> **Note:** The `-it` (interactive + TTY) flags are required for the interactive review dashboard.

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
├── enriched/                   # Normalized and enriched artifacts
├── papers/                     # Paper files and metadata
├── ai_results/                 # Per-model annotation outputs
├── aggregated/                 # Voted/merged annotations (curation input)
│   ├── logs/
│   └── validation_logs/
├── output/
│   ├── csv/                    # Curated database-ready CSVs
│   └── audit/                  # Decision provenance (audit_trail.jsonl)
├── cache/                      # Persistent API caches
├── state/                      # Machine-owned operational state
│   ├── processed_log.json      # Tracks completed/skipped PDBs
│   └── pipeline_runs/
└── tmp/                        # Ephemeral scratch space
```

### Data Requirements

The curation workflow reads from `aggregated/` inside the workspace. Place your pre-computed AI annotation data there:

```text
aggregated/
├── 8TII.json                  # Core aggregated PDB annotation data
├── 8Y72.json
├── logs/                      # (Optional) Multi-run voting discrepancies
│   └── 8TII_voting_log.json
└── validation_logs/           # (Optional) Algorithmic validation results
    └── 8TII_validation.json
```

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

### Provenance & Audit Logs

* **`output/audit/audit_trail.jsonl`:** A meticulous, append-only log of every decision made by the human expert.
* **`state/processed_log.json`:** Tracks the status of all PDBs (completed, skipped) to enable resumable curation sessions.

## Non-Interactive Mode

For CI pipelines and automated verification, curation can run without interactive prompts:

```bash
gpcr-tools curate --auto-accept
```

This processes all pending PDBs with deterministic accept-all behavior and writes the same output artifacts as the interactive path.

## System Architecture

The suite is engineered for modularity and scalability, built upon modern Python packaging standards (PEP 621):

* **Configuration Layer:** A single `WorkspaceConfig` object resolved lazily from `GPCR_WORKSPACE` and optional `GPCR_*_PATH` overrides. No import-time globals.
* **Storage Contract:** A versioned `contract/storage_contract.json` validated at every startup, ensuring deterministic workspace semantics.
* **Data Ingestion:** Safely merges core JSON metadata with auxiliary voting and validation logs.
* **Review Engine & UI:** A decoupled, recursive review tree utilizing Rich to render context-aware prompts, tabular data, and critical visual alerts.
* **Export Subsystem:** A pure, heavily-tested data transformation layer that sanitizes user-approved JSON structures into strict CSV schemas.
* **Extensible Design:** Future-proofed namespace (`gpcr_tools`) ready to absorb upstream AI pipelines, aggregation, and validation modules.

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

### Continuous Integration

GitHub Actions workflows automatically run on every push and Pull Request:
* **Code Quality:** Enforces ruff linting and formatting.
* **Type Safety:** Validates signatures via mypy.
* **Test Matrix:** Executes pytest across Python 3.11 and 3.12.
* **Docker Smoke Tests:** Builds the image and exercises `init-workspace`, `curate --help`, and `curate --auto-accept` against a real workspace mount.
* **Automated Releases:** Builds and publishes the Docker image to GHCR upon semantic version tags (`v*`), gated by a smoke-test pass.

## License

This project is licensed under the Apache License 2.0.
