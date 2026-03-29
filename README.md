***

# GPCR Annotation Tools

**A production-grade, human-in-the-loop curation suite for GPCR structural biology.**

GPCR Annotation Tools provides GPCR structural biology experts with a robust, interactive command-line dashboard to review, validate, and export AI-assisted structural metadata into database-ready CSVs. Designed for high-throughput expert curation, the suite seamlessly integrates automated quality checks, algorithmic conflict resolution, and comprehensive decision provenance.

## Key Features

* **Interactive Expert Review:** A rich, ergonomic terminal UI designed for rapid curation of GPCR complex structural data, signaling partners, and ligands.
* **Integrated Validation Engine:** Real-time, context-aware alerts for structural discrepancies (e.g., ghost chains, hallucinated ligands, and UniProt identity clashes).
* **Algorithmic Resolution:** Built-in logic to surface heteromer resolutions and assess 7TM domain completeness.
* **Provable Audit Trails:** Every human intervention—whether accepting, modifying, or rejecting AI annotations—is securely logged to a JSONL audit trail.
* **Docker-Native Deployment:** Zero-configuration setup for curators via highly optimized containerized environments.

## Quick Start

### Option 1: Docker (Recommended for Curators)
The fastest way to start curating. The tool runs completely inside a container, requiring only your data directories to be mounted.

```bash
# Pull the latest production image
docker pull ghcr.io/protwis/gpcr-annotation-tools:latest

# Run the interactive review dashboard
docker run -it \
  -v /path/to/results_aggregated:/data \
  -v /path/to/output:/output \
  ghcr.io/protwis/gpcr-annotation-tools

# Target a specific PDB entry directly
docker run -it \
  -v /path/to/results_aggregated:/data \
  -v /path/to/output:/output \
  ghcr.io/protwis/gpcr-annotation-tools 8TII
```

> **Note:** The `-it` (interactive + TTY) flags are strictly required, as the application utilizes a rich interactive terminal for expert review.

### Option 2: Local Installation (For Developers)
Requires Python 3.11+.

```bash
# Clone the repository
git clone https://github.com/protwis/GPCR-annotation-tools.git
cd GPCR-annotation-tools

# Install the package and dependencies
pip install -e ".[dev]"

# Configure data paths via environment variables
export GPCR_DATA_DIR=/path/to/results_aggregated
export GPCR_OUTPUT_DIR=/path/to/output

# Launch the suite
gpcr-csv-generator

# Target a specific PDB
gpcr-csv-generator 8Y72
```

## Data Requirements

The suite acts as the final curation layer and expects pre-computed AI annotation data. Your mounted data directory (mapped to `/data` in Docker or `GPCR_DATA_DIR` locally) must follow this structure:

```text
results_aggregated/
├── 8TII.json                # Core aggregated PDB annotation data
├── 8Y72.json
├── logs/                    # (Optional) Multi-run voting discrepancies
│   └── 8TII_voting_log.json 
└── validation_logs/         # (Optional) Algorithmic validation results
    └── 8TII_validation.json 
```

## Output Artifacts

The tool exports curated data into two main categories: strict relational CSVs for database ingestion, and provenance logs for quality assurance.

### Database CSVs
Generated in the output directory, utilizing tab-separated formatting for robust data handling:

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
* **`audit_trail.jsonl`:** A meticulous, append-only log of every decision made by the human expert.
* **`processed_log.json`:** Tracks the status of all PDBs (completed, skipped) to enable resumable curation sessions.

## System Architecture

The suite is engineered for modularity and scalability, built upon modern Python packaging standards (PEP 621):

* **Configuration Layer:** Fully environment-variable-driven, ensuring absolute parity between local development and Docker deployment.
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
* **Automated Releases:** Builds and publishes the minimal-footprint Docker image to GHCR upon semantic version tags (`v*`).

## License

This project is licensed under the Apache License 2.0.