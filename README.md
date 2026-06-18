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
│  3. gpcr-tools detect         Pre-annotation structural detection   │
│                                (coordinate-driven evidence: G-      │
│                                 protein coupling, ligand binding-   │
│                                 site geometry, oligomeric state,    │
│                                 chimera provenance → fed to the AI) │
├─────────────────────────────────────────────────────────────────────┤
│  4. gpcr-tools annotate       AI annotation via Gemini              │
│                                (10 independent runs per PDB,        │
│                                 structured output via tool calling) │
├─────────────────────────────────────────────────────────────────────┤
│  5. gpcr-tools aggregate      Majority-vote consensus + validation  │
│                                (cross-validation against PDB/       │
│                                 UniProt/PubChem ground truth +      │
│                                 warn-only safety cross-checks)      │
├─────────────────────────────────────────────────────────────────────┤
│  6. gpcr-tools curate         Interactive expert review dashboard   │
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
- **Tiered paper acquisition** — Fetches open-access PDFs via Unpaywall and the NCBI PMC open-access S3 bucket (PMCID resolved authoritatively from the DOI via the NCBI ID Converter, so the wrong paper can never be attached), with PubMed abstract fallback. Paywalled papers are handled by a DOI-grouped manual workflow: structures sharing a DOI are processed together, and a live filesystem watcher renames each PDF you drop into `papers/` and replicates it to its sibling structures. PDFs are stored canonically as one `papers/{doi}.pdf` per paper, deduplicated by content hash.

### Structural Detection (pre-annotation)

A coordinate-driven detect stage (built on [gemmi](https://gemmi.readthedocs.io/)) runs before annotation and supplies the model with objective structural **facts** — not computed verdicts — leaving the final judgment to the AI:

- **G-protein coupling subtype** — Identifies the G-alpha subtype by matching the structure's alpha5 C-terminal window against reference sequences; subtypes that share an identical alpha5 helix (an inseparable set) are routed to family-level review instead of guessing one confident subtype.
- **Binding-site geometry** — Each ligand's contacted residues are mapped to GPCRdb generic numbers and segments, with an ANVIL-style membrane-frame fit (oriented from the receptor's own intracellular landmarks: DRY, NPxxY, H8) that reports lipid-facing vs pocket-facing fraction and signed membrane depth. The model infers `site_ref` from these facts plus the paper, with `unknown` a first-class answer.
- **Dimer coupling protomer** — For an obligate Class C dimer, the protomer the G-alpha actually engages is detected from coordinates and used to pick the dimer's primary chain (e.g. GABA-B's GABBR2 couples while GABBR1 binds the agonist); the partner protomer is recorded.
- **Incidental-candidate ligands** — Dual-use molecules (cholesterol, palmitate, etc.) are surfaced to the model for a functional-vs-structural judgment rather than being silently dropped.
- **Fail-safe and incremental** — Detect output tops up only missing or transiently degraded results and never re-runs structures that legitimately produced no signal (`--force` recomputes everything).

### AI Annotation

- **Multi-run consensus** — Each PDB is annotated 10 times independently (configurable via `--runs`), producing a statistically robust basis for majority voting.
- **Structured output via tool calling** — Gemini returns annotations in a strict JSON schema enforced by function calling, not free-form text. Every field (receptor identity, ligand roles, signaling partners, oligomeric state, state classification) is constrained to defined types and enumerations.
- **Context-rich prompts** — The AI receives not just the paper PDF but also pre-enriched PDB metadata, the detect stage's structural evidence, a per-chain polymer table carrying each chain's 7TM status and residue length (to tell a true 7TM receptor from a non-receptor partner), a chain inventory reminder, and sibling structure warnings — reducing hallucination by grounding the model in API-verified and coordinate-derived facts.
- **Model-judged oligomeric state** — The model annotates the receptor's `oligomeric_state` (monomer / homo-/hetero-dimer / etc.) from neutral facts, counting only GPCR protomers (not transducer or ligand partners).
- **Flexible model selection** — Switch models at runtime via `--model` flag or `GPCR_GEMINI_MODEL` environment variable without code changes; sampling depth is tunable via `--temperature` and `--thinking-level` (threaded through both single and batch paths).
- **Batch API support** — Large-scale annotation via Gemini Batch API with JSONL submission, polling, and automatic result recovery; submissions are sharded into jobs (never splitting a structure's runs) and tracked in a registry for idempotent recovery.
- **Rate-limited client** — Sliding-window rate limiting (1000 RPM) with exponential backoff on 429 responses.

### Post-Annotation Validation

- **7-validator chain** — Each aggregated annotation passes through a chain of cross-validation steps:
  1. **Chimera detection** — Identifies fusion constructs by comparing G-alpha C-terminal tails against UniProt reference sequences.
  2. **Receptor identity verification** — Validates UniProt entry names against the UniProt API.
  3. **Ligand existence check** — Confirms every annotated ligand exists in the PDB Chemical Component Dictionary, filtering common buffers and crystallization artifacts. Entity-based typing recognizes genuine lipids, sterols, nucleotides, and saccharides; each ligand is tagged with `is_endogenous` (from the bundled, offline IUPHAR/BPS Guide to PHARMACOLOGY set). Incidental candidates the model judged non-functional (`is_functional_ligand: false` — a structural lipid or covalent palmitoylation rather than a bound ligand) are kept out of the exported ligand table.
  4. **Oligomer analysis** — Classifies complexes (monomer / homomer / heteromer), scans 7TM domain completeness per chain, suggests the primary protomer, and auto-corrects chain-ID assignments when API evidence disagrees with AI output. A deterministic cross-check compares the model's receptor-level `oligomeric_state` only at the receptor level (so receptor+transducer complexes don't flood review); a genuine receptor-level disagreement gates one-click accept-all.
  5. **Structural integrity** — Cross-checks internal consistency of the annotation structure.
  6. **Ground truth injection** — Overwrites method, resolution, and release date with PDB-authoritative values.
  7. **Controversy detection** — Flags fields where AI runs disagreed, with per-field vote breakdowns.

#### Warn-only safety cross-checks (surface, don't rewrite)

A family of checks routes likely mistakes to the review channel that disables one-click accept-all, while leaving the model's answer untouched: role-vs-site contradictions (e.g. an allosteric role at the orthosteric site), mis-filed GPCR protomers evicted from auxiliary proteins (sparing crystallization fusions and soluble partners), co-agonist reminders when multiple agonists are present, BRIL / T4-lysozyme fusion advisories, unannotated non-GPCR polymer chains, hallucinated ligands in ligand-free structures, and unrecognised G-alpha subtypes or G-protein-derived peptides mis-filed as ligands. Assembly-vs-oligomer mismatches are informational, not alerts.

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
  -e GPCR_EMAIL_FOR_APIS="you@example.com" \
  ghcr.io/protwis/gpcr-annotation-tools detect

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
gpcr-tools detect
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
gpcr-tools fetch-papers                 # Auto-download OA, then manual workflow for paywalled
gpcr-tools fetch-papers --auto-only     # Auto-download only, skip the manual step (CI/scripting)
gpcr-tools fetch-papers --watch-only    # Skip the auto retry; go straight to the manual step
gpcr-tools fetch-papers 8TII           # Single PDB
```

After the auto phase, the papers the open-access tiers couldn't fetch are handled
**one at a time**: first, any paper already downloaded is copied to its same-DOI
sibling structures (one paper often deposits several PDBs); then, for each
remaining paper, the tool prints its DOI link and **watches `papers/` for the PDF
you drop** — the dropped file is renamed to the correct `{PDB}.pdf` automatically
(it knows which one, because it processes one paper at a time), and replicated to
that paper's other structures. Press Ctrl+C anytime to stop; resume with
`--watch-only`.

Under Docker, run it interactively (so the prompts work and the folder is shared);
open each printed DOI link in your own browser, download the PDF, and save it into
the `papers/` folder of your mounted workspace on the host:

```bash
docker run --rm -it \
  -v ~/gpcr_workspace:/workspace \
  -e GPCR_EMAIL_FOR_APIS="you@example.com" \
  ghcr.io/protwis/gpcr-annotation-tools fetch-papers
# or, to skip the auto retry of already-paywalled papers:
#   ... fetch-papers --watch-only
# then save each downloaded PDF into ~/gpcr_workspace/papers/ (any filename)
```

### `gpcr-tools detect`

Pre-annotation structural detection: compute coordinate-driven evidence (G-protein coupling, binding-site geometry, oligomeric state, chimera provenance) for the AI and flag hard cases for review.

```bash
gpcr-tools detect                       # All enriched PDBs (tops up missing/degraded)
gpcr-tools detect 8TII                  # Single PDB
gpcr-tools detect --skip-api-checks     # Skip detectors needing UniProt reference fetches
gpcr-tools detect --force               # Recompute every detect output
```

### `gpcr-tools annotate`

Run Gemini AI annotation with structured output.

```bash
gpcr-tools annotate                                    # Auto-discover pending PDBs
gpcr-tools annotate 8TII --runs 5                      # Single PDB, 5 runs
gpcr-tools annotate --model gemini-2.5-flash            # Use a different model
gpcr-tools annotate --prompt prompts/custom.md          # Custom prompt template
gpcr-tools annotate --temperature 0.7                   # Sampling temperature (default: model's own)
gpcr-tools annotate --thinking-level low                # Reasoning depth: minimal|low|medium|high
gpcr-tools annotate --batch                             # Submit via Batch API
gpcr-tools annotate --check-batch                       # Poll batch status
gpcr-tools annotate --recover                           # Re-process raw batch output
```

### `gpcr-tools aggregate`

Aggregate multi-run AI results with majority voting and cross-validation.

```bash
gpcr-tools aggregate                      # All pending PDBs
gpcr-tools aggregate 8TII                 # Single PDB
gpcr-tools aggregate --skip-api-checks    # Offline mode (no UniProt/PubChem calls)
gpcr-tools aggregate --force              # Re-process already-aggregated entries
gpcr-tools aggregate --retry-unavailable  # Re-run only PDBs that hit a transient API
                                          # abstention; reuse cached definitive lookups
```

### `gpcr-tools curate`

Interactive expert review dashboard.

```bash
gpcr-tools curate                       # Review all pending PDBs
gpcr-tools curate 8TII                  # Target a single PDB
gpcr-tools curate --auto-accept         # Non-interactive mode (CI/testing)
```

### `gpcr-tools report`

Print an operational report over pipeline outputs.

```bash
gpcr-tools report pdf-coverage          # Paper-PDF outcomes
gpcr-tools report full-audit            # Validation warnings + chimera conflicts across PDBs
gpcr-tools report tail-analysis         # G-protein chimera score distribution
gpcr-tools report run-manifest          # Per-target accounting (no-PDF / incomplete /
                                        # acceptable / gated, with provenance);
                                        # writes output/run_manifest.{json,md}
```

### `gpcr-tools pipeline`

Run `fetch → fetch-papers → detect → annotate → aggregate` in dependency order.

```bash
gpcr-tools pipeline                     # Full pipeline over all targets
gpcr-tools pipeline 8TII                # Single PDB
gpcr-tools pipeline --dry-run           # Print the planned stage sequence only
gpcr-tools pipeline --batch             # Annotate via Batch API (stops after submission)
```

### `gpcr-tools migrate-papers`

One-time, idempotent consolidation of per-PDB paper PDFs into DOI-named canonical files (safe to re-run; never deletes a source until its canonical copy exists and validates).

---

## Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GPCR_WORKSPACE` | No | Workspace root (default: `/workspace`) |
| `GPCR_GEMINI_API_KEY` | For `annotate` | Google Gemini API key |
| `GPCR_GEMINI_MODEL` | No | Model override (default: `gemini-3-flash-preview`) |
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
| `GPCR_DETECT_PATH` | `{workspace}/detect` |
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
├── prompts/v5.md                     # Default annotation prompt template (Markdown)
│
├── raw/pdb_json/                     # RCSB GraphQL responses
├── enriched/                         # Enriched PDB metadata (AI input)
├── papers/                           # Downloaded PDFs and abstracts
├── ai_results/{pdb_id}/run_*.json   # 10 independent AI annotation runs
├── detect/{pdb_id}.json             # Pre-annotation detect signals
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
| `structures.csv` | PDB ID, receptor UniProt, method, resolution, state, chain, date, and (for a heterodimer) the partner protomer's UniProt + chain |
| `ligands.csv` | Ligand names, PubChem IDs, roles, binding-site type (`Site`, from the geometry-informed `site_ref`), entity types, SMILES, InChIKey, sequences, and whether the bound compound is an endogenous ligand (`is_endogenous`, GtoPdb). Incidental molecules the model judged non-functional are omitted. |
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
- **Warn-only cross-checks** — role-vs-site contradictions, mis-filed protomers, co-agonist and fusion advisories (surface for review; never silently rewritten)

### Provenance Logs

| Log | Purpose |
|-----|---------|
| `output/audit/audit_trail.jsonl` | Every human decision, timestamped and append-only |
| `aggregated/logs/*_voting_log.json` | Per-field majority-vote breakdowns across 10 AI runs (always written); includes advisory, non-gating records for minority items the best run dropped |
| `output/run_manifest.{json,md}` | Per-target accounting (no-PDF / incomplete / acceptable / gated) with source-commit provenance, written by `report run-manifest` |
| `state/processed_log.json` | Curation completion status (enables resumable sessions) |

Each annotation's `_provenance` block records the source git commit (baked into the Docker image, or read from git locally) so every output is traceable to the code that produced it.

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
├── detector/                  # Pre-annotation detect stage (runs before annotate)
│   ├── signals.py             #   DetectSignal contract (advisory→prompt, review→curator)
│   ├── gprotein.py            #   G-protein alpha5 identity detector
│   ├── coupling.py            #   G-protein-coupling protomer of a dimer (geometry)
│   ├── site_ref.py            #   Ligand binding-site detector (geometry → generic numbers)
│   ├── geometry.py            #   Dual-role ligand detector (multi-pocket burial)
│   ├── ligands.py             #   Incidental-candidate ligand detector (cholesterol, palmitate)
│   └── stage.py               #   enriched -> signals -> detect/{pdb_id}.json
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
├── validator/                 # Cross-validation + enrichment modules
│   ├── chimera.py             #   G-protein alpha5 identity (sequence matching)
│   ├── receptor_validator.py  #   UniProt identity verification
│   ├── ligand_validator.py    #   PDB-CCD existence check + endogenous tagging
│   ├── endogenous.py          #   Endogenous-ligand classifier (GtoPdb table)
│   ├── oligomer.py            #   Complex classification + 7TM completeness
│   ├── geometry.py            #   Contact / burial geometry (gemmi)
│   ├── generic_numbering.py   #   UniProt position → GPCRdb generic number
│   ├── integrity_checker.py   #   Structural consistency validation
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

The test suite includes 1,100+ tests:

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

This project's source code is licensed under the [Apache License 2.0](LICENSE).

It bundles third-party reference data under `src/gpcr_tools/data/`, each retaining its
own license: the GPCRdb generic-numbering table (CC BY 4.0) and the IUPHAR/BPS Guide to
PHARMACOLOGY endogenous-ligand set (ODbL + CC BY-SA 4.0). See [`NOTICE`](NOTICE) for
attribution and terms.
