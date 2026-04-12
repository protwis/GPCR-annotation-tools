"""Centralized configuration for GPCR Annotation Tools.

Provides a lazily-computed, resettable WorkspaceConfig that resolves all
workspace paths from environment variables.  The canonical variable is
GPCR_WORKSPACE (default ``/workspace``).  Power-user overrides use
GPCR_*_PATH variables — see storage_mounting_strategy_v3.1.md §5.

Non-path constants (CSV schema, dispatch tables, review-engine settings)
are kept in this module for backward compatibility but are independent of
workspace resolution.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType

# ---------------------------------------------------------------------------
# API base URLs
# ---------------------------------------------------------------------------

RCSB_GRAPHQL_URL: str = "https://data.rcsb.org/graphql"
RCSB_SEARCH_URL: str = "https://search.rcsb.org/rcsbsearch/v2/query"

UNIPROT_REST_URL: str = "https://rest.uniprot.org/uniprotkb"

PUBCHEM_REST_URL: str = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"

CROSSREF_API_URL: str = "https://api.crossref.org/works"
UNPAYWALL_API_URL: str = "https://api.unpaywall.org/v2"

NCBI_PMC_OA_URL: str = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
NCBI_EUTILS_EFETCH_URL: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# ---------------------------------------------------------------------------
# HTTP User-Agent strings
# ---------------------------------------------------------------------------

USER_AGENT_ENRICHER: str = "GPCR_Annotation_Pipeline/1.0 (scientific_research_script)"

# ---------------------------------------------------------------------------
# HTTP retry strategy (shared by enricher & downloader sessions)
# ---------------------------------------------------------------------------

HTTP_RETRY_TOTAL: int = 5
HTTP_RETRY_READ: int = 5
HTTP_RETRY_CONNECT: int = 5
HTTP_RETRY_BACKOFF_FACTOR: int = 1
HTTP_RETRY_STATUS_FORCELIST: tuple[int, ...] = (429, 500, 502, 503, 504)
HTTP_RETRY_ALLOWED_METHODS: tuple[str, ...] = ("HEAD", "GET", "POST", "OPTIONS")

# ---------------------------------------------------------------------------
# Per-endpoint timeout values (seconds)
# ---------------------------------------------------------------------------

TIMEOUT_RCSB_GRAPHQL: int = 30
TIMEOUT_RCSB_GRAPHQL_VALIDATION: int = 15
TIMEOUT_RCSB_CHEM_COMP: int = 10
TIMEOUT_RCSB_SEARCH: int = 10

TIMEOUT_UNIPROT_BATCH: int = 30
TIMEOUT_UNIPROT_VALIDATION: int = 5
TIMEOUT_UNIPROT_FASTA: int = 10

TIMEOUT_PUBCHEM_CID: int = 20
TIMEOUT_PUBCHEM_SYNONYMS: int = 60
TIMEOUT_PUBCHEM_VALIDATION: int = 5

TIMEOUT_CROSSREF: int = 15
TIMEOUT_UNPAYWALL: int = 15
TIMEOUT_NCBI_PMC_OA: int = 20
TIMEOUT_NCBI_EUTILS: int = 20
TIMEOUT_PDF_DOWNLOAD: int = 60
TIMEOUT_BATCH_RESULT_DOWNLOAD: int = 60

# ---------------------------------------------------------------------------
# Rate-limit sleep durations (seconds)
# ---------------------------------------------------------------------------

SLEEP_NCBI_RATE_LIMIT: float = 0.4
SLEEP_RCSB_POST_REQUEST: float = 1.0
SLEEP_VALIDATION_RETRY: float = 1.0
SLEEP_GEMINI_429: float = 5.0

# ---------------------------------------------------------------------------
# Enricher thresholds
# ---------------------------------------------------------------------------

LIGAND_WEIGHT_THRESHOLD: float = 900.0

# ---------------------------------------------------------------------------
# PDF download / compression
# ---------------------------------------------------------------------------

PDF_DOWNLOAD_CHUNK_SIZE: int = 8192
PDF_COMPRESSION_THRESHOLD_BYTES: int = 19 * 1024 * 1024

# ---------------------------------------------------------------------------
# Gemini / annotation configuration
# ---------------------------------------------------------------------------

GEMINI_MODEL_NAME_DEFAULT: str = "gemini-2.5-pro"


def get_gemini_model_name() -> str:
    """Resolve the Gemini model name from environment or default (lazy)."""
    return os.environ.get("GPCR_GEMINI_MODEL") or GEMINI_MODEL_NAME_DEFAULT


# Kept for backward-compat import; prefer get_gemini_model_name() for fresh reads.
GEMINI_MODEL_NAME: str = GEMINI_MODEL_NAME_DEFAULT
GEMINI_API_KEY_ENV: str = "GPCR_GEMINI_API_KEY"
GEMINI_API_KEY_ENV_LEGACY: str = "GPCR_GEMINI_API_KEYS"
GEMINI_RPM_LIMIT: int = 1000
GEMINI_WINDOW_SECONDS: int = 60
GEMINI_MAX_RETRIES: int = 5
GEMINI_BASE_BACKOFF: int = 10
GEMINI_DEFAULT_RUNS: int = 10
GEMINI_MAX_WORKERS: int = 10

# ---------------------------------------------------------------------------
# Watcher polling configuration
# ---------------------------------------------------------------------------

WATCHER_POLL_INTERVAL: float = 2.0
WATCHER_STABILITY_CHECKS: int = 2
WATCHER_STABILITY_INTERVAL: float = 1.0

# ---------------------------------------------------------------------------
# Workspace contract
# ---------------------------------------------------------------------------

SUPPORTED_CONTRACT_VERSION: int = 1

# ---------------------------------------------------------------------------
# Workspace configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkspaceConfig:
    """Immutable snapshot of resolved workspace paths.

    Every path is guaranteed absolute after resolution.
    """

    workspace: Path

    raw_dir: Path
    enriched_dir: Path
    papers_dir: Path
    ai_results_dir: Path
    aggregated_dir: Path
    output_dir: Path
    cache_dir: Path
    state_dir: Path
    tmp_dir: Path

    raw_pdb_json_dir: Path

    contract_file: Path
    csv_output_dir: Path
    audit_output_dir: Path
    processed_log_file: Path
    pipeline_runs_dir: Path
    targets_file: Path
    download_log_file: Path
    current_batch_job_file: Path
    uploaded_files_registry_file: Path
    default_prompt_file: Path


# Mapping from subdirectory name → env-var override
OVERRIDE_VARS: MappingProxyType[str, str] = MappingProxyType(
    {
        "raw": "GPCR_RAW_PATH",
        "enriched": "GPCR_ENRICHED_PATH",
        "papers": "GPCR_PAPERS_PATH",
        "ai_results": "GPCR_AI_RESULTS_PATH",
        "aggregated": "GPCR_AGGREGATED_PATH",
        "output": "GPCR_OUTPUT_PATH",
        "cache": "GPCR_CACHE_PATH",
        "state": "GPCR_STATE_PATH",
        "tmp": "GPCR_TMP_PATH",
    }
)


def _resolve(workspace: Path, explicit_var: str, workspace_subdir: str) -> Path:
    """Resolve a workspace subdirectory, preferring an explicit override."""
    explicit = os.environ.get(explicit_var)
    return Path(explicit).resolve() if explicit else (workspace / workspace_subdir).resolve()


@lru_cache(maxsize=1)
def get_config() -> WorkspaceConfig:
    """Build and cache the workspace configuration from environment variables.

    Call :func:`reset_config` to invalidate the cache (e.g. between tests).
    """
    workspace = Path(os.environ.get("GPCR_WORKSPACE", "/workspace")).resolve()

    raw_dir = _resolve(workspace, "GPCR_RAW_PATH", "raw")
    enriched_dir = _resolve(workspace, "GPCR_ENRICHED_PATH", "enriched")
    papers_dir = _resolve(workspace, "GPCR_PAPERS_PATH", "papers")
    ai_results_dir = _resolve(workspace, "GPCR_AI_RESULTS_PATH", "ai_results")
    aggregated_dir = _resolve(workspace, "GPCR_AGGREGATED_PATH", "aggregated")
    output_dir = _resolve(workspace, "GPCR_OUTPUT_PATH", "output")
    cache_dir = _resolve(workspace, "GPCR_CACHE_PATH", "cache")
    state_dir = _resolve(workspace, "GPCR_STATE_PATH", "state")
    tmp_dir = _resolve(workspace, "GPCR_TMP_PATH", "tmp")

    return WorkspaceConfig(
        workspace=workspace,
        raw_dir=raw_dir,
        enriched_dir=enriched_dir,
        papers_dir=papers_dir,
        ai_results_dir=ai_results_dir,
        aggregated_dir=aggregated_dir,
        output_dir=output_dir,
        cache_dir=cache_dir,
        state_dir=state_dir,
        tmp_dir=tmp_dir,
        raw_pdb_json_dir=raw_dir / "pdb_json",
        contract_file=workspace / "contract" / "storage_contract.json",
        csv_output_dir=output_dir / "csv",
        audit_output_dir=output_dir / "audit",
        processed_log_file=state_dir / "processed_log.json",
        pipeline_runs_dir=state_dir / "pipeline_runs",
        targets_file=workspace / "targets.txt",
        download_log_file=state_dir / "download_log.json",
        current_batch_job_file=state_dir / "current_batch_job.txt",
        uploaded_files_registry_file=state_dir / "uploaded_files_registry.json",
        default_prompt_file=workspace / "prompts" / "v5.txt",
    )


def reset_config() -> None:
    """Clear the cached config so the next :func:`get_config` re-resolves."""
    get_config.cache_clear()


# ---------------------------------------------------------------------------
# Voting & Aggregation constants
# ---------------------------------------------------------------------------

SOFT_FIELD_KEYS: frozenset[str] = frozenset(
    {
        "note",
        "reasoning",
        "quote_or_path",
        "key_findings",
        "synonyms",
        "confidence",
    }
)

GROUND_TRUTH_PATHS: frozenset[str] = frozenset(
    {
        "structure_info.method",
        "structure_info.resolution",
        "structure_info.release_date",
    }
)

LIST_ITEM_KEY_FIELDS: MappingProxyType[str, str] = MappingProxyType(
    {
        "ligands": "chem_comp_id",
        "auxiliary_proteins": "name",
    }
)

# ---------------------------------------------------------------------------
# Sentinel values
# ---------------------------------------------------------------------------

API_MAX_RETRIES: int = 3

EMPTY_VALUES: frozenset[str] = frozenset({"none", "n/a", "null", "", "-"})

APO_SENTINEL: str = "apo"

# Ligand type classifiers
LIGAND_TYPE_PEPTIDE: str = "peptide"
LIGAND_TYPE_PROTEIN: str = "protein"

# ---------------------------------------------------------------------------
# Validation statuses (Ligand / Receptor)
# ---------------------------------------------------------------------------

VALIDATION_SKIPPED_APO: str = "SKIPPED_APO"
VALIDATION_MATCHED_POLYMER: str = "MATCHED_POLYMER"
VALIDATION_MATCHED_SMALL_MOLECULE: str = "MATCHED_SMALL_MOLECULE"
VALIDATION_EXCLUDED_BUFFER: str = "EXCLUDED_BUFFER"
VALIDATION_GHOST_LIGAND: str = "GHOST_LIGAND"
VALIDATION_RECEPTOR_MATCH: str = "RECEPTOR_MATCH"
VALIDATION_UNIPROT_CLASH: str = "UNIPROT_CLASH"

# ---------------------------------------------------------------------------
# Ligand exclude list (common buffers, ions, artifacts)
# ---------------------------------------------------------------------------

LIGAND_EXCLUDE_LIST: frozenset[str] = frozenset(
    {
        "HOH",
        "WAT",
        "DOD",
        "SO4",
        "PO4",
        "GOL",
        "EDO",
        "PEG",
        "PGE",
        "PG4",
        "BME",
        "TRS",
        "MES",
        "HEPES",
        "CIT",
        "ACE",
        "FMT",
        "DMSO",
        "NA",
        "K",
        "CL",
        "MG",
        "ZN",
        "MN",
        "FE",
        "HG",
        "CD",
        "NAD",
        "NADP",
        "FAD",
        "COA",
        "NAG",
        "MAN",
        "GAL",
        "FUC",
        "PLM",
    }
)

# ---------------------------------------------------------------------------
# Chimera statuses
# ---------------------------------------------------------------------------

CHIMERA_STATUS_SUCCESS: str = "success"
CHIMERA_STATUS_NO_G_PROTEIN: str = "no_g_protein_found"
CHIMERA_STATUS_TOO_SHORT: str = "sequence_too_short"
CHIMERA_STATUS_NO_VALID_COMPARISONS: str = "no_valid_comparisons"
CHIMERA_STATUS_SKIPPED: str = "skipped"

# ---------------------------------------------------------------------------
# Chimera domain data
# ---------------------------------------------------------------------------

CHIMERA_TAIL_LENGTH: int = 4

FULL_G_ALPHA_CANDIDATES: MappingProxyType[str, str] = MappingProxyType(
    {
        # Gs family
        "P63092": "gnas2_human",
        "P38405": "gnal_human",
        # Gi/o family
        "P63096": "gnai1_human",
        "P04899": "gnai2_human",
        "P08754": "gnai3_human",
        "P09471": "gnao_human",
        "P19086": "gnaz_human",
        "P11488": "gnat1_human",
        "P19087": "gnat2_human",
        "A8MTJ3": "gnat3_human",
        # Gq/11 family
        "P50148": "gnaq_human",
        "P29992": "gna11_human",
        "O95837": "gna14_human",
        "P30679": "gna15_human",
        # G12/13 family
        "Q03113": "gna12_human",
        "Q14344": "gna13_human",
    }
)

FAMILY_LEADERS: tuple[str, ...] = (
    "gnas2_human",
    "gnai1_human",
    "gnaq_human",
    "gna13_human",
)

G_ALPHA_EXCLUDE_KEYWORDS: tuple[str, ...] = (
    "receptor",
    "antibody",
    "nanobody",
    "fab",
    "scfv",
    "ubiquitin",
    "beta",
    "gamma",
    "gbg",
    "gbb",
    "subunit b",
    "subunit c",
    "subunit g",
)

# ---------------------------------------------------------------------------
# Oligomer classifications
# ---------------------------------------------------------------------------

OLIGOMER_NO_GPCR: str = "NO_GPCR"
OLIGOMER_MONOMER: str = "MONOMER"
OLIGOMER_HOMOMER: str = "HOMOMER"
OLIGOMER_HETEROMER: str = "HETEROMER"

# ---------------------------------------------------------------------------
# Oligomer alert types
# ---------------------------------------------------------------------------

ALERT_HALLUCINATION: str = "HALLUCINATION"
ALERT_MISSED_PROTOMER: str = "MISSED_PROTOMER"
ALERT_CONFIRMED_OLIGOMER: str = "CONFIRMED_OLIGOMER"
ALERT_CHAIN_ID_OVERRIDDEN: str = "CHAIN_ID_OVERRIDDEN"
ALERT_7TM_UPGRADE: str = "7TM_UPGRADE"
ALERT_SUSPICIOUS_7TM: str = "SUSPICIOUS_7TM"

# ---------------------------------------------------------------------------
# 7TM statuses & detection constants
# ---------------------------------------------------------------------------

TM_STATUS_UNKNOWN: str = "UNKNOWN"
TM_STATUS_COMPLETE: str = "COMPLETE"
TM_STATUS_INCOMPLETE: str = "INCOMPLETE_7TM"

TM_COVERAGE_THRESHOLD: float = 0.50

TM_ENTITY_FEATURE_TYPES: frozenset[str] = frozenset(
    {
        "TRANSMEMBRANE",
        "MEMBRANE_REGION",
        "MEMBRANE_TOPOLOGY",
        "MEMBRANE_SEGMENT",
        "MEMBRANE_DOMAIN",
        "MEMBRANE",
    }
)

TM_UNIPROT_FEATURE_TYPES: frozenset[str] = frozenset(
    {
        "TRANSMEMBRANE",
        "MEMBRANE",
        "TOPOLOGICAL_DOMAIN",
        "TRANSMEMBRANE_REGION",
        "MEMBRANE_SEGMENT",
        "MEMBRANE_DOMAIN",
    }
)

# ---------------------------------------------------------------------------
# GPCR slug negative prefixes (for is_gpcr_slug filter)
# ---------------------------------------------------------------------------

GPCR_SLUG_NEGATIVE_PREFIXES: tuple[str, ...] = (
    # G-alpha
    "gnai",
    "gnas",
    "gnaq",
    "gna1",
    "gnao",
    "gnaz",
    "gnal",
    "gnat",
    # G-protein beta/gamma
    "gbb",
    "gbg",
    # Arrestins, GRKs, RAMPs
    "arr",
    "grk",
    "ramp",
    # Glycoprotein hormones (ligands)
    "glha",
    "fshb",
    "lhb",
    "tshb",
    "cgb",
    # Non-GPCR fusion partners and other proteins
    "enlys",
    "c562",
    "fkb",
    "mamb",
    "gloc",
    "iapp",
    "gluc",
    "gon",
    "rel",
    "racd",
    "npmb",
    "rarr2",
    "a0a",
    "mtor",
)

# ---------------------------------------------------------------------------
# Download log status values (produced by papers/downloader, consumed by papers/watcher)
# ---------------------------------------------------------------------------

DL_STATUS_SUCCESS: str = "success_pdf_downloaded"
DL_STATUS_SKIPPED_EXISTS: str = "skipped_already_downloaded"
DL_STATUS_SKIPPED_NO_ENRICHED: str = "skipped_no_enriched_data"
DL_STATUS_FAILED_NO_DOI: str = "failed_no_doi"
DL_STATUS_FAILED_NO_DATA: str = "failed_no_data"
DL_STATUS_PAYWALLED: str = "fallback_paywalled"
DL_STATUS_MANUAL: str = "manual_user_provided"
DL_STATUS_ABSTRACT_ONLY: str = "fallback_abstract_only"
DL_STATUS_SKIPPED_NO_PAPER: str = "skipped_no_paper"

# ---------------------------------------------------------------------------
# Aggregation / curation status values
# ---------------------------------------------------------------------------

AGG_STATUS_COMPLETED: str = "completed"
AGG_STATUS_FAILED: str = "failed"
AGG_STATUS_SKIPPED: str = "skipped"

# ---------------------------------------------------------------------------
# Alert prefix strings (used in validation reports)
# ---------------------------------------------------------------------------

ALERT_PREFIX_TIE_BREAKER_ALIGNED: str = "[TIE-BREAKER ALIGNED]"
ALERT_PREFIX_TIE_BREAKER_OVERRIDE: str = "[TIE-BREAKER OVERRIDE]"
ALERT_PREFIX_HALLUCINATION: str = "[HALLUCINATION ALERT]"
ALERT_PREFIX_ALGO_WARNING: str = "[ALGO WARNING]"
ALERT_PREFIX_API_UNAVAILABLE: str = "[API_UNAVAILABLE]"

# ---------------------------------------------------------------------------
# Annotator function call name
# ---------------------------------------------------------------------------

ANNOTATOR_FUNCTION_NAME: str = "annotate_gpcr_db_structure"

# ---------------------------------------------------------------------------
# Non-path constants (unchanged, not part of workspace resolution)
# ---------------------------------------------------------------------------

CSV_SCHEMA: MappingProxyType[str, tuple[str, ...]] = MappingProxyType(
    {
        "structures.csv": (
            "PDB",
            "Receptor_UniProt",
            "Method",
            "Resolution",
            "State",
            "ChainID",
            "label_asym_id",
            "Note",
            "Date",
        ),
        "ligands.csv": (
            "PDB",
            "ChainID",
            "label_asym_id",
            "Name",
            "PubChemID",
            "Role",
            "Title",
            "Type",
            "Date",
            "In structure",
            "SMILES",
            "InChIKey",
            "Sequence",
        ),
        "g_proteins.csv": (
            "PDB",
            "Alpha_UniProt",
            "Alpha_ChainID",
            "Alpha_label_asym_id",
            "Beta_UniProt",
            "Beta_ChainID",
            "Beta_label_asym_id",
            "Gamma_UniProt",
            "Gamma_ChainID",
            "Gamma_label_asym_id",
            "Note",
        ),
        "arrestins.csv": ("PDB", "UniProt", "ChainID", "label_asym_id", "Note"),
        "fusion_proteins.csv": ("PDB", "Name"),
        "nanobodies.csv": ("PDB", "Name"),
        "grk.csv": ("PDB", "Name"),
        "ramp.csv": ("PDB", "Name"),
        "antibodies.csv": ("PDB", "Name"),
        "scfv.csv": ("PDB", "Name"),
        "other_aux_proteins.csv": ("PDB", "Name"),
    }
)

AUX_PROTEIN_DISPATCH: MappingProxyType[str, str] = MappingProxyType(
    {
        "Fusion protein": "fusion_proteins.csv",
        "Nanobody": "nanobodies.csv",
        "GRK": "grk.csv",
        "RAMP": "ramp.csv",
        "MRAP": "ramp.csv",
        "Antibody": "antibodies.csv",
        "Antibody fab fragment": "antibodies.csv",
        "scFv": "scfv.csv",
        "Other": "other_aux_proteins.csv",
    }
)

BLACKLISTED_KEYS: frozenset[str] = frozenset(
    {
        "evidence",
        "confidence",
        "reasoning",
        "quote_or_path",
        "synonyms",
        "validation_status",
        "UNIPROT_CLASH",
        "api_reality",
        "InChIKey",
        "SMILES",
        "SMILES_stereo",
        "Sequence",
        "api_pubchem_cid",
        "oligomer_analysis",
        "_verified_fields",
    }
)

AUTO_RESOLVE_KEYS: frozenset[str] = frozenset(
    {
        "source",
        "reasoning",
        "quote_or_path",
        "confidence",
        "synonyms",
    }
)

VALIDATION_FATAL_KEYWORDS: tuple[str, ...] = (
    "ghost chain",
    "ghost ligand",
    "ghost_ligand",
    "fake uniprot",
    "does not exist in uniprot",
    "does not exist in uniprotkb",
    "not in pdb source",
    "not found in api entities",
    # BL8 audit: "invalid uniprot" pruned — no warning text in the new system
    # produces this phrase.  The "Fake UniProt" and "does not exist" keywords
    # cover all UniProt validation failures.
    "hallucination alert",
)

TOPLEVEL_BLOCK_KEYS: tuple[str, ...] = (
    "structure_info",
    "receptor_info",
    "ligands",
    "signaling_partners",
    "auxiliary_proteins",
    "key_findings",
)
