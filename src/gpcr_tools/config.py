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
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any

# ---------------------------------------------------------------------------
# API base URLs
# ---------------------------------------------------------------------------

RCSB_GRAPHQL_URL: str = "https://data.rcsb.org/graphql"
RCSB_SEARCH_URL: str = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_STRUCTURE_DOWNLOAD_URL: str = "https://files.rcsb.org/download"

UNIPROT_REST_URL: str = "https://rest.uniprot.org/uniprotkb"

PUBCHEM_REST_URL: str = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"

CROSSREF_API_URL: str = "https://api.crossref.org/works"
UNPAYWALL_API_URL: str = "https://api.unpaywall.org/v2"

# PMC open-access PDFs moved off the legacy FTP tree to an AWS S3 open-data
# bucket (the old oa_pdf FTP/HTTPS paths now 404, and the web reader is behind a
# bot challenge). The per-article metadata JSON at {base}/metadata/{PMCID}.1.json
# gives the authoritative object path (an ``s3://`` URI) for the PDF.
PMC_S3_BASE_URL: str = "https://pmc-oa-opendata.s3.amazonaws.com"
# NCBI ID Converter: resolve a DOI/PMID to its PMCID, so the PMC open-access
# download uses a PMCID that provably belongs to THIS paper (never one guessed
# from a CrossRef reference link or an unverified metadata field). The legacy
# /pmc/utils/idconv/v1.0/ path 301-redirects here; use the canonical URL.
NCBI_IDCONV_URL: str = "https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/"
# A downloaded "PDF" smaller than this is rejected: too small to be a real paper,
# so it is almost certainly an error stub or truncated stream, not a fed-to-model
# article.
PDF_MIN_VALID_BYTES: int = 2048

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
# PubChem's /description endpoint routinely takes 4-6s, so a tight timeout leaves
# almost no headroom and times out on any latency blip (or under a request burst).
# 15s matches the RCSB GraphQL validation timeout; the synonyms timeout is already 60s.
TIMEOUT_PUBCHEM_VALIDATION: int = 15

TIMEOUT_CROSSREF: int = 15
TIMEOUT_UNPAYWALL: int = 15
TIMEOUT_NCBI_PMC_OA: int = 20
TIMEOUT_PDF_DOWNLOAD: int = 60
TIMEOUT_BATCH_RESULT_DOWNLOAD: int = 60
TIMEOUT_RCSB_STRUCTURE: int = 60  # coordinate files are larger than metadata responses

# ---------------------------------------------------------------------------
# Rate-limit sleep durations (seconds)
# ---------------------------------------------------------------------------

SLEEP_NCBI_RATE_LIMIT: float = 0.4
SLEEP_RCSB_POST_REQUEST: float = 1.0
SLEEP_VALIDATION_RETRY: float = 1.0
# Exponential backoff between validation retries: the nth retry waits
# SLEEP_VALIDATION_RETRY * VALIDATION_RETRY_BACKOFF_FACTOR**attempt (1s, 3s, ...),
# so a transient rate-limit/congestion spike gets a widening gap to recover rather
# than three rapid-fire retries that all hit the same throttle.
VALIDATION_RETRY_BACKOFF_FACTOR: int = 3
SLEEP_GEMINI_429: float = 5.0

# ---------------------------------------------------------------------------
# PDF download / compression
# ---------------------------------------------------------------------------

PDF_DOWNLOAD_CHUNK_SIZE: int = 8192
PDF_COMPRESSION_THRESHOLD_BYTES: int = 19 * 1024 * 1024

# ---------------------------------------------------------------------------
# Gemini / annotation configuration
# ---------------------------------------------------------------------------

GEMINI_MODEL_NAME_DEFAULT: str = "gemini-3-flash-preview"


def get_gemini_model_name() -> str:
    """Resolve the Gemini model name from environment or default (lazy)."""
    return os.environ.get("GPCR_GEMINI_MODEL") or GEMINI_MODEL_NAME_DEFAULT


def model_run_subdir(model_name: str | None) -> str:
    """Filesystem-safe per-model subdirectory name for AI run outputs.

    Runs are namespaced by model (``ai_results/<pdb>/<model>/run_N.json``) so
    annotating the same structure with different models no longer overwrites.
    """
    return (model_name or "default").replace("/", "_")


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
# Files uploaded to the Gemini Files API expire (~48h). A cached upload URI in
# the batch registry older than this must be treated as gone and re-uploaded,
# so a re-submission after a long-failed batch doesn't embed a dead fileUri.
GEMINI_FILE_TTL_HOURS: int = 47

# Cap on the number of generation requests packed into one submitted batch job.
# The full corpus (thousands of structures x GEMINI_DEFAULT_RUNS runs) is far
# more than one job should carry: an oversized submission risks being rejected
# wholesale with no per-chunk retry, and one huge job is the most likely to sit
# in the queue past the provider's 48-hour expiry. Outstanding requests are
# split into jobs of at most this many requests, never splitting a single PDB's
# runs across jobs (so 200 ~= 20 PDBs at the default 10 runs each). Tunable.
GEMINI_BATCH_MAX_REQUESTS: int = 200

# Batch jobs are tracked in a registry (state/batch_jobs.json) keyed by job
# name, so a sharded submission's multiple jobs are all tracked and recovered
# (the legacy single-file pointer could hold only one). Bumped if the shape
# of a registry entry changes.
BATCH_REGISTRY_VERSION: int = 1
BATCH_STATUS_SUBMITTED: str = "submitted"
BATCH_STATUS_DOWNLOADED: str = "downloaded"
BATCH_STATUS_RECOVERED: str = "recovered"
BATCH_STATUS_FAILED: str = "failed"

# Kill switch: delete a job's uploaded inputs (per-PDB PDFs + the JSONL source)
# from the Files API once the job reaches a terminal state, and sweep orphaned
# uploads older than the TTL at submit start. The 20 GB Files-API cap fills fast
# at corpus scale without this. Set False to restore the previous never-delete
# behaviour (uploads accumulate until they expire on their own).
CLOUD_CLEANUP: bool = True

# Kill switch: dedup the paper upload (upload each unique paper once, reference
# its fileUri from every same-paper request) and pack same-paper PDBs adjacently
# into batch jobs. Set False to upload one PDF per PDB (the previous behaviour).
UPLOAD_DEDUP: bool = True

# Same-paper batch packing cap (Tier 1 enqueued-token budget): a single in-flight
# job carries at most this many requests, ~18 PDBs x GEMINI_DEFAULT_RUNS runs. A
# paper with more PDBs than fit spans consecutive jobs; its shared upload is
# ref-counted (deleted only when the last referencing job is terminal). Distinct
# from GEMINI_BATCH_MAX_REQUESTS (the hard per-job ceiling) — this is the packing
# target used when UPLOAD_DEDUP groups same-paper PDBs.
GEMINI_BATCH_PACK_REQUESTS: int = 180

# Kill switch: store one canonical paper PDF per DOI on disk, named by the
# sanitized DOI (``papers/{sanitized_doi}.pdf``); a no-DOI PDB keeps
# ``papers/{pdb}.pdf``. Set False to keep the per-PDB ``{pdb}.pdf`` layout (the
# previous behaviour, with the watcher physically replicating to each sibling).
DOI_FILENAME_STORAGE: bool = True


# Characters kept verbatim in a sanitized DOI filename; everything else (notably
# the DOI's ``/``, and any other filesystem-unsafe char) collapses to ``_``.
_DOI_SAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")
_DOI_UNDERSCORE_RUNS = re.compile(r"_+")


def sanitize_doi(doi: str) -> str:
    """Deterministic, reversible-enough filesystem-safe token for a DOI.

    Rule (stable — the single source of truth shared by the downloader, watcher,
    annotator, and reports): NFKC normalise, replace every char outside
    ``[A-Za-z0-9._-]`` (notably ``/``) with ``_``, collapse runs of ``_`` to one,
    strip leading/trailing ``_``, and lowercase. Two byte-variants of the same DOI
    therefore map to ONE canonical filename. Returns ``""`` for an empty DOI.
    """
    text = unicodedata.normalize("NFKC", str(doi or "")).strip().lower()
    text = _DOI_SAFE_CHARS.sub("_", text)
    text = _DOI_UNDERSCORE_RUNS.sub("_", text)
    return text.strip("_")


# ---------------------------------------------------------------------------
# Watcher polling configuration
# ---------------------------------------------------------------------------

WATCHER_POLL_INTERVAL: float = 2.0
WATCHER_STABILITY_CHECKS: int = 2
WATCHER_STABILITY_INTERVAL: float = 1.0

# ---------------------------------------------------------------------------
# Workspace contract
# ---------------------------------------------------------------------------

# Bumped to 2 when the pre-annotation detect stage added the `detect/` directory
# to the workspace layout. Workspaces created under version 1 must be re-created
# with `init-workspace` (or have a `detect/` directory added and the contract
# version updated).
SUPPORTED_CONTRACT_VERSION: int = 2

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
    detect_dir: Path
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
    batch_jobs_registry_file: Path
    uploaded_files_registry_file: Path
    default_prompt_file: Path


# Mapping from subdirectory name → env-var override
OVERRIDE_VARS: MappingProxyType[str, str] = MappingProxyType(
    {
        "raw": "GPCR_RAW_PATH",
        "enriched": "GPCR_ENRICHED_PATH",
        "papers": "GPCR_PAPERS_PATH",
        "ai_results": "GPCR_AI_RESULTS_PATH",
        "detect": "GPCR_DETECT_PATH",
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
    detect_dir = _resolve(workspace, "GPCR_DETECT_PATH", "detect")
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
        detect_dir=detect_dir,
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
        batch_jobs_registry_file=state_dir / "batch_jobs.json",
        uploaded_files_registry_file=state_dir / "uploaded_files_registry.json",
        default_prompt_file=workspace / "prompts" / "v5.md",
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
        # Free-text justification prose. These hold per-run explanatory writing
        # (why a binding site was called, what evidence was cited), not a decision
        # value. Each run phrases them differently, so voting on them produces
        # only churn and the discrepancy walker should not recurse into them.
        "site_ref_justification",
        "evidence",
        # Provenance label (paper vs PDB metadata): explanatory, never an
        # ingested decision value — excluded from cross-run voting like its
        # sibling evidence fields above so it produces no vote churn.
        "source",
        # Per-run provenance block (model / prompt / run metadata): an internal
        # record stamped at write time, never a voted value.
        "_provenance",
    }
)

# Scalar majority votes whose top two candidates are within this many votes of
# each other are flagged for human review: a near-tie is too fragile to present
# as settled, even when the selected value equals the majority.
VOTE_NEAR_TIE_MARGIN: int = 1

# Leaf fields whose votes differ only by letter case ("Nanobody" vs "nanobody")
# should not be treated as competing candidates: the runs agree on the entity
# and disagree only on capitalisation. For these fields the voting and
# discrepancy stages compare values case-insensitively (plain casefold) so a
# pure-case split is not surfaced as a near-tie, and a selected value that
# differs from the majority only by case is not surfaced as a discrepancy.
# Membership is deliberately narrow — ONLY a free-text display name belongs
# here. `chain_id` is excluded on purpose: chains "A" and "a" are distinct
# chains in some entries, so folding their case would silently merge them.
# Other identifier fields (type.value, role.value, uniprot_entry_name,
# chem_comp_id) are likewise left case-sensitive.
CASE_FOLD_NAME_FIELDS: frozenset[str] = frozenset({"name"})

# Self-reported confidence levels that should be promoted to human review even
# when all runs agree — a unanimous low-confidence inference is still a guess.
LOW_CONFIDENCE_LEVELS: frozenset[str] = frozenset({"Low"})

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
        # Auxiliary proteins have no stable component id, so their grouping key is
        # the free-text name. The name alone over-merges: a model reuses one label
        # (e.g. "BRIL") for several physically distinct chains, so the identity is
        # built from the normalized name PLUS a chain-set suffix — see
        # ``list_item_identity``.
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
LIGAND_TYPE_LIPID: str = "lipid"


# ---------------------------------------------------------------------------
# List-item grouping identity
# ---------------------------------------------------------------------------
# Shared by vote aggregation and curator review so both address the same list
# item by the same path. Keeping it here (not in either consumer) prevents the
# two from drifting — if they disagree, keyless items (protein/Apo ligands with
# chem_comp_id="None") get controversies stored under one path and looked up
# under another, silently hiding them from human review.


def is_empty_key(value: Any) -> bool:
    """True if *value* cannot serve as a grouping key.

    Guards against the placeholder strings the schema injects for keyless
    items (protein / Apo ligands get ``chem_comp_id="None"``) and blanks — see
    ``EMPTY_VALUES``.  Without this, ``"None"`` is truthy and every protein
    ligand would collapse into a single bogus ``"None"`` group.
    """
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in EMPTY_VALUES
    return not value


def ensure_alert_prefix(alert_type: str, message: str | None) -> str:
    """Return *message* guaranteed to carry its ``[TYPE]`` prefix exactly once.

    Oligomer alert messages built by the current validator already start with
    ``"[TYPE] at '...': ..."``, but older recorded data stored the bare
    description with no prefix.  Renderers must therefore be idempotent: prepend
    ``[TYPE]`` only when it is absent, so the prefix is never duplicated
    (e.g. ``"[MISSED_PROTOMER] [MISSED_PROTOMER] at ..."``) and never dropped.
    """
    text = (message or "").strip()
    prefix = f"[{alert_type}]"
    if not text:
        return prefix
    if text.startswith(prefix):
        return text
    return f"{prefix} {text}"


# Greek letters models interchange with their spelled-out names. Folding the
# bare letter to its word lets "Gα" share one grouping identity with the
# adjacent spelled form "Galpha". Note the substitution happens BEFORE separator
# collapse, so a separated spelling like "G-alpha" (which collapses to "g alpha")
# is NOT unified with "Gα"/"Galpha" — that residual is an accepted under-merge
# (separate vote groups), never an over-merge.
_GREEK_LETTER_NAMES: MappingProxyType[str, str] = MappingProxyType(
    {"α": "alpha", "β": "beta", "γ": "gamma"}
)

# One trailing parenthetical, e.g. "Stalk peptide (tethered agonist)" -> the
# bracket is a per-run annotation on the same entity, so it is dropped from the
# grouping key. Only a *trailing* group is stripped; a mid-string parenthetical
# (rare, and more likely load-bearing) is left in place.
_TRAILING_PAREN_RE = re.compile(r"\s*\([^()]*\)\s*$")

# Runs of separators (whitespace, hyphen, underscore, asterisk — the markdown
# emphasis the model sometimes leaks into a name) collapse to a single space so
# "anti-Fab", "anti Fab" and "anti_fab" share one key.
_NAME_SEPARATOR_RE = re.compile(r"[\s\-_*]+")


def safe_name_normalize(name: Any) -> str:
    """Deterministically normalize a free-text entity name for grouping.

    SAFE means purely rule-based — there is no fuzzy matching, edit distance,
    token-subset, embedding, or model call. The steps only erase formatting
    noise that the same entity is written with from run to run:

    1. Unicode NFKC normalize, then casefold (case-insensitive).
    2. Map Greek alpha/beta/gamma to their spelled-out word forms.
    3. Strip ONE trailing parenthetical ``(...)``.
    4. Collapse runs of whitespace / hyphen / underscore / asterisk to a
       single space, then trim.

    Trailing digits are LOAD-BEARING and never stripped: bare "mGlu2" and
    "mGlu7" are different receptors and must key apart, as must "Compound 28"
    vs "Compound 29".

    Two intentional consequences of step 3 to keep in mind:

    * A *trailing* parenthetical is folded even when it reads load-bearing
      (e.g. "peptide (agonist)" and "peptide (antagonist)" both collapse to
      "peptide"). This is deliberate: for these entities the pharmacological
      distinction lives in the separate ``role`` field, so the parenthetical
      is a per-run annotation on the same entity, not a different one.
    * The trailing-digit guarantee covers bare digits only. A digit *inside* a
      trailing parenthetical is dropped with the bracket, so "Compound (28)"
      and "Compound (29)" would collapse — preserve a load-bearing index
      outside parentheses ("Compound 28") to keep it.
    """
    text = unicodedata.normalize("NFKC", str(name)).casefold()
    for greek, word in _GREEK_LETTER_NAMES.items():
        text = text.replace(greek, word)
    text = _TRAILING_PAREN_RE.sub("", text)
    text = _NAME_SEPARATOR_RE.sub(" ", text)
    return text.strip()


def _chain_bucket(item: dict[str, Any]) -> str:
    """Sorted, de-duplicated chain-id set for an item, as a stable suffix token.

    The chain string a model emits varies in order and spacing ("H, L" / "L,H"
    / "H,L"), so it is split on commas/whitespace, lowercased, de-duplicated and
    sorted. Returns e.g. ``"h,l"`` or ``"a"``; empty string when no chain is
    recorded.
    """
    raw = item.get("chain_id")
    if is_empty_key(raw):
        return ""
    parts = {p for p in re.split(r"[,\s]+", str(raw).casefold()) if p}
    return ",".join(sorted(parts))


def list_item_identity(item: dict[str, Any], key_field: str, idx: int) -> str:
    """Stable grouping identity for a list item — the key field when usable,
    else a namespaced fallback (name -> type -> index).

    Used everywhere a list item needs a path-addressable identity (vote
    grouping, discrepancy detection, and curator-review navigation) so the
    same item resolves to the same string in every stage.

    Identity by list type:

    * Ligands (``key_field == "chem_comp_id"``) are keyed on the unchanged
      component id (plus ``site_ref`` when present), so the same compound
      modelled at two distinct sites stays two entries and never over-merges.
    * Keyless ligands (no component id — protein / peptide / Apo entries) fall
      back to a SAFE-normalized name so spelling/formatting variants of one
      entity ("Stalk peptide" / "Stalk peptide (tethered agonist)") collapse
      into a single vote group.
    * Auxiliary proteins (``key_field == "name"``) are keyed on the
      SAFE-normalized name PLUS a chain-set suffix. Two entries merge only when
      they share both a normalized name and a chain set; entries on disjoint
      chains never merge (so a reused label like "BRIL" splits into the distinct
      fusion / Fab / nanobody entities the model put on different chains).

    Chain-set membership is part of the key, so a run that records a chain
    ("BRIL" on chain A -> ``bril|ch:a``) and a run that omits it (``bril``)
    produce different identities and are voted separately. That is an accepted
    under-merge (never an over-merge); two entries that BOTH omit a chain and
    share a normalized name do merge, as intended.
    """
    group_key = item.get(key_field)

    # Auxiliary proteins: the name is the group key, but the name alone
    # over-merges distinct chains. Normalize the name and append a chain-set
    # suffix so disjoint-chain entities never collapse together.
    if key_field == "name":
        normalized = safe_name_normalize(group_key) if not is_empty_key(group_key) else f"idx{idx}"
        chains = _chain_bucket(item)
        return f"{normalized}|ch:{chains}" if chains else normalized

    if not is_empty_key(group_key):
        # A ligand modelled at two distinct sites is emitted as two entries with
        # the same component id but different site_ref; without the site in the
        # identity the two would collapse into one during voting. Only ligands
        # carry site_ref, so other list types are unaffected. (A single-site
        # ligand keys as "comp:site"; if some runs in a batch instead emit
        # 'unknown', those key as "comp" and surface as a real cross-run
        # disagreement -- which is correct to flag, not hide.)
        site_ref = item.get("site_ref")
        if site_ref and not is_empty_key(site_ref) and str(site_ref).lower() != SITE_REF_UNKNOWN:
            return f"{group_key}:{site_ref}"
        return str(group_key)
    # Keyless ligand (no component id): SAFE-normalize the fallback name so
    # spelling/formatting variants of one keyless entity share a group.
    fallback_raw = item.get("name") or item.get("type")
    fallback_id = (
        safe_name_normalize(fallback_raw) if not is_empty_key(fallback_raw) else f"idx{idx}"
    )
    return f"__keyless__:{fallback_id}"


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
VALIDATION_RECEPTOR_NO_API_DATA: str = "RECEPTOR_NO_API_DATA"
# A reported chain carries a UniProt accession in the source data but no resolved
# GPCRdb slug -- an upstream mapping gap to recover, distinct from a chain the
# source has no UniProt reference for at all (RECEPTOR_NO_API_DATA). The receptor
# identity cannot be confirmed here, so this state gates one-click accept.
VALIDATION_RECEPTOR_RCSB_UNMAPPED: str = "RECEPTOR_RCSB_UNMAPPED"

# ---------------------------------------------------------------------------
# Ligand exclude list (common buffers, ions, artifacts, detergents, matrix lipids)
# ---------------------------------------------------------------------------

# Codes here are stripped from the metadata the model sees, so a code is only
# safe to add if it is NEVER a functional GPCR ligand. The list keys on the PDB
# three-letter chem_comp id (what the enriched data carries), so a chemical's
# human name (e.g. "DMSO", "HEPES") never matches and must be given as its code
# (DMS, EPE). Any molecule that can EVER be an agonist — notably free fatty
# acids and other biological lipids — belongs in INCIDENTAL_CANDIDATES, never
# here.
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
        "CIT",
        "ACE",
        "FMT",
        "DMS",  # dimethyl sulfoxide cosolvent (the PDB code; the name "DMSO" never matched)
        "EPE",  # HEPES buffer (the PDB code; the name "HEPES" never matched)
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
        # Lipidic-cubic-phase host / matrix lipids (monoacylglycerols). Synthetic
        # crystallization matrix, never a functional ligand.
        "OLC",  # monoolein (glyceryl monooleate)
        "OLB",  # monoolein stereoisomer
        # Cholesterol hemisuccinate: a solubilization additive (cholesterol
        # surrogate). Distinct from free cholesterol (CLR), which is incidental.
        "Y01",
        # Non-ionic detergents (alkyl glucosides / thioglucosides, maltosides,
        # HEGA, amine oxide). Solubilization agents, not receptor ligands.
        "BOG",  # octyl beta-D-glucoside
        "BNG",  # nonyl beta-D-glucoside
        "SOG",  # octyl 1-thio-beta-D-glucoside
        "HTG",  # heptyl 1-thio-beta-D-glucoside
        "2CV",  # HEGA-10
        "LMN",  # lauryl maltose neopentyl glycol
        "AV0",  # lauryl maltose neopentyl glycol (alternate code)
        "LMT",  # dodecyl beta-D-maltoside
        "LDA",  # lauryl dimethylamine-N-oxide
        # Polyethylene-glycol oligomers (cryoprotectant / precipitant family,
        # alongside the PEG/PGE/PG4 already listed).
        "1PE",  # pentaethylene glycol
        "12P",  # dodecaethylene glycol
        "P6G",  # hexaethylene glycol
        # Other crystallization buffers / cryo-additives.
        "HTO",  # heptane-1,2,3-triol
        "D10",  # decane
        "TAR",  # D-tartaric acid
        "TLA",  # L-tartaric acid
    }
)

# Incidental-candidate molecules: present in many structures as EITHER a functional ligand OR
# an incidental / structural lipid. The incidental-candidate prompt fork presents
# them to the model (any member on LIGAND_EXCLUDE_LIST, e.g. PLM, is un-stripped
# from the simplified metadata so the model can see it) and guides it to judge
# the role, recording a dedicated pharmacological_role_check. ~CLR 22% / PLM 5% of corpus.
#
# The additions below are biological lipids / metabolites that flood structures as
# membrane or matrix components yet are the endogenous agonist at their cognate
# receptors (free-fatty-acid, sphingosine-1-phosphate, lysophosphatidic-acid and
# succinate receptors). They must reach the model for a role judgement, never be
# hard-excluded, because copy count alone cannot tell agonist from membrane filler.
INCIDENTAL_CANDIDATES: frozenset[str] = frozenset(
    {
        "CLR",  # cholesterol
        "PLM",  # palmitic acid
        "OLA",  # oleic acid (free fatty-acid agonist at FFA receptors)
        "S1P",  # sphingosine-1-phosphate (S1P-receptor agonist)
        "HXA",  # docosahexaenoic acid / DHA (FFA-receptor agonist)
        "NKP",  # lysophosphatidic acid (LPA-receptor agonist)
        "ACT",  # acetate (short-chain fatty-acid agonist at FFA receptors)
        "SIN",  # succinic acid (succinate-receptor agonist)
    }
)

# Genuine-lipid chem_comp ids for deterministic ``lipid`` typing. The CCD
# ``_chem_comp.type`` is "non-polymer" for almost every ligand and so cannot
# separate a lipid from any other small molecule; this frozen whitelist supplies
# that distinction. It was built offline by taking the nonpolymer comp_ids that
# occur bound in GPCR structures, keeping only members of an established lipid
# class — sterols/steroids, free fatty acids (saturated, unsaturated, hydroxy),
# fatty-acyl amides and N-acylethanolamines, sphingolipids, glycerophospholipids,
# lysophospholipids and lysophosphatidic acids — verified by chemical name and
# formula against ChEBI (is-a lipid, CHEBI:18059) and the RCSB CCD, plus a few
# canonical lipid codes (e.g. STE, MYR, PCW) that future depositions may carry.
# Deliberately excluded as a separate, model-overridable boundary: retinoids and
# other isoprenoids/prenols, short- and medium-chain acids (acetate, succinate,
# pentanoic/octanoic acid), prostaglandins/leukotrienes and other eicosanoid
# drug analogues, and synthetic lipid-receptor-agonist scaffolds. The whitelist
# is frozen in code and refreshed offline, never queried at inference time.
# Membrane-matrix and solubilization lipids (monoolein OLC/OLB, cholesterol
# hemisuccinate Y01) are intentionally absent because LIGAND_EXCLUDE_LIST strips
# them before classification.
LIPID_COMP_IDS: frozenset[str] = frozenset(
    {
        # Sterols / steroids
        "CLR",  # cholesterol
        "CO1",  # oxidized cholesterol (oxysterol)
        # Free fatty acids
        "PLM",  # palmitic acid
        "OLA",  # oleic acid
        "MYR",  # myristic acid
        "STE",  # stearic acid
        "DAO",  # lauric acid
        "EIC",  # linoleic acid
        "EPA",  # eicosapentaenoic acid
        "HXA",  # docosahexaenoic acid
        "HXD",  # 3-hydroxydodecanoic acid
        "7NR",  # 9-hydroxyoctadecanoic acid
        "9HO",  # hydroxyoctadecadienoic acid
        # Fatty-acyl amides / N-acylethanolamines
        "140",  # N-palmitoylglycine
        "5YM",  # oleoylethanolamide
        "ZI5",  # N-acylethanolamide (anandamide analogue)
        # Sphingolipids
        "S1P",  # sphingosine-1-phosphate
        "SPH",  # sphingosine
        # Glycerophospholipids / lyso / phosphatidic acid
        "LPC",  # lysophosphatidylcholine
        "LSC",  # lysophosphatidylcholine
        "PFS",  # platelet-activating-factor ether phospholipid
        "PCW",  # phosphatidylcholine
        "A1LYA",  # phosphatidylcholine
        "U3D",  # phosphatidylcholine
        "L9Q",  # phosphatidylethanolamine
        "U3G",  # phosphatidylethanolamine
        "NKP",  # lysophosphatidic acid
        "TJR",  # lysophosphatidic acid
        "UBL",  # lysophosphatidic acid
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

# G-alpha alpha5 helix comparison window. The alpha5 C-terminal hook is the
# receptor-coupling determinant: 11 residues resolve the coupling family and
# separate the transducin subgroup from Gi, while longer windows reach into
# engineered mini-G scaffolds and degrade family accuracy. When the best window
# score falls below the anchor threshold the C-terminus is unlikely to be the
# alpha5 helix (fusion / tag / truncation), so a sliding scan locates it.
CHIMERA_A5_WINDOW: int = 11
CHIMERA_A5_ANCHOR_MIN_SCORE: int = 8

# Cached UniProt reference sequences expire after this many days, so a reference
# that drifts upstream is eventually refetched instead of persisting forever.
SEQUENCE_CACHE_TTL_DAYS: int = 30

# Cached RCSB polymer-feature responses (transmembrane-helix annotations used by
# the oligomer 7TM analysis) expire after this many days. A released entry's
# features are near-immutable but can be revised upstream, so the same TTL
# pattern as the sequence cache keeps a stale copy from persisting forever while
# letting the 7TM count survive a transient RCSB outage.
POLYMER_FEATURES_CACHE_TTL_DAYS: int = 30
POLYMER_FEATURES_CACHE_NAME: str = "polymer_features_cache.json"

# Top-level marker stamped on a detect output that was written while a
# sequence-based detector transiently failed to fetch a UniProt reference (a
# timeout or 5xx -- NOT a definitive 404). The detect resume skip recomputes a
# record carrying this marker (and only such records), so a G-protein identity call
# degraded by an upstream outage self-heals on a later run, while a structure
# that legitimately produced no signal carries no marker and is never re-run.
DETECT_INCOMPLETE_MARKER_KEY: str = "_detect_incomplete"

# Detect-stage locus anchors (a signal's target_ref). Shared so detectors that
# anchor to the same top-level annotation block use one string, and the curate
# validation bucketing routes them consistently.
LOCUS_LIGANDS: str = "ligands"

# ---------------------------------------------------------------------------
# Structure geometry (gemmi-based detect-stage analysis of coordinate files)
# ---------------------------------------------------------------------------

# Coordinate files (mmCIF) are cached under cache_dir/<this subdir>. A released
# PDB's coordinates are immutable, so they are fetched once and never expire.
STRUCTURE_CACHE_SUBDIR: str = "structure_files"

# Burial ("angular coverage"): the fraction of evenly spread directions around a
# ligand copy's centroid that have a protein atom within a narrow cone. A copy
# enclosed in a pocket is covered from most directions; a copy lying on the
# membrane-facing surface (a scattered structural lipid) is covered from a narrow
# cone only. This is the load-bearing separator between a functional pocket and a
# surface lipid (calibrated on a known two-pocket case vs. detergent floods).
GEOMETRY_BURIAL_SPHERE_DIRS: int = 200
GEOMETRY_BURIAL_CONE_DEG: float = 25.0
GEOMETRY_ENV_RADIUS: float = 5.0  # protein atoms within this of any ligand atom shield it
GEOMETRY_BURIAL_MIN: float = 0.80  # a knob: raise toward 0.85 for higher precision

# Pocket contacts: a receptor residue is in a copy's pocket if any of its atoms is
# within this distance of any ligand atom. A genuine pocket lines the copy with at
# least a handful of receptor residues.
GEOMETRY_CONTACT_RADIUS: float = 4.5
GEOMETRY_MIN_POCKET_RESIDUES: int = 5
GEOMETRY_NEIGHBOR_SEARCH_RADIUS: float = 6.0  # >= every find_atoms query radius
# Element-level ligand-protein interaction typing (heavy-atom, no hydrogens; a
# coarse proxy, not directional/aromatic typing). All <= GEOMETRY_NEIGHBOR_SEARCH_RADIUS.
GEOMETRY_HBOND_HEAVY_DIST: float = 3.5  # N/O/S <-> N/O/S polar contact (H-bond proxy)
GEOMETRY_HYDROPHOBIC_DIST: float = 4.0  # C <-> C apolar contact (also the outer query radius)
GEOMETRY_METAL_COORD_DIST: float = 3.0  # metal <-> coordinating N/O/S (or metal)

# Dual-role rule: the same ligand modelled in two distinct functional pockets on
# one receptor chain. The copy cap rejects detergent floods (a lipid appears
# 5-34x), and the pocket-overlap cap requires the two pockets to be genuinely
# different (orthosteric vs. allosteric), not the same site re-modelled.
GEOMETRY_DUAL_ROLE_MAX_COPIES: int = 3
GEOMETRY_DUAL_ROLE_POCKET_JACCARD_MAX: float = 0.5

# ── Membrane frame (ANVIL-style hydrophobic-slab fit; gemmi-only, no GPCRdb) ──
# Find the bilayer from coordinates alone: search candidate membrane normals and
# slab thicknesses for the slab that best encloses exposed HYDROPHOBIC residues
# while excluding hydrophilic ones. Defaults follow the published ANVIL/Mol*
# implementation; the exposure proxy + cutoffs are ours and want calibration
# against OPM on a labelled GPCR set before the values are trusted as exact.
MEMBRANE_NORMAL_CANDIDATES: int = 175  # candidate normals (golden-spiral sphere points)
MEMBRANE_THICKNESS_MIN: float = 20.0  # Å, slab total-thickness search lower bound
MEMBRANE_THICKNESS_MAX: float = 40.0  # Å, upper bound
MEMBRANE_THICKNESS_STEP: float = 1.0  # Å
MEMBRANE_OFFSET_STEP: float = 1.0  # Å, slab-centre slide step along the normal
MEMBRANE_MIN_TM_CA: int = 60  # too few residues -> no reliable frame (None)
# Surface-exposure proxy (no SASA dep): a residue Cα with fewer than this many
# other Cα within the radius is treated as surface-exposed (membrane/solvent).
MEMBRANE_EXPOSURE_CA_RADIUS: float = 10.0  # Å
MEMBRANE_EXPOSURE_MAX_NEIGHBOR_CA: int = 22
# Membrane-hydrophobic residue set (aliphatic + aromatic) for the slab objective;
# all others count as hydrophilic. Polar HIS/SER and side-chain-less GLY are
# excluded and the aromatic-girdle TYR included; exact membership is
# calibration-pending (see membrane.py docstring).
MEMBRANE_HYDROPHOBIC_RESIDUES: frozenset[str] = frozenset(
    {"ALA", "CYS", "ILE", "LEU", "MET", "PHE", "TRP", "TYR", "VAL"}
)
# A ligand centroid within this margin (Å) of the fitted bilayer band counts as
# "in the membrane band"; the signed depth is reported regardless.
MEMBRANE_BAND_MARGIN: float = 3.0
# Lipid-facing vs pocket-facing radial test: a contact whose in-plane cosine is
# within +/- this of 0 (i.e. within ~12 deg of tangential) is ambiguous and
# excluded from the fraction. Calibration-pending, like the other membrane knobs.
MEMBRANE_FACING_DEADZONE_COS: float = 0.2

# G-protein coupling protomer (detect stage, geometry). A Class C receptor is an
# obligate dimer and only ONE protomer engages the G protein; in a heterodimer
# that protomer is often NOT the agonist-binding one (GABA-B: GABBR1 binds, GABBR2
# couples). The G-alpha contacts exactly one receptor chain, so the chain with the
# most G-alpha interface residues is the coupling (active/primary) protomer.
# Measured across the corpus: the coupling protomer carries ~11-29 interface
# residues, the partner 0 -- a decisive, upstream-computable separation.
GEOMETRY_COUPLING_MIN_CONTACTS: int = 4  # a real receptor<->G-alpha interface
GEOMETRY_COUPLING_DECISIVE_RATIO: float = 0.25  # runner-up must be <= this * top, else ambiguous

# ---------------------------------------------------------------------------
# Ligand binding-site classification (site_ref)
# ---------------------------------------------------------------------------
# A ligand's binding site is computed upstream from its receptor contact
# residues: structure contacts -> UniProt positions (via RCSB alignment) ->
# GPCRdb generic numbers + segments (via the shipped table) -> a controlled
# site_ref. The generic-numbering reference is sequence-level (like the slug),
# never GPCRdb's downstream per-structure curation.
SITE_REF_DATA_FILE: str = "gpcrdb_generic_numbers.json.gz"

# Endogenous-ligand lookup: a flat set of InChIKeys / PubChem CIDs for every
# compound the Guide to PHARMACOLOGY lists as an endogenous ligand of any target.
# is_endogenous is a ligand-intrinsic property (is THIS compound endogenous), not
# tied to the receptor of a given structure. GtoPdb-derived (CC-BY-SA), built by
# scripts/build_endogenous_table.py -- upstream-independent of GPCRdb.
ENDOGENOUS_DATA_FILE: str = "gtopdb_endogenous_ligands.json.gz"

SITE_REF_ORTHOSTERIC: str = "orthosteric"
SITE_REF_ALLOSTERIC_7TM: str = "allosteric_7tm"
SITE_REF_EXTRACELLULAR_VESTIBULE: str = "extracellular_vestibule"
SITE_REF_INTRACELLULAR: str = "intracellular"
SITE_REF_EXTRACELLULAR_DOMAIN: str = "extracellular_domain"
SITE_REF_MEMBRANE_FACING: str = (
    "membrane_facing"  # lipid-exposed outer wall of the 7TM bundle (bulk-bilayer surface)
)
SITE_REF_UNKNOWN: str = "unknown"
SITE_REF_VALUES: tuple[str, ...] = (
    SITE_REF_ORTHOSTERIC,
    SITE_REF_ALLOSTERIC_7TM,
    SITE_REF_EXTRACELLULAR_VESTIBULE,
    SITE_REF_INTRACELLULAR,
    SITE_REF_EXTRACELLULAR_DOMAIN,
    SITE_REF_MEMBRANE_FACING,
    SITE_REF_UNKNOWN,
)

# Validated minimal Class A orthosteric-pocket core (GPCRdb structure-based "x"
# numbers), from Moitra et al. 2012. Fed to the model only as a FACT -- how many
# Class A core positions a ligand contacts -- never as a site verdict; 0 for a
# non-Class-A pocket (taste/Class C/B) is honest, not a counter-signal. None of
# these sit at a TM bulge where Ballesteros-Weinstein and the x-number diverge.
ORTHOSTERIC_CORE_GENERIC: frozenset[str] = frozenset({"3x33", "5x42", "6x51", "7x39"})

# A ligand needs at least this many mapped receptor contacts to be classified
# (fewer -> the signature is too sparse, so site_ref is unknown).
SITE_REF_MIN_MAPPED_CONTACTS: int = 5

# Intracellular landmarks for orienting the membrane normal (which side is the
# cytoplasm). The membrane frame leaves the normal's sign arbitrary; projecting
# the centroid of these receptor cytoplasmic-face residues onto the normal fixes
# the sign so a ligand's signed depth gains a physical "which side" meaning. This
# uses the receptor's own 7TM backbone via the shipped generic numbering, so it
# works for apo / no-G-protein structures too (where a G-alpha reference fails).
# Canonical cytoplasmic-anchor generic numbers: the DRY arginine (3x50) and the
# NPxxY motif (7x49-7x53), both at the intracellular ends of their helices.
MEMBRANE_INTRACELLULAR_ANCHOR_GENERIC: frozenset[str] = frozenset(
    {"3x50", "7x49", "7x50", "7x51", "7x52", "7x53"}
)
# Segments that lie wholly on the cytoplasmic face of the receptor.
MEMBRANE_INTRACELLULAR_SEGMENTS: frozenset[str] = frozenset({"H8", "ICL1", "ICL2", "ICL3"})
# Too few located landmark Cα -> orientation by landmarks is unreliable; fall
# back to the G-alpha cross-check or stay unoriented (honest abstain).
MEMBRANE_MIN_ORIENT_LANDMARKS: int = 3

# How a subtype call was resolved against the alpha5 window.
CHIMERA_SUBTYPE_RESOLVED: str = "resolved"
CHIMERA_SUBTYPE_INSEPARABLE_SET: str = "inseparable_set"
CHIMERA_SUBTYPE_FAMILY_ONLY: str = "family_only"
CHIMERA_SUBTYPE_LOW_CONFIDENCE: str = "low_confidence"

# Coupling-family labels.
G_FAMILY_GS: str = "Gs"
G_FAMILY_GIO: str = "Gi/o"
G_FAMILY_GQ11: str = "Gq/11"
G_FAMILY_G1213: str = "G12/13"

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

# Coupling family of each G-alpha subtype slug.
A5_SUBTYPE_FAMILY: MappingProxyType[str, str] = MappingProxyType(
    {
        "gnas2_human": G_FAMILY_GS,
        "gnal_human": G_FAMILY_GS,
        "gnai1_human": G_FAMILY_GIO,
        "gnai2_human": G_FAMILY_GIO,
        "gnai3_human": G_FAMILY_GIO,
        "gnao_human": G_FAMILY_GIO,
        "gnaz_human": G_FAMILY_GIO,
        "gnat1_human": G_FAMILY_GIO,
        "gnat2_human": G_FAMILY_GIO,
        "gnat3_human": G_FAMILY_GIO,
        "gnaq_human": G_FAMILY_GQ11,
        "gna11_human": G_FAMILY_GQ11,
        "gna14_human": G_FAMILY_GQ11,
        "gna15_human": G_FAMILY_GQ11,
        "gna12_human": G_FAMILY_G1213,
        "gna13_human": G_FAMILY_G1213,
    }
)

# Subtype sets whose alpha5 helices are identical and cannot be told apart by
# the alpha5 window alone. The call stops at the family and is routed to review
# rather than forced to one member.
A5_INSEPARABLE_SUBTYPE_SETS: tuple[frozenset[str], ...] = (
    frozenset({"gnat1_human", "gnat2_human", "gnat3_human"}),
    frozenset({"gnai1_human", "gnai2_human"}),
    frozenset({"gnaq_human", "gna11_human"}),
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

# GPCRdb slug prefixes that mark a heterotrimeric G-protein subunit (alpha, beta,
# or gamma). A transducer-derived / G-protein-mimetic peptide whose chain carries
# one of these slugs is a signaling partner, not a receptor ligand. "gnb" covers
# G-beta-5 (curated slug gnb5_*), which "gbb" does not. ("gnat" stays for clarity
# though it is redundant under "gna" for str.startswith.)
G_PROTEIN_SUBUNIT_SLUG_PREFIXES: tuple[str, ...] = ("gna", "gnat", "gnb", "gbb", "gbg")

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
ALERT_MULTI_COPY_LIGAND: str = "MULTI_COPY_LIGAND"
ALERT_PROTOMER_IN_AUXILIARY: str = "PROTOMER_IN_AUXILIARY"
ALERT_ASSEMBLY_MISMATCH: str = "ASSEMBLY_MISMATCH"
# The roster carried no GPCR protomer -- a GPCR-annotation tool concluding "no
# GPCR" must alarm the curator rather than pass silently (the emptiness may be an
# unresolved/missing UniProt mapping, not a true absence). Promoted to a gating
# warning, unlike the other advisory oligomer alerts.
ALERT_NO_GPCR: str = "NO_GPCR"
# The transmembrane-helix annotation fetch (RCSB) failed and no cached copy was
# available, so a chain carrying a GPCR slug could not be transmembrane-verified.
# Rather than fail open (count an unverified peptide/partner as a receptor
# protomer), the chain's receptor status is treated as low-confidence and routed
# to a curator. Promoted to a gating warning.
ALERT_TM_DATA_UNAVAILABLE: str = "TM_DATA_UNAVAILABLE"

# ---------------------------------------------------------------------------
# 7TM statuses & detection constants
# ---------------------------------------------------------------------------

TM_STATUS_UNKNOWN: str = "UNKNOWN"
TM_STATUS_COMPLETE: str = "COMPLETE"
TM_STATUS_INCOMPLETE: str = "INCOMPLETE_7TM"

TM_COVERAGE_THRESHOLD: float = 0.50

# A chain counts as a 7TM GPCR protomer (for oligomer classification + the
# missed-protomer check) only if its UniProt annotation carries at least this
# many TM helices. Single-pass (1) / few-pass partners and soluble chains that
# RCSB/GPCRdb mis-mapped to a GPCR slug are excluded, so they don't inflate
# HETEROMER or trigger a false MISSED_PROTOMER (they remain in auxiliary_proteins
# and are still surfaced by the SUSPICIOUS_7TM alert).
GPCR_MIN_ANNOTATED_TM: int = 4

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
# Crystallization fusion partners (BRIL / T4-lysozyme) — engineering aids fused
# into a receptor to aid crystallization, not part of the biological receptor.
# A receptor entity carrying one is surfaced as an advisory note.
# ---------------------------------------------------------------------------

CRYSTALLIZATION_FUSION_SLUGS: tuple[str, ...] = ("c562", "enlys")
CRYSTALLIZATION_FUSION_KEYWORDS: tuple[str, ...] = (
    "bril",
    "b562",
    "cytochrome b562",
    "lysozyme",
    "endolysin",
    # Green fluorescent protein, fused into a receptor as a folding/expression
    # reporter rather than part of the receptor. Matched on the full entity name
    # (the form PDB/UniProt records) so a real protomer whose display name merely
    # contains the token "gfp" is not mistaken for a fusion.
    "green fluorescent protein",
    # Glycogen synthase, a soluble globular domain that has been used as a
    # fusion aid. Requires both words to match, so the false-keep surface is small.
    "glycogen synthase",
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
DL_STATUS_SKIPPED_NO_PAPER: str = "skipped_no_paper"

# ---------------------------------------------------------------------------
# Aggregation / curation status values
# ---------------------------------------------------------------------------

AGG_STATUS_COMPLETED: str = "completed"
AGG_STATUS_FAILED: str = "failed"
AGG_STATUS_SKIPPED: str = "skipped"

# ---------------------------------------------------------------------------
# Run-manifest output filenames
# ---------------------------------------------------------------------------
# A comprehensive per-run record a curator reads for full situational awareness:
# what was targeted, what never ran (and why), what ran incomplete, the quality
# breakdown, and the provenance. Written to the workspace output/ directory in
# both a machine (JSON) and a human (Markdown) form.

RUN_MANIFEST_JSON_NAME: str = "run_manifest.json"
RUN_MANIFEST_MD_NAME: str = "run_manifest.md"

# ---------------------------------------------------------------------------
# Alert prefix strings (used in validation reports)
# ---------------------------------------------------------------------------

ALERT_PREFIX_TIE_BREAKER_ALIGNED: str = "[TIE-BREAKER ALIGNED]"
ALERT_PREFIX_TIE_BREAKER_OVERRIDE: str = "[TIE-BREAKER OVERRIDE]"
ALERT_PREFIX_HALLUCINATION: str = "[HALLUCINATION ALERT]"
ALERT_PREFIX_ALGO_WARNING: str = "[ALGO WARNING]"
ALERT_PREFIX_API_UNAVAILABLE: str = "[API_UNAVAILABLE]"
ALERT_PREFIX_CHIMERIC_REVIEW: str = "[CHIMERIC G-PROTEIN]"
ALERT_PREFIX_MISSED_POLYMER: str = "[UNANNOTATED CHAIN]"
ALERT_PREFIX_FUSION_NOTE: str = "[CRYSTALLIZATION FUSION]"
ALERT_PREFIX_ALPHA5_GRAFT: str = "[ALPHA5 GRAFT]"
ALERT_PREFIX_UNRECOGNISED_G_ALPHA: str = "[UNRECOGNISED G-ALPHA]"
ALERT_PREFIX_G_PROTEIN_LIGAND: str = "[G-PROTEIN PEPTIDE AS LIGAND]"
ALERT_PREFIX_MULTIPLE_AGONISTS: str = "[MULTIPLE AGONISTS]"

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
            "Note",
            "Date",
            "label_asym_id",
            # APPENDED, never inserted: the downstream build reads the leading columns
            # positionally, so new columns go at the end. The other protomer(s) of a
            # dimer -- the partner gene a heterodimer would otherwise drop (GABA-B:
            # GABBR1 alongside the GABBR2 primary).
            "Partner_UniProt",
            "Partner_ChainID",
        ),
        "ligands.csv": (
            "PDB",
            "ChainID",
            "Name",
            "PubChemID",
            "Role",
            "Title",
            "Type",
            "Date",
            "In structure",
            "label_asym_id",
            "SMILES",
            "InChIKey",
            "Sequence",
            # Appended: is this bound compound an endogenous ligand (GtoPdb)?
            "is_endogenous",
            # Appended, never inserted: the downstream build reads the leading
            # columns positionally (PDB..In structure), so the binding-site type
            # goes at the end alongside the other added columns.
            "Site",
        ),
        "g_proteins.csv": (
            "PDB",
            "Alpha_UniProt",
            "Alpha_ChainID",
            "Beta_UniProt",
            "Beta_ChainID",
            "Gamma_UniProt",
            "Gamma_ChainID",
            "Note",
            "Alpha_label_asym_id",
            "Beta_label_asym_id",
            "Gamma_label_asym_id",
        ),
        "arrestins.csv": ("PDB", "UniProt", "ChainID", "Note", "label_asym_id"),
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
    # "invalid uniprot" was pruned — no warning text in the new system
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
