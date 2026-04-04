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

    contract_file: Path
    csv_output_dir: Path
    audit_output_dir: Path
    processed_log_file: Path
    pipeline_runs_dir: Path


# Mapping from subdirectory name → env-var override
OVERRIDE_VARS: dict[str, str] = {
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
        contract_file=workspace / "contract" / "storage_contract.json",
        csv_output_dir=output_dir / "csv",
        audit_output_dir=output_dir / "audit",
        processed_log_file=state_dir / "processed_log.json",
        pipeline_runs_dir=state_dir / "pipeline_runs",
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
# Non-path constants (unchanged, not part of workspace resolution)
# ---------------------------------------------------------------------------

CSV_SCHEMA: dict[str, list[str]] = {
    "structures.csv": [
        "PDB",
        "Receptor_UniProt",
        "Method",
        "Resolution",
        "State",
        "ChainID",
        "label_asym_id",
        "Note",
        "Date",
    ],
    "ligands.csv": [
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
    ],
    "g_proteins.csv": [
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
    ],
    "arrestins.csv": ["PDB", "UniProt", "ChainID", "label_asym_id", "Note"],
    "fusion_proteins.csv": ["PDB", "Name"],
    "nanobodies.csv": ["PDB", "Name"],
    "grk.csv": ["PDB", "Name"],
    "ramp.csv": ["PDB", "Name"],
    "antibodies.csv": ["PDB", "Name"],
    "scfv.csv": ["PDB", "Name"],
    "other_aux_proteins.csv": ["PDB", "Name"],
}

AUX_PROTEIN_DISPATCH: dict[str, str] = {
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
