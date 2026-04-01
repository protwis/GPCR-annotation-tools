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
    "fake uniprot",
    "does not exist in uniprot",
    "does not exist in uniprotkb",
    "not in pdb source",
    "not found in api entities",
    "invalid uniprot",
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
