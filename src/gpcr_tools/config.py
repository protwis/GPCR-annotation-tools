"""Centralized configuration for GPCR Annotation Tools.

All paths are configurable via environment variables so the same code works
both in local development and inside a Docker container.
"""

import os
from pathlib import Path

# ── Directory Paths ─────────────────────────────────────────────────────
# In Docker:  DATA_DIR=/data, OUTPUT_DIR=/output  (mount points)
# Locally:    override via env vars, or they default to ./data, ./output
DATA_DIR = Path(os.environ.get("GPCR_DATA_DIR", "./data"))
OUTPUT_DIR = Path(os.environ.get("GPCR_OUTPUT_DIR", "./output"))

# Sub-directories within DATA_DIR (mirrors results_aggregated/ structure)
RESULTS_DIR = DATA_DIR
LOGS_DIR = DATA_DIR / "logs"
VALIDATION_DIR = DATA_DIR / "validation_logs"

# Output files
PROCESSED_LOG_FILE = OUTPUT_DIR / "processed_log.json"
AUDIT_TRAIL_FILE = OUTPUT_DIR / "audit_trail.jsonl"

# ── CSV Schema ──────────────────────────────────────────────────────────
CSV_SCHEMA: dict[str, list[str]] = {
    "structures.csv": [
        "PDB",
        "Receptor_UniProt",
        "Method",
        "Resolution",
        "State",
        "ChainID",
        "Note",
        "Date",
    ],
    "ligands.csv": [
        "PDB",
        "ChainID",
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
        "Beta_UniProt",
        "Beta_ChainID",
        "Gamma_UniProt",
        "Gamma_ChainID",
        "Note",
    ],
    "arrestins.csv": ["PDB", "UniProt", "ChainID", "Note"],
    "fusion_proteins.csv": ["PDB", "Name"],
    "nanobodies.csv": ["PDB", "Name"],
    "grk.csv": ["PDB", "Name"],
    "ramp.csv": ["PDB", "Name"],
    "antibodies.csv": ["PDB", "Name"],
    "scfv.csv": ["PDB", "Name"],
    "other_aux_proteins.csv": ["PDB", "Name"],
}

# ── Auxiliary Protein Dispatch ──────────────────────────────────────────
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

# ── Review Engine Constants ─────────────────────────────────────────────
BLACKLISTED_KEYS: set[str] = {
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
    "heteromer_resolution",
    "_verified_fields",
    "tm_completeness",
}

AUTO_RESOLVE_KEYS: set[str] = {
    "source",
    "reasoning",
    "quote_or_path",
    "confidence",
    "synonyms",
}

VALIDATION_FATAL_KEYWORDS: tuple[str, ...] = (
    "ghost chain",
    "ghost_ligand",
    "fake uniprot",
    "does not exist in uniprot",
    "does not exist in uniprotkb",
    "not in pdb source",
    "not found in api entities",
    "invalid uniprot",
    "hallucination alert",
)

# ── Top-Level Block Processing Order ────────────────────────────────────
TOPLEVEL_BLOCK_KEYS: list[str] = [
    "structure_info",
    "receptor_info",
    "ligands",
    "signaling_partners",
    "auxiliary_proteins",
    "key_findings",
]
