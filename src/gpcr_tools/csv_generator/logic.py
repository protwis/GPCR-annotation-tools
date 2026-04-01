"""Scientific data transformations for the CSV generator.

Pure functions — no I/O, no Rich imports, no prompts. Every function takes
data in and returns data out.  Covers label_asym_id mapping, multi-chain
receptor truncation with orphaned-ligand radar, and structure note assembly.
"""

from __future__ import annotations


def map_label_asym_id(chain_id_raw: str, label_map: dict[str, str]) -> str:
    """Translate auth_asym_id → label_asym_id.  Handles comma-separated chains.

    Args:
        chain_id_raw: One or more auth_asym_ids, comma-separated (e.g. ``"A"``
            or ``"A, B"``).
        label_map: Mapping from auth_asym_id to label_asym_id.

    Returns:
        The mapped label_asym_id string.  Keys missing from *label_map* fall
        through unchanged.
    """
    if not chain_id_raw:
        return ""
    parts = [c.strip() for c in chain_id_raw.split(",")]
    mapped = [label_map.get(p, p) for p in parts]
    return ", ".join(mapped)


def collect_ligand_chains(ligands: list[dict]) -> set[str]:
    """Extract all chain IDs from ligand entries for orphaned-ligand detection.

    Filters out empty strings and common null-like sentinels
    (``"none"``, ``"null"``, ``"n/a"``).  Handles comma-separated
    ``chain_id`` values within a single ligand entry.
    """
    chains: set[str] = set()
    for lig in ligands:
        lc = lig.get("chain_id", "")
        if isinstance(lc, str) and lc.strip() and lc.lower() not in ("none", "null", "n/a"):
            for part in lc.split(","):
                part = part.strip()
                if part:
                    chains.add(part)
    return chains


def apply_db_truncation(
    receptor_chain: str,
    receptor_uniprot: str,
    oligo: dict,
    ligand_chains: set[str],
) -> tuple[str, str, str]:
    """Truncate multi-chain receptor to primary protomer for DB compliance.

    When the AI correctly identifies multiple GPCR chains (e.g. ``"A, B"``
    for a homodimer), databases typically require a single chain.  This
    function selects the primary protomer from
    ``oligo["primary_protomer_suggestion"]`` and generates a ``truncation_note``
    documenting the decision.

    Args:
        receptor_chain: Current receptor chain string (may contain commas).
        receptor_uniprot: Current receptor UniProt entry name.
        oligo: The ``oligomer_analysis`` dict from the JSON data.
        ligand_chains: Set of chain IDs where ligands are bound.

    Returns:
        ``(final_chain, final_uniprot, truncation_note)`` — if no truncation
        is needed, *truncation_note* is an empty string.
    """
    if "," not in receptor_chain or not oligo.get("primary_protomer_suggestion"):
        return receptor_chain, receptor_uniprot, ""

    suggestion = oligo["primary_protomer_suggestion"]
    primary_chain = suggestion.get("chain_id")
    if not primary_chain:
        return receptor_chain, receptor_uniprot, ""

    reason = suggestion.get("reason", "No reason provided")

    # Get accurate UniProt for the primary chain from GPCR roster.
    primary_uniprot = receptor_uniprot
    for chain_info in oligo.get("all_gpcr_chains", []):
        if chain_info.get("chain_id") == primary_chain:
            primary_uniprot = chain_info.get("slug") or receptor_uniprot
            break

    # Orphaned ligand radar — detect ligands on soon-to-be-truncated chains.
    original_chains = {c.strip() for c in receptor_chain.split(",")}
    truncated_chains = original_chains - {primary_chain}
    orphaned = ligand_chains & truncated_chains

    ligand_alert = (
        f" [WARNING: Ligands are bound to truncated chains {', '.join(sorted(orphaned))}!]"
        if orphaned
        else ""
    )

    truncation_note = (
        f"[DB TRUNCATION: AI correctly identified chains {receptor_chain}, "
        f"but DB requires 1. Auto-selected primary chain {primary_chain} "
        f"based on: {reason}]{ligand_alert}"
    )

    return primary_chain, primary_uniprot, truncation_note


def build_structure_note(
    s_info: dict,
    oligo: dict,
    truncation_note: str = "",
) -> str:
    """Build the Note field for structures.csv, appending oligomer + truncation info.

    Concatenates:
    1. The base ``s_info["note"]`` (if any)
    2. Chain-ID correction annotation (if ``chain_id_override`` was applied)
    3. Classification annotation (HOMOMER / HETEROMER with chain IDs)
    4. Structural alerts (HALLUCINATION / MISSED_PROTOMER)
    5. DB truncation note (from :func:`apply_db_truncation`)
    """
    parts: list[str] = []
    base_note = s_info.get("note", "")
    if isinstance(base_note, str):
        base_note = base_note.strip()
    else:
        base_note = str(base_note).strip() if base_note else ""
    if base_note:
        parts.append(base_note)

    if oligo:
        override = oligo.get("chain_id_override") or {}
        if override.get("applied"):
            parts.append(
                f"[CHAIN CORRECTED: {override.get('original_chain_id')} -> "
                f"{override.get('corrected_chain_id')}, reason: {override.get('trigger')}]"
            )

        classification = oligo.get("classification", "")
        if classification in ("HOMOMER", "HETEROMER"):
            chain_ids = [c.get("chain_id") or "?" for c in oligo.get("all_gpcr_chains", [])]
            parts.append(f"[{classification}: chains {', '.join(chain_ids)}]")

        for alert in oligo.get("alerts", []):
            atype = alert.get("type", "")
            if atype in ("HALLUCINATION", "MISSED_PROTOMER"):
                parts.append(f"[{atype}: {alert.get('message', '')}]")

    if truncation_note:
        parts.append(truncation_note)

    return " ".join(parts)
