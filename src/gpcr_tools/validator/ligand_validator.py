"""Ligand cross-validation and chemical identity injection.

Validates each AI-reported ligand against the enriched PDB metadata and
injects pre-fetched chemical identifiers (InChIKey, SMILES, etc.) in-place.

Reads primarily from the pre-loaded enriched entry dict.  The one network
touch point is the optional PubChem synonym gate, which runs only for a ligand
that carries a model-supplied PubChem CID with no matched chemical component,
and only when a cache is provided; without a cache the pass stays fully offline.
"""

from __future__ import annotations

import logging
from typing import Any

from gpcr_tools.config import (
    ALERT_PREFIX_API_UNAVAILABLE,
    ALERT_PREFIX_G_PROTEIN_LIGAND,
    ALERT_PREFIX_MULTIPLE_AGONISTS,
    APO_SENTINEL,
    EMPTY_VALUES,
    G_PROTEIN_SUBUNIT_SLUG_PREFIXES,
    LIGAND_EXCLUDE_LIST,
    LIGAND_TYPE_LIPID,
    LIGAND_TYPE_PEPTIDE,
    LIGAND_TYPE_PROTEIN,
    SITE_REF_ORTHOSTERIC,
    VALIDATION_EXCLUDED_BUFFER,
    VALIDATION_GHOST_LIGAND,
    VALIDATION_MATCHED_POLYMER,
    VALIDATION_MATCHED_SMALL_MOLECULE,
    VALIDATION_SKIPPED_APO,
)
from gpcr_tools.validator.api_clients import SynonymCache, check_pubchem_synonym_match
from gpcr_tools.validator.chimera import is_g_alpha_description
from gpcr_tools.validator.endogenous import ENDOGENOUS_UNKNOWN, classify_endogenous

logger = logging.getLogger(__name__)


def _build_ligand_api_context(
    enriched_entry: dict[str, Any],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Build lookup indexes from enriched entry.

    Returns ``{"np_by_comp": {...}, "poly_by_chain": {...}}``.
    All `.get()` calls use the None-safe ``or {}`` / ``or ""`` pattern.
    """
    np_by_comp: dict[str, dict[str, Any]] = {}
    for np_ent in enriched_entry.get("nonpolymer_entities") or []:
        if not isinstance(np_ent, dict):
            continue
        comp = np_ent.get("nonpolymer_comp") or {}
        cc = comp.get("chem_comp") or {}
        descriptor = comp.get("rcsb_chem_comp_descriptor") or {}
        comp_id = cc.get("id") or ""
        if not comp_id or comp_id in LIGAND_EXCLUDE_LIST:
            continue
        np_by_comp[comp_id] = {
            "name": cc.get("name"),
            "InChIKey": descriptor.get("InChIKey"),
            "SMILES": descriptor.get("SMILES"),
            "SMILES_stereo": descriptor.get("SMILES_stereo"),
            "pubchem_cid": comp.get("gpcrdb_pubchem_cid"),
        }

    poly_by_chain: dict[str, dict[str, Any]] = {}
    for p_ent in enriched_entry.get("polymer_entities") or []:
        ep = p_ent.get("entity_poly") or {}
        desc = (p_ent.get("rcsb_polymer_entity") or {}).get("pdbx_description") or ""
        seq = ep.get("pdbx_seq_one_letter_code_can") or ""
        slug: str | None = None
        for u in p_ent.get("uniprots") or []:
            if isinstance(u, dict) and u.get("gpcrdb_entry_name_slug"):
                slug = u["gpcrdb_entry_name_slug"]
                break
        for inst in p_ent.get("polymer_entity_instances") or []:
            chain = (inst.get("rcsb_polymer_entity_instance_container_identifiers") or {}).get(
                "auth_asym_id"
            )
            if chain:
                poly_by_chain[chain] = {
                    "description": desc,
                    "type": (ep.get("type") or ""),
                    "sequence": seq,
                    "slug": slug,
                }

    return {"np_by_comp": np_by_comp, "poly_by_chain": poly_by_chain}


def validate_and_enrich_ligands(
    pdb_id: str,
    best_run_data: dict[str, Any],
    enriched_entry: dict[str, Any],
    *,
    synonym_cache: SynonymCache | None = None,
) -> list[str]:
    """Validate AI-reported ligands and inject chemical identifiers.

    Mutates *best_run_data* ligand dicts in-place.
    Returns a list of warning strings (``GHOST_LIGAND`` detections, plus apo
    placeholders that coexist with real ligands).

    When *synonym_cache* is provided, a ligand that carries a model-supplied
    PubChem CID but matched no chemical component (no authoritative CID to copy)
    has that CID cross-checked against PubChem's synonym list; a CID that names a
    different molecule is blanked and flagged.  Without a cache this step is
    skipped and the pass remains fully offline.

    Warning format:
        ``f"GHOST_LIGAND at 'ligands[{label}]': '{name}' ({cid}) not found in API entities."``
    """
    warnings: list[str] = []
    ligands = best_run_data.get("ligands")
    if not isinstance(ligands, list) or not ligands:
        return warnings

    api = _build_ligand_api_context(enriched_entry)

    for lig in ligands:
        if not isinstance(lig, dict):
            continue

        # Default: only a matched small molecule can be classified endogenous;
        # peptides / ions / buffers / ghosts have no usable identifier -> unknown.
        lig["is_endogenous"] = ENDOGENOUS_UNKNOWN

        comp_id = (lig.get("chem_comp_id") or "").strip()
        chain_id = (lig.get("chain_id") or "").strip()
        ai_name = (lig.get("name") or "").strip()
        ai_type = (lig.get("type") or "").strip()

        comp_id_valid = bool(comp_id) and comp_id.lower() not in EMPTY_VALUES
        chain_id_valid = bool(chain_id) and chain_id.lower() not in EMPTY_VALUES

        # 1. Explicit Apo
        if ai_name.lower() == APO_SENTINEL or comp_id.lower() == APO_SENTINEL:
            lig["validation_status"] = VALIDATION_SKIPPED_APO
            continue

        # 2. Polymer path: peptides/proteins validated by chain_id
        if ai_type.lower() in (LIGAND_TYPE_PEPTIDE, LIGAND_TYPE_PROTEIN) and chain_id_valid:
            chains = [c.strip() for c in chain_id.split(",")]
            matched_sequences = []
            found_any_chain = False

            for c in chains:
                poly_match = api["poly_by_chain"].get(c)
                if poly_match:
                    found_any_chain = True
                    seq = poly_match.get("sequence")
                    if seq:
                        matched_sequences.append(seq)

            if found_any_chain:
                lig["validation_status"] = VALIDATION_MATCHED_POLYMER
                lig["Sequence"] = " / ".join(matched_sequences)
                continue
        # 3. Small-molecule path
        if comp_id_valid:
            np_match = api["np_by_comp"].get(comp_id)
            if np_match:
                lig["validation_status"] = VALIDATION_MATCHED_SMALL_MOLECULE
                lig["InChIKey"] = np_match.get("InChIKey")
                lig["api_pubchem_cid"] = np_match.get("pubchem_cid")
                lig["SMILES_stereo"] = np_match.get("SMILES_stereo")
                lig["SMILES"] = np_match.get("SMILES")
                lig["is_endogenous"] = classify_endogenous(lig["InChIKey"], lig["api_pubchem_cid"])
                continue

            if comp_id in LIGAND_EXCLUDE_LIST:
                lig["validation_status"] = VALIDATION_EXCLUDED_BUFFER
                continue

        # 4. Ghost fallback
        lig["validation_status"] = VALIDATION_GHOST_LIGAND
        label = comp_id if comp_id_valid else ai_name
        cid_display = comp_id or "no comp_id"
        warnings.append(
            f"GHOST_LIGAND at 'ligands[{label}]': "
            f"'{ai_name}' ({cid_display}) not found in API entities."
        )

    if synonym_cache is not None:
        _gate_keyless_pubchem_ids(ligands, synonym_cache, warnings)
    _warn_on_apo_with_real_ligands(ligands, warnings)
    _warn_on_role_site_mismatch(ligands, warnings)
    _warn_on_g_protein_peptide_as_ligand(ligands, api["poly_by_chain"], warnings)
    _warn_on_multiple_agonists(ligands, warnings)
    return warnings


def _gate_keyless_pubchem_ids(
    ligands: list[Any],
    synonym_cache: SynonymCache,
    warnings: list[str],
) -> None:
    """Cross-check a model-supplied PubChem CID against the CID's own synonyms.

    Runs only for a *keyless* ligand -- one with no ``chem_comp_id`` (a peptide,
    protein, or other entity the model named, and whose CID it therefore supplied
    from its own memory).  Any ligand carrying a ``chem_comp_id`` is left
    untouched: a matched
    small molecule keeps the authoritative CID copied from enriched data, and an
    excluded buffer (e.g. a structural lipid such as PLM) carries a CID echoed from
    that same metadata -- neither is a from-memory guess, and matched CIDs also
    carry occasional sparse-synonym entries that a synonym check would wrongly
    reject.

    Candidate names are the union of the reported name and any reported synonyms;
    matching against this union, rather than the bare name, keeps the false-
    reject rate low.  On a definitive mismatch the CID is blanked and a warning
    is appended.  On a network abstention the value is left untouched and an
    ``[API_UNAVAILABLE]`` note is emitted, mirroring the other API checks.
    """
    for lig in ligands:
        if not isinstance(lig, dict) or "api_pubchem_cid" in lig:
            continue  # Matched small molecule -> authoritative CID, leave it.
        comp_id = lig.get("chem_comp_id")
        if comp_id and str(comp_id).strip().lower() not in EMPTY_VALUES:
            # A keyed component (e.g. an excluded buffer like PLM) is identified by
            # its chem_comp_id and its CID is echoed from metadata, not guessed.
            continue
        cid = lig.get("pubchem_id")
        if not cid or str(cid).strip().lower() in EMPTY_VALUES:
            continue

        name = (lig.get("name") or "").strip()
        synonyms = lig.get("synonyms")
        candidate_names = [name] if name else []
        if isinstance(synonyms, list):
            candidate_names.extend(str(s) for s in synonyms if s)
        if not candidate_names:
            # No name or synonyms to compare against: there is nothing to verify and
            # no network call is made, so emit no warning (an [API_UNAVAILABLE] note
            # here would wrongly imply a network failure that never happened).
            continue

        verdict = check_pubchem_synonym_match(str(cid), candidate_names, synonym_cache)
        if verdict is False:
            lig["pubchem_id"] = None
            display = name or lig.get("chem_comp_id") or "?"
            warnings.append(
                f"PubChem CID Mismatch at 'ligands': CID '{cid}' is not a known "
                f"synonym of '{display}' -- the identifier appears to name a "
                f"different compound and has been cleared."
            )
        elif verdict is None:
            display = name or lig.get("chem_comp_id") or "?"
            warnings.append(
                f"{ALERT_PREFIX_API_UNAVAILABLE} at 'ligands': could not verify "
                f"PubChem CID '{cid}' for '{display}'."
            )


# Role values (subsets of the schema role enum) used by the role/site consistency
# check; the schema owns the full enum.
_ALLOSTERIC_ROLES = frozenset(
    {"PAM", "NAM", "Allosteric antagonist", "Allosteric agonist", "Ago-PAM"}
)
_FUNCTIONAL_POCKET_ROLES = frozenset(
    {"Agonist", "Antagonist", "Inverse agonist", "Agonist (partial)", "Co-agonist"}
)
# A 'lipid'-typed ligand at the orthosteric site is only a contradiction when it
# has NO pocket-justifying role: endogenous lipid agonists/modulators (S1P, LPA,
# 2-AG, prostaglandins) legitimately occupy the orthosteric pocket as type 'lipid'.
_POCKET_JUSTIFYING_ROLES = _ALLOSTERIC_ROLES | _FUNCTIONAL_POCKET_ROLES | frozenset({"Cofactor"})


def _warn_on_role_site_mismatch(ligands: list[Any], warnings: list[str]) -> None:
    """Flag (for the curator) a ligand whose pharmacological role and binding site
    contradict each other in the AI's OWN output -- a cheap second safety net that
    does not depend on the retired geometry classifier. Only unambiguous
    contradictions are flagged; with no receptor-class lookup, legitimate class C/B
    extracellular-domain agonists and intracellular agonist sites are never flagged.
    """
    for lig in ligands:
        if not isinstance(lig, dict):
            continue
        site = (lig.get("site_ref") or "").strip().lower()
        role = ((lig.get("role") or {}).get("value") or "").strip()
        lig_type = (lig.get("type") or "").strip().lower()
        name = lig.get("name") or lig.get("chem_comp_id") or "?"
        reason: str | None = None
        if (
            lig_type == LIGAND_TYPE_LIPID
            and site == SITE_REF_ORTHOSTERIC
            and role not in _POCKET_JUSTIFYING_ROLES
        ):
            reason = "type 'lipid' with no functional role at the orthosteric site"
        elif role in _ALLOSTERIC_ROLES and site == SITE_REF_ORTHOSTERIC:
            reason = f"allosteric role '{role}' at the orthosteric site"
        if reason:
            warnings.append(
                f"ROLE_SITE_MISMATCH at 'ligands': '{name}' has {reason} "
                f"-- verify the role and binding site are consistent."
            )


def _warn_on_g_protein_peptide_as_ligand(
    ligands: list[Any],
    poly_by_chain: dict[str, dict[str, Any]],
    warnings: list[str],
) -> None:
    """Flag (for the curator) a transducer-derived / G-protein-mimetic peptide that
    the model has filed as a receptor ligand with a functional pocket role.

    A peptide whose chain is a G-protein subunit (its polymer description reads as a
    G-alpha, or its GPCRdb slug is a G-protein alpha/beta/gamma subunit) is a
    signaling partner, not an agonist. This catches a G-alpha C-terminal /
    transducin-mimetic peptide mislabelled as e.g. role 'Agonist', which would
    otherwise sit next to the genuine small-molecule agonist.

    Fires only when the model committed to a functional pocket role; an honest
    'unknown' / 'Apo' / absent role is never flagged (abstaining is not an error).
    Warning-only -- the ligand is left untouched for the curator to decide.
    """
    for lig in ligands:
        if not isinstance(lig, dict):
            continue
        lig_type = (lig.get("type") or "").strip().lower()
        if lig_type not in (LIGAND_TYPE_PEPTIDE, LIGAND_TYPE_PROTEIN):
            continue
        role = ((lig.get("role") or {}).get("value") or "").strip()
        if role not in _FUNCTIONAL_POCKET_ROLES:
            continue

        chain_id = (lig.get("chain_id") or "").strip()
        matched_desc: str | None = None
        for c in (c.strip() for c in chain_id.split(",")):
            poly = poly_by_chain.get(c)
            if not poly:
                continue
            desc = (poly.get("description") or "").strip()
            slug = (poly.get("slug") or "").strip().lower()
            if is_g_alpha_description(desc) or slug.startswith(G_PROTEIN_SUBUNIT_SLUG_PREFIXES):
                matched_desc = desc
                break
        if matched_desc is None:
            continue

        name = lig.get("name") or lig.get("chem_comp_id") or "?"
        warnings.append(
            f"{ALERT_PREFIX_G_PROTEIN_LIGAND} at 'ligands': ligand '{name}' "
            f"(chain {chain_id}) is described as '{matched_desc}', a "
            f"G-protein-derived / transducer-mimetic peptide, but is annotated as "
            f"role '{role}'. Verify it belongs under signaling partners, not as a "
            f"receptor agonist."
        )


def _warn_on_apo_with_real_ligands(ligands: list[Any], warnings: list[str]) -> None:
    """Flag (for the curator) an apo placeholder sitting alongside real ligands
    — a contradiction worth a human's eye.  Emits a warning only; the data is
    left untouched so the curator decides what is correct.  A buffer/solvent
    next to an apo entry is normal and does not warn.
    """
    real_statuses = {
        VALIDATION_MATCHED_SMALL_MOLECULE,
        VALIDATION_MATCHED_POLYMER,
        VALIDATION_GHOST_LIGAND,
    }
    has_apo = any(
        isinstance(lig, dict) and lig.get("validation_status") == VALIDATION_SKIPPED_APO
        for lig in ligands
    )
    real = [
        lig
        for lig in ligands
        if isinstance(lig, dict) and lig.get("validation_status") in real_statuses
    ]
    if has_apo and real:
        names = ", ".join(str(lig.get("name") or lig.get("chem_comp_id") or "?") for lig in real)
        warnings.append(
            f"APO_WITH_LIGANDS at 'ligands': an apo (ligand-free) placeholder "
            f"coexists with {len(real)} real ligand(s) [{names}] — verify whether "
            f"this structure is truly apo."
        )


def _warn_on_multiple_agonists(ligands: list[Any], warnings: list[str]) -> None:
    """Flag (for the curator) a structure carrying two or more *distinct* ligands
    each annotated with the plain 'Agonist' role -- a configuration the model may
    have read as two independent agonists when they in fact act together as
    co-agonists (e.g. a metal ion and an amino-acid agonist co-occupying one site).

    Distinct MOLECULES are counted, not entries: one agonist modelled at two sites
    is emitted as two ligand entries (site_ref split) with the same identity and
    must count once. Identity is the chem_comp_id when present, else the
    case-insensitive name. Only the plain 'Agonist' role is considered: 'Co-agonist'
    means the model already recognised the relationship, and 'Allosteric agonist' /
    'Agonist (partial)' / 'Ago-PAM' describe different mechanisms.

    Warning-only and non-asserting: co-agonism is a nuanced call, so the reminder
    asks the curator to verify it rather than declaring it. No role or data is
    changed.
    """
    seen: set[str] = set()
    display_names: list[str] = []
    for lig in ligands:
        if not isinstance(lig, dict):
            continue
        role = ((lig.get("role") or {}).get("value") or "").strip()
        if role != "Agonist":
            continue
        comp_id = (lig.get("chem_comp_id") or "").strip()
        name = (lig.get("name") or "").strip()
        identity = (
            comp_id.lower() if comp_id and comp_id.lower() not in EMPTY_VALUES else name.lower()
        )
        if not identity or identity in seen:
            continue
        seen.add(identity)
        display_names.append(name or comp_id or "?")

    if len(seen) >= 2:
        names = ", ".join(display_names)
        warnings.append(
            f"{ALERT_PREFIX_MULTIPLE_AGONISTS} at 'ligands': two or more distinct "
            f"ligands are annotated as agonists ({names}); verify whether they are "
            f"co-agonists acting together, or whether one is the primary agonist and "
            f"the other is incidental."
        )
