"""Oligomer analysis suite: classification, 7TM completeness, protomer suggestion, alerts.

Classifies GPCR oligomeric state (monomer/homomer/heteromer), scans chains
for 7TM completeness, suggests a primary protomer, generates alerts for
AI hallucinations and missed protomers, and applies smart chain_id overrides.

Blood Lesson 1 — None-safety:
    ``(inst.get("rcsb_polymer_entity_instance_container_identifiers") or {}).get("auth_asym_id")``
Blood Lesson 3 — Warning format:
    Alert messages follow ``f"[{ALERT_TYPE}] at 'oligomer_analysis': description"``.
Blood Lesson 4 — Magic strings:
    All alert types, classifications, and TM statuses are constants from ``config.py``.
"""

from __future__ import annotations

import logging
from typing import Any

from gpcr_tools.config import (
    ALERT_7TM_UPGRADE,
    ALERT_CHAIN_ID_OVERRIDDEN,
    ALERT_CONFIRMED_OLIGOMER,
    ALERT_HALLUCINATION,
    ALERT_MISSED_PROTOMER,
    EMPTY_VALUES,
    GPCR_SLUG_NEGATIVE_PREFIXES,
    OLIGOMER_HETEROMER,
    OLIGOMER_HOMOMER,
    OLIGOMER_MONOMER,
    OLIGOMER_NO_GPCR,
    TM_COVERAGE_THRESHOLD,
    TM_ENTITY_FEATURE_TYPES,
    TM_STATUS_COMPLETE,
    TM_STATUS_INCOMPLETE,
    TM_STATUS_UNKNOWN,
    TM_UNIPROT_FEATURE_TYPES,
)
from gpcr_tools.validator.api_clients import fetch_polymer_features

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def is_gpcr_slug(slug: str) -> bool:
    """Return True if *slug* is a GPCR protein, filtering out known non-GPCRs.

    Uses negative-prefix matching from ``GPCR_SLUG_NEGATIVE_PREFIXES``.
    """
    if not slug:
        return False
    return not slug.lower().startswith(GPCR_SLUG_NEGATIVE_PREFIXES)


def get_sequence_length(entity: dict[str, Any]) -> int:
    """Extract sample sequence length from *entity*'s polymer data.

    Blood Lesson 1: guard ``rcsb_sample_sequence_length`` for None.
    """
    poly = entity.get("entity_poly") or {}
    length = poly.get("rcsb_sample_sequence_length")
    if length is not None:
        return int(length)
    seq = poly.get("pdbx_seq_one_letter_code_can")
    if seq:
        return len(seq)
    return 0


# ---------------------------------------------------------------------------
# 7TM analysis
# ---------------------------------------------------------------------------


def map_uniprot_to_entity(
    u_start: int,
    u_end: int,
    alignments: list[dict[str, Any]],
) -> list[tuple[int, int]]:
    """Map UniProt feature coordinates to entity coordinates via alignment regions."""
    mapped_segments: list[tuple[int, int]] = []
    for reg in alignments:
        ref_start = reg["ref_beg_seq_id"]
        ref_end = ref_start + reg["length"] - 1
        ent_start = reg["entity_beg_seq_id"]

        overlap_start = max(u_start, ref_start)
        overlap_end = min(u_end, ref_end)

        if overlap_start <= overlap_end:
            offset_start = overlap_start - ref_start
            offset_end = overlap_end - ref_start
            e_start = ent_start + offset_start
            e_end = ent_start + offset_end
            mapped_segments.append((e_start, e_end))
    return mapped_segments


def _analyze_tm_for_entity_instance(
    entity: dict[str, Any],
    instance: dict[str, Any],
) -> dict[str, Any]:
    """Analyse 7TM completeness for a single entity/instance pair.

    Returns ``{"resolved_tms": int, "total_tms": int, "status": str}``.
    Status is one of ``TM_STATUS_COMPLETE``, ``TM_STATUS_INCOMPLETE``,
    or ``TM_STATUS_UNKNOWN``.
    """
    tm_regions: list[tuple[int, int]] = []

    # Strategy 1: entity-level membrane features
    for f in entity.get("rcsb_polymer_entity_feature") or []:
        if (f.get("type") or "").upper() in TM_ENTITY_FEATURE_TYPES:
            for pos in f.get("feature_positions") or []:
                tm_regions.append((pos["beg_seq_id"], pos["end_seq_id"]))

    # Strategy 2: fallback to UniProt features mapped through alignments
    if not tm_regions:
        align_by_accession: dict[str, list[dict[str, Any]]] = {}
        for align in entity.get("rcsb_polymer_entity_align") or []:
            if align.get("reference_database_name") == "UniProt":
                acc = align.get("reference_database_accession") or ""
                align_by_accession.setdefault(acc, []).extend(align.get("aligned_regions") or [])

        for u in entity.get("uniprots") or []:
            uid = u.get("rcsb_id") or ""
            u_alignments = align_by_accession.get(uid, [])
            for f in u.get("rcsb_uniprot_feature") or []:
                f_type = (f.get("type") or "").upper()
                if f_type not in TM_UNIPROT_FEATURE_TYPES:
                    continue
                if f_type == "TOPOLOGICAL_DOMAIN":
                    desc = (f.get("description") or "").upper()
                    if "TRANSMEMBRANE" not in desc and "MEMBRANE" not in desc:
                        continue
                for pos in f.get("feature_positions") or []:
                    mapped = map_uniprot_to_entity(
                        pos["beg_seq_id"], pos["end_seq_id"], u_alignments
                    )
                    tm_regions.extend(mapped)

    if not tm_regions:
        return {"resolved_tms": 0, "total_tms": 0, "status": TM_STATUS_UNKNOWN}

    # Collect and merge unmodeled regions from instance features
    unmodeled_regions: list[tuple[int, int]] = []
    for f in instance.get("rcsb_polymer_instance_feature") or []:
        if f.get("type") in ("UNOBSERVED_RESIDUE_XYZ", "UNMODELED"):
            for pos in f.get("feature_positions") or []:
                unmodeled_regions.append((pos["beg_seq_id"], pos["end_seq_id"]))

    unmodeled_regions.sort(key=lambda x: x[0])
    merged_unmodeled: list[tuple[int, int]] = []
    for current in unmodeled_regions:
        if not merged_unmodeled:
            merged_unmodeled.append(current)
        else:
            prev = merged_unmodeled[-1]
            if current[0] <= prev[1]:
                merged_unmodeled[-1] = (prev[0], max(prev[1], current[1]))
            else:
                merged_unmodeled.append(current)

    resolved_tms = 0
    for tm_start, tm_end in tm_regions:
        tm_length = tm_end - tm_start + 1
        if tm_length <= 0:
            continue
        unmodeled_count = 0
        for un_start, un_end in merged_unmodeled:
            ov_start = max(tm_start, un_start)
            ov_end = min(tm_end, un_end)
            if ov_start <= ov_end:
                unmodeled_count += ov_end - ov_start + 1
        coverage = (tm_length - unmodeled_count) / tm_length
        if coverage >= TM_COVERAGE_THRESHOLD:
            resolved_tms += 1

    total_tms = len(tm_regions)
    status = TM_STATUS_COMPLETE if resolved_tms >= 6 else TM_STATUS_INCOMPLETE
    return {"resolved_tms": resolved_tms, "total_tms": total_tms, "status": status}


def scan_all_chains_7tm(
    pdb_id: str,
    gpcr_chain_ids: set[str],
    graphql_entry: dict[str, Any] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any] | None]:
    """Scan all GPCR chains for 7TM completeness.

    Returns ``(results, graphql_entry)`` where *results* maps
    ``auth_asym_id`` to TM analysis dicts.
    """
    if graphql_entry is None:
        graphql_entry = fetch_polymer_features(pdb_id)
    if not graphql_entry:
        return {}, None

    results: dict[str, dict[str, Any]] = {}
    for entity in graphql_entry.get("polymer_entities") or []:
        for inst in entity.get("polymer_entity_instances") or []:
            auth_id = (inst.get("rcsb_polymer_entity_instance_container_identifiers") or {}).get(
                "auth_asym_id"
            )
            if not auth_id or auth_id not in gpcr_chain_ids:
                continue
            results[auth_id] = _analyze_tm_for_entity_instance(entity, inst)

    return results, graphql_entry


# ---------------------------------------------------------------------------
# GPCR roster
# ---------------------------------------------------------------------------


def _build_gpcr_roster(
    enriched_entry: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Build ``{auth_asym_id: {"slug": str, "length": int, "asym_id": str}}`` for GPCR chains."""
    roster: dict[str, dict[str, Any]] = {}
    for entity in enriched_entry.get("polymer_entities") or []:
        if not isinstance(entity, dict):
            continue
        slug: str | None = None
        for u in entity.get("uniprots") or []:
            if not isinstance(u, dict):
                continue
            s = u.get("gpcrdb_entry_name_slug")
            if s and is_gpcr_slug(s):
                slug = s
                break
        if not slug:
            continue
        length = get_sequence_length(entity)
        for inst in entity.get("polymer_entity_instances") or []:
            if not isinstance(inst, dict):
                continue
            cid = inst.get("rcsb_polymer_entity_instance_container_identifiers") or {}
            auth = cid.get("auth_asym_id")
            asym = cid.get("asym_id")
            if auth:
                roster[auth] = {"slug": slug, "length": length, "asym_id": asym or auth}
    return roster


def _refine_fusion_slugs(
    gpcr_roster: dict[str, dict[str, Any]],
    enriched_entry: dict[str, Any],
    graphql_entry: dict[str, Any] | None,
) -> None:
    """Correct misassigned slugs in fusion constructs (mutates *gpcr_roster* in-place).

    When an entity is a chimera (e.g. mTOR-mGlu7), the initial roster build may
    pick the wrong UniProt's slug.  This cross-references per-UniProt TM features
    from GraphQL data to identify the actual GPCR component.
    """
    if not graphql_entry or not gpcr_roster:
        return

    accession_to_slug: dict[str, str] = {}
    for entity in enriched_entry.get("polymer_entities") or []:
        if not isinstance(entity, dict):
            continue
        for u in entity.get("uniprots") or []:
            if not isinstance(u, dict):
                continue
            acc = u.get("rcsb_id") or ""
            slug = u.get("gpcrdb_entry_name_slug") or ""
            if acc and slug:
                accession_to_slug[acc] = slug

    chain_to_gql_entity: dict[str, dict[str, Any]] = {}
    for entity in graphql_entry.get("polymer_entities") or []:
        if not isinstance(entity, dict):
            continue
        for inst in entity.get("polymer_entity_instances") or []:
            if not isinstance(inst, dict):
                continue
            auth_id = (inst.get("rcsb_polymer_entity_instance_container_identifiers") or {}).get(
                "auth_asym_id"
            )
            if auth_id and auth_id in gpcr_roster:
                chain_to_gql_entity[auth_id] = entity

    for chain_id, info in gpcr_roster.items():
        gql_entity = chain_to_gql_entity.get(chain_id)
        if not gql_entity:
            continue
        uniprots = gql_entity.get("uniprots") or []
        if len(uniprots) <= 1:
            continue

        best_acc: str | None = None
        best_tm_count = 0
        for u in uniprots:
            if not isinstance(u, dict):
                continue
            acc = u.get("rcsb_id") or ""
            tm_count = sum(
                len(f.get("feature_positions") or [])
                for f in u.get("rcsb_uniprot_feature") or []
                if (f.get("type") or "").upper() in ("TRANSMEMBRANE", "TRANSMEMBRANE_REGION")
            )
            if tm_count > best_tm_count:
                best_tm_count = tm_count
                best_acc = acc

        if not best_acc or best_acc not in accession_to_slug:
            continue
        correct_slug = accession_to_slug[best_acc]
        if correct_slug != info["slug"] and is_gpcr_slug(correct_slug):
            logger.info(
                "[%s] Fusion slug correction: '%s' -> '%s' (UniProt %s has %d TM regions)",
                chain_id,
                info["slug"],
                correct_slug,
                best_acc,
                best_tm_count,
            )
            info["slug"] = correct_slug


# ---------------------------------------------------------------------------
# Label / assembly helpers
# ---------------------------------------------------------------------------


def _build_label_asym_id_map(
    enriched_entry: dict[str, Any],
) -> dict[str, str]:
    """Build ``{auth_asym_id: asym_id}`` for ALL polymer chains (not just GPCR)."""
    mapping: dict[str, str] = {}
    for entity in enriched_entry.get("polymer_entities") or []:
        if not isinstance(entity, dict):
            continue
        for inst in entity.get("polymer_entity_instances") or []:
            if not isinstance(inst, dict):
                continue
            cid = inst.get("rcsb_polymer_entity_instance_container_identifiers") or {}
            auth = cid.get("auth_asym_id")
            asym = cid.get("asym_id")
            if auth and asym:
                mapping[auth] = asym
    return mapping


def _get_assembly_cross_check(
    enriched_entry: dict[str, Any],
) -> dict[str, Any]:
    """Extract oligomeric_state from first assembly for informational annotation."""
    for asm in enriched_entry.get("assemblies") or []:
        if not isinstance(asm, dict):
            continue
        for sym in asm.get("rcsb_struct_symmetry") or []:
            if not isinstance(sym, dict):
                continue
            return {
                "oligomeric_state": sym.get("oligomeric_state"),
                "stoichiometry": sym.get("stoichiometry"),
                "kind": sym.get("kind"),
                "type": sym.get("type"),
            }
    return {}


# ---------------------------------------------------------------------------
# Protomer suggestion (5-rank framework)
# ---------------------------------------------------------------------------


def _suggest_primary_protomer(
    gpcr_roster: dict[str, dict[str, Any]],
    tm_roster: dict[str, dict[str, Any]],
    classification: str,
    ai_chain: str | None,
    signaling_partners: dict[str, Any],
    ligands: list[dict[str, Any]],
) -> dict[str, Any]:
    """Suggest a primary protomer chain using the 5-rank framework.

    Rank 0: Homomer context (identical chains, prefer better 7TM).
    Rank 1: G-protein bound (AI's chain if in roster and G-protein present).
    Rank 2: Exclusive ligand-binding chain.
    Rank 3: Best 7TM completeness.
    Rank 4: Longest sequence OR valid AI choice.
    """
    if not gpcr_roster:
        return {"chain_id": None, "reason": "No GPCR chains found", "rank_used": None}

    primary: str | None = None
    reason = ""
    rank: int | None = None

    # Rank 1: G-protein bound
    has_gprotein = False
    if signaling_partners:
        if "g_protein" in signaling_partners:
            has_gprotein = True
        else:
            sp_str = str(signaling_partners).lower()
            if any(tag in sp_str for tag in ("gnai", "gnas", "gnaq", "gnao")):
                has_gprotein = True

    if has_gprotein and ai_chain and ai_chain in gpcr_roster:
        primary = ai_chain
        reason = f"Rank 1: G-protein bound — AI-determined active complex on Chain {primary}"
        rank = 1

    # Rank 2: Exclusive ligand binding
    if not primary:
        ligand_chains: set[str] = set()
        for lig in ligands:
            if not isinstance(lig, dict):
                continue
            lc = lig.get("chain_id")
            if lc and str(lc).lower() not in EMPTY_VALUES:
                ligand_chains.add(str(lc))
        bound_gpcrs = [c for c in gpcr_roster if c in ligand_chains]
        if len(bound_gpcrs) == 1:
            primary = bound_gpcrs[0]
            reason = f"Rank 2: Ligand binds exclusively to GPCR Chain {primary}"
            rank = 2

    # Rank 3: Best 7TM completeness
    if not primary and tm_roster:
        scored = sorted(
            [
                (c, (tm_roster.get(c) or {"resolved_tms": 0}).get("resolved_tms", 0))
                for c in gpcr_roster
            ],
            key=lambda x: -x[1],
        )
        if scored and scored[0][1] > 0 and (len(scored) < 2 or scored[0][1] > scored[1][1]):
            primary = scored[0][0]
            tm_str = ", ".join(f"Chain {c}: {t}/7" for c, t in scored)
            reason = f"Rank 3: Best 7TM completeness ({tm_str})"
            rank = 3

    # Rank 4: Valid AI choice OR longest sequence
    if not primary:
        valid_ai_chains: list[str] = []
        if ai_chain:
            valid_ai_chains = [
                c.strip() for c in str(ai_chain).split(",") if c.strip() in gpcr_roster
            ]

        if valid_ai_chains:
            primary = valid_ai_chains[0]
            reason = f"Rank 4: Preserving AI's originally correct choice (Chain {primary})"
            rank = 4
        else:
            sorted_by_len = sorted(gpcr_roster.items(), key=lambda x: -x[1]["length"])
            primary = sorted_by_len[0][0]
            len_str = ", ".join(f"Chain {c}: {info['length']}aa" for c, info in sorted_by_len)
            reason = f"Rank 4: Longest sequence ({len_str})"
            rank = 4

    # Prepend classification context for Rank 0 (homomer)
    if classification == OLIGOMER_HOMOMER:
        reason = f"Homomer ({len(gpcr_roster)} identical GPCR chains) — {reason}"
        rank = 0

    return {"chain_id": primary, "reason": reason, "rank_used": rank}


# ---------------------------------------------------------------------------
# Alert generation
# ---------------------------------------------------------------------------


def _generate_alerts(
    gpcr_roster: dict[str, dict[str, Any]],
    classification: str,
    ai_chain: str | None,
    best_run_data: dict[str, Any],
) -> list[dict[str, str]]:
    """Generate non-invasive oligomer alerts.

    Blood Lesson 3: alert messages follow
    ``f"[{ALERT_TYPE}] at 'oligomer_analysis': ..."``
    """
    alerts: list[dict[str, str]] = []

    if not ai_chain or not gpcr_roster:
        return alerts

    ai_chains = {c.strip() for c in str(ai_chain).split(",") if c.strip()}
    roster_keys = set(gpcr_roster.keys())
    non_gpcr = ai_chains - roster_keys
    gpcr_hits = ai_chains & roster_keys

    # HALLUCINATION: AI-selected chain not in GPCR roster
    if non_gpcr:
        alerts.append(
            {
                "type": ALERT_HALLUCINATION,
                "message": (
                    f"[{ALERT_HALLUCINATION}] at 'oligomer_analysis': "
                    f"AI selected chain(s) {sorted(non_gpcr)} which are NOT in the GPCR roster "
                    f"(roster: {sorted(roster_keys)}). "
                    f"The AI may have picked a G-protein, nanobody, or other non-GPCR chain."
                ),
            }
        )

    # MISSED_PROTOMER / CONFIRMED_OLIGOMER
    if len(gpcr_roster) > 1 and gpcr_hits:
        missed = roster_keys - ai_chains
        if missed:
            alerts.append(
                {
                    "type": ALERT_MISSED_PROTOMER,
                    "message": (
                        f"[{ALERT_MISSED_PROTOMER}] at 'oligomer_analysis': "
                        f"GPCR roster has chains {sorted(roster_keys)} "
                        f"but AI only reported {sorted(gpcr_hits)}. "
                        f"Missed: {sorted(missed)}."
                    ),
                }
            )
        else:
            alerts.append(
                {
                    "type": ALERT_CONFIRMED_OLIGOMER,
                    "message": (
                        f"[{ALERT_CONFIRMED_OLIGOMER}] at 'oligomer_analysis': "
                        f"AI reported chain(s) {sorted(ai_chains)} — "
                        f"matches GPCR roster {sorted(roster_keys)}."
                    ),
                }
            )

    return alerts


# ---------------------------------------------------------------------------
# Chain override
# ---------------------------------------------------------------------------


def _apply_chain_override(
    receptor_info: dict[str, Any],
    ai_chain: str | None,
    suggestion: dict[str, Any],
    gpcr_roster: dict[str, dict[str, Any]],
    tm_roster: dict[str, dict[str, Any]],
    alerts: list[dict[str, str]],
) -> dict[str, Any]:
    """Smart override: correct ``receptor_info`` when AI is objectively wrong.

    Two trigger conditions:
      1. HALLUCINATION — AI's chain not in GPCR roster.
      2. 7TM_UPGRADE — AI's chain INCOMPLETE_7TM, suggestion's chain COMPLETE.

    When triggered, both ``chain_id`` and ``uniprot_entry_name`` are corrected.
    Original AI values are recorded for transparency.

    Blood Lesson 7: return dict includes ``original_chain_id`` and
    ``corrected_chain_id`` as explicit keys.
    """
    suggested_chain = suggestion.get("chain_id")

    if not ai_chain or not suggested_chain or not receptor_info:
        return {"applied": False, "reason": "No chain data available"}

    if ai_chain == suggested_chain:
        return {
            "applied": False,
            "reason": "AI chain matches suggestion — no override needed",
        }

    def _do_override(trigger: str, detail: str) -> dict[str, Any]:
        original_chain = ai_chain
        original_uniprot = receptor_info.get("uniprot_entry_name")
        corrected_slug = (gpcr_roster.get(suggested_chain) or {}).get("slug")

        receptor_info["chain_id"] = suggested_chain
        if corrected_slug:
            receptor_info["uniprot_entry_name"] = corrected_slug

        msg = (
            f"[{ALERT_CHAIN_ID_OVERRIDDEN}] at 'oligomer_analysis': "
            f"receptor_info corrected: "
            f"chain_id '{original_chain}' -> '{suggested_chain}', "
            f"uniprot_entry_name '{original_uniprot}' -> "
            f"'{corrected_slug or original_uniprot}'. "
            f"Reason: {trigger} — {detail}"
        )
        alerts.append({"type": ALERT_CHAIN_ID_OVERRIDDEN, "message": msg})
        return {
            "applied": True,
            "trigger": trigger,
            "original_chain_id": original_chain,
            "corrected_chain_id": suggested_chain,
            "original_uniprot": original_uniprot,
            "corrected_uniprot": corrected_slug or original_uniprot,
            "reason": msg,
        }

    # Trigger 1: HALLUCINATION
    if any(a["type"] == ALERT_HALLUCINATION for a in alerts):
        return _do_override(
            ALERT_HALLUCINATION,
            f"AI selected Chain {ai_chain} which is not a GPCR.",
        )

    # Trigger 2: 7TM_UPGRADE
    ai_tm = tm_roster.get(ai_chain) or {}
    suggested_tm = tm_roster.get(suggested_chain) or {}
    ai_status = ai_tm.get("status")
    suggested_status = suggested_tm.get("status")

    if ai_status == TM_STATUS_INCOMPLETE and suggested_status == TM_STATUS_COMPLETE:
        return _do_override(
            ALERT_7TM_UPGRADE,
            f"Chain {ai_chain} has "
            f"{ai_tm.get('resolved_tms', '?')}/{ai_tm.get('total_tms', '?')} TMs "
            f"({TM_STATUS_INCOMPLETE}), "
            f"Chain {suggested_chain} has "
            f"{suggested_tm.get('resolved_tms', '?')}/{suggested_tm.get('total_tms', '?')} TMs "
            f"({TM_STATUS_COMPLETE}).",
        )

    return {
        "applied": False,
        "reason": (
            f"AI chain '{ai_chain}' differs from suggestion '{suggested_chain}' "
            f"but no objective override trigger met (AI 7TM: {ai_status}, "
            f"suggestion 7TM: {suggested_status})"
        ),
    }


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def analyze_oligomer(
    pdb_id: str,
    best_run_data: dict[str, Any],
    enriched_entry: dict[str, Any],
) -> None:
    """Run oligomer analysis on *best_run_data* against *enriched_entry*.

    Writes ``best_run_data["oligomer_analysis"]`` in-place.
    May correct ``receptor_info.chain_id`` and ``uniprot_entry_name``
    when AI is objectively wrong (HALLUCINATION or 7TM_UPGRADE).
    """
    # 1. Build GPCR roster
    gpcr_roster = _build_gpcr_roster(enriched_entry)

    # 2. Scan all GPCR chains for 7TM
    tm_roster: dict[str, dict[str, Any]] = {}
    graphql_entry: dict[str, Any] | None = None
    if gpcr_roster:
        tm_roster, graphql_entry = scan_all_chains_7tm(pdb_id, set(gpcr_roster.keys()))

    # 3. Refine fusion slugs using per-UniProt TM features
    _refine_fusion_slugs(gpcr_roster, enriched_entry, graphql_entry)

    # 4. Classify (after refinement so slugs are correct)
    unique_slugs = {info["slug"] for info in gpcr_roster.values()}

    if len(gpcr_roster) == 0:
        classification = OLIGOMER_NO_GPCR
    elif len(gpcr_roster) == 1:
        classification = OLIGOMER_MONOMER
    elif len(unique_slugs) == 1:
        classification = OLIGOMER_HOMOMER
    else:
        classification = OLIGOMER_HETEROMER

    all_gpcr_chains: list[dict[str, Any]] = []
    for chain_id in sorted(gpcr_roster.keys()):
        info = gpcr_roster[chain_id]
        tm = tm_roster.get(chain_id) or {
            "resolved_tms": 0,
            "total_tms": 0,
            "status": TM_STATUS_UNKNOWN,
        }
        all_gpcr_chains.append(
            {
                "chain_id": chain_id,
                "slug": info["slug"],
                "7tm_status": tm["status"],
                "resolved_tms": tm["resolved_tms"],
                "total_tms": tm["total_tms"],
            }
        )

    # 5. Primary protomer suggestion
    receptor_info = best_run_data.get("receptor_info") or {}
    ai_chain = receptor_info.get("chain_id")
    signaling_partners = best_run_data.get("signaling_partners") or {}
    ligands_data = best_run_data.get("ligands") or []

    suggestion = _suggest_primary_protomer(
        gpcr_roster,
        tm_roster,
        classification,
        ai_chain,
        signaling_partners,
        ligands_data,
    )

    # 6. Alerts
    alerts = _generate_alerts(
        gpcr_roster,
        classification,
        ai_chain,
        best_run_data,
    )

    # 7. Smart override: correct chain_id when AI is objectively wrong
    override_info = _apply_chain_override(
        receptor_info,
        ai_chain,
        suggestion,
        gpcr_roster,
        tm_roster,
        alerts,
    )

    # 8. label_asym_id map
    label_map = _build_label_asym_id_map(enriched_entry)

    # 9. Assembly cross-check (informational only)
    assembly_info = _get_assembly_cross_check(enriched_entry)

    # 10. Write output
    best_run_data["oligomer_analysis"] = {
        "classification": classification,
        "all_gpcr_chains": all_gpcr_chains,
        "primary_protomer_suggestion": suggestion,
        "assembly_cross_check": assembly_info,
        "alerts": alerts,
        "chain_id_override": override_info,
        "label_asym_id_map": label_map,
    }
