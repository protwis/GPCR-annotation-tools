"""Oligomer analysis suite: classification, 7TM completeness, protomer suggestion, alerts.

Classifies GPCR oligomeric state (monomer/homomer/heteromer), scans chains
for 7TM completeness, suggests a primary protomer, generates alerts for
AI hallucinations and missed protomers, and applies smart chain_id overrides.

Conventions:
    - None-safe: ``(... or {}).get("auth_asym_id")`` at every nested access.
    - Alert messages follow ``f"[{ALERT_TYPE}] at 'oligomer_analysis': description"``.
    - All alert types, classifications, and TM statuses are constants from ``config.py``.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from gpcr_tools.config import (
    ALERT_7TM_UPGRADE,
    ALERT_ASSEMBLY_MISMATCH,
    ALERT_CHAIN_ID_OVERRIDDEN,
    ALERT_CONFIRMED_OLIGOMER,
    ALERT_HALLUCINATION,
    ALERT_MISSED_PROTOMER,
    ALERT_MULTI_COPY_LIGAND,
    ALERT_NO_GPCR,
    ALERT_PREFIX_FUSION_NOTE,
    ALERT_PREFIX_MISSED_POLYMER,
    ALERT_PROTOMER_IN_AUXILIARY,
    ALERT_SUSPICIOUS_7TM,
    ALERT_TM_DATA_UNAVAILABLE,
    APO_SENTINEL,
    CRYSTALLIZATION_FUSION_KEYWORDS,
    CRYSTALLIZATION_FUSION_SLUGS,
    EMPTY_VALUES,
    GPCR_MIN_ANNOTATED_TM,
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
from gpcr_tools.validator.api_clients import (
    PolymerFeaturesCacheLike,
    fetch_polymer_features,
    fetch_polymer_features_cached,
)

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

    Guard ``rcsb_sample_sequence_length`` for None.
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
        ref_start = reg.get("ref_beg_seq_id")
        length = reg.get("length")
        ent_start = reg.get("entity_beg_seq_id")
        # RCSB can omit/null these coordinate fields; skip the region rather
        # than raise (which would fail the whole PDB's oligomer analysis).
        if ref_start is None or length is None or ent_start is None:
            continue
        ref_end = ref_start + length - 1

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
                beg, end = pos.get("beg_seq_id"), pos.get("end_seq_id")
                if beg is not None and end is not None:
                    tm_regions.append((beg, end))

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
                    beg, end = pos.get("beg_seq_id"), pos.get("end_seq_id")
                    if beg is None or end is None:
                        continue
                    tm_regions.extend(map_uniprot_to_entity(beg, end, u_alignments))

    if not tm_regions:
        return {"resolved_tms": 0, "total_tms": 0, "status": TM_STATUS_UNKNOWN}

    # Collect and merge unmodeled regions from instance features
    unmodeled_regions: list[tuple[int, int]] = []
    for f in instance.get("rcsb_polymer_instance_feature") or []:
        if f.get("type") in ("UNOBSERVED_RESIDUE_XYZ", "UNMODELED"):
            for pos in f.get("feature_positions") or []:
                beg, end = pos.get("beg_seq_id"), pos.get("end_seq_id")
                if beg is not None and end is not None:
                    unmodeled_regions.append((beg, end))

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
    cache: PolymerFeaturesCacheLike | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any] | None]:
    """Scan all GPCR chains for 7TM completeness.

    Returns ``(results, graphql_entry)`` where *results* maps
    ``auth_asym_id`` to TM analysis dicts.

    When *graphql_entry* is not supplied it is fetched live; passing *cache*
    serves a fresh copy from disk and stores a successful fetch, so the 7TM
    count survives a transient RCSB outage.
    """
    if graphql_entry is None:
        graphql_entry = (
            fetch_polymer_features_cached(pdb_id, cache)
            if cache is not None
            else fetch_polymer_features(pdb_id)
        )
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


def format_7tm_status(tm: dict[str, Any] | None) -> str:
    """Render a TM analysis dict as a ``"STATUS resolved/total"`` string.

    ``None`` (no analysis available) renders as ``"UNKNOWN 0/0"`` -- the same
    shape a chain with no transmembrane annotation produces, so a missing fetch
    and a genuine non-7TM chain read identically as "not a 7TM receptor".
    """
    tm = tm or {}
    status = tm.get("status") or TM_STATUS_UNKNOWN
    resolved = tm.get("resolved_tms", 0)
    total = tm.get("total_tms", 0)
    return f"{status} {resolved}/{total}"


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


# ---------------------------------------------------------------------------
# Missed non-GPCR polymer reconciliation
# ---------------------------------------------------------------------------


# Slug prefixes that mark a chain as a signaling partner (G-protein alpha via
# "gna", beta "gbb", gamma "gbg", arrestin "arr"); an unannotated chain with one
# of these is routed to the signaling_partners review block, everything else to
# auxiliary_proteins. Used only to anchor the alert to the right block.
_SIGNALING_SLUG_PREFIXES: tuple[str, ...] = ("gna", "gbb", "gbg", "arr")


def _split_chain_ids(value: Any) -> set[str]:
    """Parse a chain_id field (single, comma- or semicolon-separated) to a set."""
    if not value or not isinstance(value, str):
        return set()
    out: set[str] = set()
    for part in value.replace(";", ",").split(","):
        token = part.strip()
        if token and token.lower() not in EMPTY_VALUES and token.lower() != APO_SENTINEL:
            out.add(token)
    return out


def _build_all_polymer_chains(enriched_entry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Build ``{auth_asym_id: {"description": str, "slug": str | None}}`` per chain.

    Covers all polymer entities (GPCR and non-GPCR alike). Branched
    oligosaccharides and non-polymer ligands live in other buckets and are not
    included here. The slug (when present) only anchors the alert to a review
    block; it never decides whether a chain is flagged.
    """
    chains: dict[str, dict[str, Any]] = {}
    for entity in enriched_entry.get("polymer_entities") or []:
        if not isinstance(entity, dict):
            continue
        desc = (
            (entity.get("rcsb_polymer_entity") or {}).get("pdbx_description")
        ) or "unknown polymer"
        slug: str | None = None
        for u in entity.get("uniprots") or []:
            if isinstance(u, dict) and u.get("gpcrdb_entry_name_slug"):
                slug = u["gpcrdb_entry_name_slug"]
                break
        for inst in entity.get("polymer_entity_instances") or []:
            if not isinstance(inst, dict):
                continue
            auth = (inst.get("rcsb_polymer_entity_instance_container_identifiers") or {}).get(
                "auth_asym_id"
            )
            if auth:
                chains[auth] = {"description": desc, "slug": slug}
    return chains


def _missed_chain_block(slug: str | None) -> str:
    """Pick the review block an unannotated chain should surface under.

    The curate UI buckets a warning to a block when the block key appears in the
    warning text, so the alert must name a real block. Signaling partners go to
    ``signaling_partners``; everything else (nanobody / scFv / RAMP / fusion /
    peptide) goes to ``auxiliary_proteins`` where a curator adds missed partners.
    """
    if slug and slug.lower().startswith(_SIGNALING_SLUG_PREFIXES):
        return "signaling_partners"
    return "auxiliary_proteins"


def collect_ai_claimed_chains(best_run_data: dict[str, Any]) -> set[str]:
    """Union every polymer chain id the annotation claims, across all slots.

    The model can name a chain under the receptor, the G-protein subunits, the
    arrestin, the auxiliary proteins (nanobody / scFv / RAMP / ...), or a
    polymeric ligand. A chain claimed in any slot counts as annotated.
    """
    claimed: set[str] = set()
    claimed |= _split_chain_ids((best_run_data.get("receptor_info") or {}).get("chain_id"))

    partners = best_run_data.get("signaling_partners") or {}
    g_protein = partners.get("g_protein") or {}
    for subunit in ("alpha_subunit", "beta_subunit", "gamma_subunit"):
        claimed |= _split_chain_ids((g_protein.get(subunit) or {}).get("chain_id"))
    claimed |= _split_chain_ids((partners.get("arrestin") or {}).get("chain_id"))

    for aux in best_run_data.get("auxiliary_proteins") or []:
        if isinstance(aux, dict):
            claimed |= _split_chain_ids(aux.get("chain_id"))
    for ligand in best_run_data.get("ligands") or []:
        if isinstance(ligand, dict):
            claimed |= _split_chain_ids(ligand.get("chain_id"))
    return claimed


def reconcile_missed_polymers(
    enriched_entry: dict[str, Any],
    best_run_data: dict[str, Any],
) -> list[str]:
    """Flag non-GPCR polymer chains present in the structure but unannotated.

    GPCR chains are the missed-protomer check's responsibility and are excluded
    here. Returns one warning string per unannotated non-GPCR polymer chain,
    anchored to the review block a curator would add it under.
    """
    all_chains = _build_all_polymer_chains(enriched_entry)
    if not all_chains:
        return []
    # Exclude every chain a GPCR slug identifies (via the full roster, not the
    # 7TM-gated classify_roster): those belong to the missed-protomer check.
    gpcr_chains = set(_build_gpcr_roster(enriched_entry))
    claimed = collect_ai_claimed_chains(best_run_data)

    # A chain the model named as the receptor but that a chain-id override later
    # corrected already carries a HALLUCINATION alert; do not re-flag it here.
    override = (best_run_data.get("oligomer_analysis") or {}).get("chain_id_override") or {}
    if override.get("applied"):
        claimed |= _split_chain_ids(override.get("original_chain_id"))

    warnings: list[str] = []
    for auth, info in sorted(all_chains.items()):
        if auth in gpcr_chains or auth in claimed:
            continue
        block = _missed_chain_block(info["slug"])
        warnings.append(
            f"{ALERT_PREFIX_MISSED_POLYMER} at '{block}': chain '{auth}' "
            f"('{info['description']}') is present in the structure but not "
            f"annotated; confirm it."
        )
    return warnings


def detect_crystallization_fusions(enriched_entry: dict[str, Any]) -> list[str]:
    """Note any receptor entity carrying a BRIL / T4-lysozyme crystallization fusion.

    BRIL (cytochrome b562RIL) and T4 lysozyme are engineering aids fused into a
    receptor to aid crystallization, not part of the biological receptor. They
    are detected by a fusion-partner slug or a description keyword, but only on
    an entity that is itself a GPCR (so a standalone lysozyme is not flagged).
    A fusion modelled as its own separate entity (no GPCR slug) is intentionally
    left to :func:`reconcile_missed_polymers`, which flags it as an unannotated
    chain. Advisory only -- returns non-blocking notes anchored to receptor_info.
    """
    notes: list[str] = []
    for entity in enriched_entry.get("polymer_entities") or []:
        if not isinstance(entity, dict):
            continue
        slugs = [
            (u.get("gpcrdb_entry_name_slug") or "")
            for u in (entity.get("uniprots") or [])
            if isinstance(u, dict)
        ]
        if not any(is_gpcr_slug(s) for s in slugs):
            continue  # only receptor entities can carry a receptor-side fusion
        description = ((entity.get("rcsb_polymer_entity") or {}).get("pdbx_description")) or ""
        has_fusion_slug = any(s.lower().startswith(CRYSTALLIZATION_FUSION_SLUGS) for s in slugs)
        desc_lower = description.lower()
        has_fusion_keyword = any(kw in desc_lower for kw in CRYSTALLIZATION_FUSION_KEYWORDS)
        if not (has_fusion_slug or has_fusion_keyword):
            continue
        chain_set: set[str] = set()
        for inst in entity.get("polymer_entity_instances") or []:
            if not isinstance(inst, dict):
                continue
            auth = (inst.get("rcsb_polymer_entity_instance_container_identifiers") or {}).get(
                "auth_asym_id"
            )
            if auth:
                chain_set.add(auth)
        chain_str = ", ".join(sorted(chain_set)) or "?"
        notes.append(
            f"{ALERT_PREFIX_FUSION_NOTE} at 'receptor_info': chain(s) {chain_str} "
            f"carry a crystallization fusion ('{description}'); confirm the receptor "
            f"annotation excludes the fusion partner."
        )
    return notes


def build_nonpolymer_instance_index(
    enriched_entry: dict[str, Any],
) -> dict[str, list[dict[str, str]]]:
    """Index every modelled copy of each small-molecule (non-polymer) component.

    An annotation records a ligand by its chemical component, but a structure can
    contain several modelled copies of the same component -- ions, lipids such as
    cholesterol, or a ligand bound at more than one site.  Copies are told apart
    only by their PDB instance identifier (``label_asym_id``): the author chain
    (``auth_asym_id``) is frequently shared between copies, so it cannot
    distinguish them on its own.

    Returns ``{component_id: [{"auth_asym_id", "label_asym_id", "auth_seq_id"}, ...]}``,
    with each component's instance list sorted by ``label_asym_id`` for stable
    output.  Instances missing a usable identifier, and entities missing a
    component id, are skipped.

    RCSB names the per-instance label identifier ``asym_id`` inside the instance
    container identifiers (``auth_asym_id`` is the author chain) -- the same
    convention :func:`_build_label_asym_id_map` relies on for polymer chains.
    """
    index: dict[str, list[dict[str, str]]] = {}
    for entity in enriched_entry.get("nonpolymer_entities") or []:
        if not isinstance(entity, dict):
            continue
        entity_ids = entity.get("rcsb_nonpolymer_entity_container_identifiers") or {}
        comp_id = entity_ids.get("nonpolymer_comp_id")
        if not comp_id:
            continue
        for inst in entity.get("nonpolymer_entity_instances") or []:
            if not isinstance(inst, dict):
                continue
            cid = inst.get("rcsb_nonpolymer_entity_instance_container_identifiers") or {}
            label_asym_id = cid.get("asym_id")
            if not label_asym_id:
                continue
            index.setdefault(comp_id, []).append(
                {
                    "auth_asym_id": cid.get("auth_asym_id") or "",
                    "label_asym_id": label_asym_id,
                    "auth_seq_id": cid.get("auth_seq_id") or "",
                }
            )
    for instances in index.values():
        instances.sort(key=lambda rec: rec["label_asym_id"])
    return index


def find_multi_copy_components(
    instance_index: dict[str, list[dict[str, str]]],
) -> dict[str, int]:
    """Return ``{component_id: copy_count}`` for components modelled more than once.

    A copy count above one means a single ligand entry stands for several modelled
    copies -- the signal a reviewer needs when deciding whether those copies play
    distinct roles.
    """
    return {
        comp_id: len(instances)
        for comp_id, instances in instance_index.items()
        if len(instances) > 1
    }


def _get_assembly_cross_check(
    enriched_entry: dict[str, Any],
) -> dict[str, Any]:
    """Extract the biological-assembly oligomeric state for informational annotation.

    RCSB can deposit several assemblies per entry (e.g. the author-provided one
    plus software-predicted alternatives).  The assembly RCSB marks as the
    representative biological unit carries ``pdbx_struct_assembly.rcsb_candidate_assembly
    == "Y"``; that one is preferred here, falling back to the first assembly only
    when none is flagged.  The first global-symmetry block of the chosen assembly
    supplies ``oligomeric_state`` / ``stoichiometry`` / ``kind`` / ``type``; the
    candidate flag and modeled-monomer count are surfaced alongside so a caller can
    reconcile the GPCR-centric classification against the biological assembly.

    The first block's state is kept as the displayed ``oligomeric_state`` (changing
    it would perturb the surfaced annotation), but ALL symmetry blocks of the chosen
    assembly are scanned for a derived ``has_homo_symmetry`` flag -- True when ANY
    block's ``oligomeric_state`` starts with "Homo" (case-insensitive). RCSB can
    record the homo-oligomer in a later Local/Pseudo block rather than the first
    Global one, so reading only the first block misses a real homodimer.

    All data is already in the enriched entry -- no network call.  Returns ``{}``
    when no assembly carries a symmetry block.
    """
    assemblies = [a for a in (enriched_entry.get("assemblies") or []) if isinstance(a, dict)]
    if not assemblies:
        return {}

    # Prefer the assembly RCSB flags as the biological candidate; fall back to the
    # first assembly when none is flagged (None-safe on a missing struct-assembly).
    chosen = next(
        (
            a
            for a in assemblies
            if (a.get("pdbx_struct_assembly") or {}).get("rcsb_candidate_assembly") == "Y"
        ),
        assemblies[0],
    )

    symmetry_blocks = [s for s in (chosen.get("rcsb_struct_symmetry") or []) if isinstance(s, dict)]
    if not symmetry_blocks:
        return {}

    # ANY block declaring a Homo oligomer corroborates a receptor homo-oligomer,
    # even when it is not the first (e.g. a later Local/Pseudo block).
    has_homo_symmetry = any(
        isinstance(s.get("oligomeric_state"), str)
        and s["oligomeric_state"].strip().lower().startswith("homo")
        for s in symmetry_blocks
    )

    first = symmetry_blocks[0]
    return {
        "oligomeric_state": first.get("oligomeric_state"),
        "stoichiometry": first.get("stoichiometry"),
        "kind": first.get("kind"),
        "type": first.get("type"),
        "has_homo_symmetry": has_homo_symmetry,
        "rcsb_candidate_assembly": (chosen.get("pdbx_struct_assembly") or {}).get(
            "rcsb_candidate_assembly"
        ),
        "modeled_polymer_monomer_count": (chosen.get("rcsb_assembly_info") or {}).get(
            "modeled_polymer_monomer_count"
        ),
    }


# ---------------------------------------------------------------------------
# Assembly-consistency advisory
# ---------------------------------------------------------------------------


_OLIGOMERIC_MER_RE = re.compile(r"(\d+)-mer")


def _parse_oligomeric_count(oligomeric_state: Any) -> int | None:
    """Parse the subunit count from an RCSB oligomeric_state string.

    The value space is ``"Monomer"`` / ``"Homo N-mer"`` / ``"Hetero N-mer"`` /
    ``None``.  ``"Monomer"`` maps to ``1``; an ``N-mer`` maps to ``N``; anything
    unparseable (including ``None``) returns ``None`` so the caller treats it as
    no signal rather than a contradiction.
    """
    if not isinstance(oligomeric_state, str):
        return None
    text = oligomeric_state.strip()
    if text.lower() == "monomer":
        return 1
    match = _OLIGOMERIC_MER_RE.search(text)
    if match:
        return int(match.group(1))
    return None


def _reconcile_assembly_consistency(
    classification: str,
    assembly_info: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Compare the GPCR-centric classification against the biological assembly.

    The top-line classification counts GPCR-slug chains only, so it can disagree
    with the RCSB biological assembly in two clear ways.  This pure, None-safe
    check surfaces that contradiction as a parallel advisory; it never changes the
    classification.  Returns ``(consistency, alert_or_None)`` where *consistency*
    is ``{"agrees": bool, "note": str}`` (note empty when they agree) and the
    alert is appended only when a contradiction fires.

    Fires on exactly two contradictions, and is silent otherwise (the normal case,
    including monomer-with-monomer-assembly and an absent assembly):

    - ``MONOMER`` while the biological assembly is a higher-order complex
      (oligomeric_state parses to N >= 2) -- the receptor may be one chain of a
      larger hetero-complex the GPCR-only count cannot see.
    - ``HOMOMER`` while NO symmetry block of the biological assembly corroborates a
      receptor homo-oligomer (no ``Homo N-mer`` block at all) -- the two same-slug
      chains may be crystallographic copies rather than a biological homodimer.
      Reading every block (not just the first) keeps a real homodimer whose Homo
      block is recorded later (Local/Pseudo) from false-firing.
    """
    state = assembly_info.get("oligomeric_state") if isinstance(assembly_info, dict) else None
    if not isinstance(state, str) or not state.strip():
        return {"agrees": True, "note": ""}, None

    count = _parse_oligomeric_count(state)
    stoich = assembly_info.get("stoichiometry")
    has_homo_symmetry = bool(assembly_info.get("has_homo_symmetry"))

    # MONOMER vs higher-order biological assembly (e.g. a 2:2:2 hetero-complex
    # whose single GPCR chain makes the GPCR-only count read MONOMER).
    if classification == OLIGOMER_MONOMER and count is not None and count >= 2:
        note = (
            f"GPCR-centric MONOMER but RCSB biological assembly is a higher-order "
            f"complex ({state}, {stoich}); verify the oligomer label."
        )
        return (
            {"agrees": False, "note": note},
            {
                "type": ALERT_ASSEMBLY_MISMATCH,
                "message": f"[{ALERT_ASSEMBLY_MISMATCH}] at 'oligomer_analysis': {note}",
            },
        )

    # HOMOMER vs a biological assembly with no Homo symmetry block at all: the
    # same-slug chains may be crystallographic copies, not a dimer. Checking every
    # block keeps a real homodimer whose Homo block is later (Local/Pseudo) silent.
    if classification == OLIGOMER_HOMOMER and not has_homo_symmetry:
        note = (
            f"two same-slug chains may be crystallographic copies, not a biological "
            f"homodimer ({state}, {stoich}); confirm."
        )
        return (
            {"agrees": False, "note": note},
            {
                "type": ALERT_ASSEMBLY_MISMATCH,
                "message": f"[{ALERT_ASSEMBLY_MISMATCH}] at 'oligomer_analysis': {note}",
            },
        )

    return {"agrees": True, "note": ""}, None


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
    coupling_chain: str | None = None,
) -> dict[str, Any]:
    """Suggest a primary protomer chain using the rank framework.

    Rank 0: Geometric G-protein coupling protomer (the detect stage measured which
        protomer the G-alpha engages -- an objective fact that beats the AI guess).
    Rank 1: G-protein bound (AI's chain if in roster and G-protein present).
    Rank 2: Exclusive ligand-binding chain.
    Rank 3: Best 7TM completeness.
    Rank 4: Longest sequence OR valid AI choice.

    A homomer keeps its selected chain but is relabelled rank 0 with a "Homomer"
    context prefix (the protomers are identical, so the choice is informational).
    """
    if not gpcr_roster:
        return {"chain_id": None, "reason": "No GPCR chains found", "rank_used": None}

    primary: str | None = None
    reason = ""
    rank: int | None = None

    # Rank 0: geometric G-protein coupling. Only one protomer of an obligate dimer
    # couples the G protein, and in a heterodimer it is often NOT the agonist-binding
    # one (GABA-B: GABBR1 binds, GABBR2 couples). The detect stage reads the coupling
    # protomer from the G-alpha interface in the coordinates; that measured fact wins
    # over the AI's chain choice. (Computed upstream, no GPCRdb per-structure data.)
    if coupling_chain and coupling_chain in gpcr_roster:
        primary = coupling_chain
        reason = f"Rank 0: G-protein coupling protomer (structure geometry) on Chain {primary}"
        rank = 0

    # Rank 1: G-protein bound
    has_gprotein = False
    if signaling_partners:
        if "g_protein" in signaling_partners:
            has_gprotein = True
        else:
            sp_str = str(signaling_partners).lower()
            if any(tag in sp_str for tag in ("gnai", "gnas", "gnaq", "gnao")):
                has_gprotein = True

    if not primary and has_gprotein and ai_chain and ai_chain in gpcr_roster:
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

    # A homomer's primary is informational (the protomers are identical), so relabel
    # it rank 0 with a "Homomer" context prefix regardless of which rank actually
    # selected the chain. (A coupling-driven rank-0 keeps its reason, now prefixed.)
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

    Alert messages follow
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

    The return dict includes ``original_chain_id`` and
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
# GPCR protomer mis-filed as auxiliary protein
# ---------------------------------------------------------------------------


def _is_crystallization_fusion_aux(aux: dict[str, Any]) -> bool:
    """Is this ``auxiliary_proteins`` entry a crystallization fusion, not a protomer?

    A crystallization fusion (BRIL / cytochrome b562, T4 lysozyme, GFP, glycogen
    synthase, …) is an engineering aid spliced into the receptor chain, not a
    separate biological protomer. The model marks it either by typing it
    "Fusion protein" or by naming a known fusion partner. Either signal keeps the
    entry from being evicted when it sits on a chain that is also a real protomer.
    """
    type_value = (aux.get("type") or {}).get("value") if isinstance(aux.get("type"), dict) else None
    if isinstance(type_value, str) and "fusion" in type_value.lower():
        return True
    name_lower = (aux.get("name") or "").lower()
    return any(kw in name_lower for kw in CRYSTALLIZATION_FUSION_KEYWORDS)


def reconcile_gpcr_in_auxiliary(
    best_run_data: dict[str, Any],
    validated_roster: dict[str, dict[str, Any]],
    classification: str,
    alerts: list[dict[str, str]],
) -> None:
    """Evict a GPCR protomer mis-filed under ``auxiliary_proteins`` (mutates in-place).

    A Class C receptor is an obligate dimer; its partner protomer is a real GPCR
    chain. When the model files that partner under ``auxiliary_proteins`` (often as
    type "Other"), it pollutes ``other_aux_proteins.csv``. The partner is already
    recorded independently in the structures.csv Partner columns (via
    :func:`resolve_partner_protomer` over ``all_gpcr_chains``), so removing the
    auxiliary entry loses no data.

    Two guards keep this from deleting legitimate auxiliary entries:

    * *validated_roster* is the transmembrane-gated roster (a chain is a protomer
      only if its UniProt annotation carries enough transmembrane helices). A
      soluble partner mis-mapped to a receptor slug — an E3 ligase, an R-spondin
      ectodomain — is not in this roster, so its chain never matches and it stays.
    * *classification* decides whether eviction is even possible. A single-protomer
      structure has no second protomer to recover, so anything sharing the receptor
      chain is a fusion or sub-domain and is always kept. Only a homo-/heteromer
      (two or more real protomers, e.g. a Class C dimer) can hide a mis-filed
      partner — and even then a crystallization fusion sitting on a protomer chain
      (typed "Fusion protein" or named BRIL / T4 lysozyme / GFP / …) is kept.

    An evicted entry yields a domain-language alert. An entry with an
    empty/garbled/missing chain_id is left untouched (fail-safe).
    """
    aux_list = best_run_data.get("auxiliary_proteins")
    if not isinstance(aux_list, list) or not validated_roster:
        return

    # A single-protomer structure cannot hide a mis-filed second protomer; every
    # auxiliary entry is a fusion or sub-domain of the lone receptor chain. Keep all.
    if classification == OLIGOMER_MONOMER:
        return

    kept: list[Any] = []
    for aux in aux_list:
        if not isinstance(aux, dict):
            kept.append(aux)
            continue
        aux_chains = _split_chain_ids(aux.get("chain_id"))
        roster_hits = sorted(aux_chains & set(validated_roster.keys()))
        # Keep the entry when its chain is not a validated protomer (nothing to
        # recover) or when it is a crystallization fusion sitting on a protomer
        # chain. The two keep-reasons are split so a future edit cannot silently
        # swap them into an evict.
        if not roster_hits:
            kept.append(aux)
            continue
        if _is_crystallization_fusion_aux(aux):
            kept.append(aux)
            continue
        name = aux.get("name") or "unknown"
        slug = validated_roster[roster_hits[0]].get("slug") or "unknown"
        chains_text = ", ".join(roster_hits)
        alerts.append(
            {
                "type": ALERT_PROTOMER_IN_AUXILIARY,
                "message": (
                    f"[{ALERT_PROTOMER_IN_AUXILIARY}] at 'auxiliary_proteins': "
                    f"auxiliary protein '{name}' (chain {chains_text}, slug {slug}) "
                    f"is a GPCR protomer recorded as the dimer partner; "
                    f"removed from auxiliary proteins."
                ),
            }
        )
    best_run_data["auxiliary_proteins"] = kept


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def analyze_oligomer(
    pdb_id: str,
    best_run_data: dict[str, Any],
    enriched_entry: dict[str, Any],
    coupling_chain: str | None = None,
    polymer_features_cache: PolymerFeaturesCacheLike | None = None,
) -> None:
    """Run oligomer analysis on *best_run_data* against *enriched_entry*.

    Writes ``best_run_data["oligomer_analysis"]`` in-place.
    May correct ``receptor_info.chain_id`` and ``uniprot_entry_name``
    when AI is objectively wrong (HALLUCINATION or 7TM_UPGRADE).

    *coupling_chain* is the detect stage's geometric G-protein-coupling protomer (or
    ``None``); when set it is the highest-priority primary-protomer choice.

    *polymer_features_cache*, when supplied, serves the transmembrane-helix
    annotations from disk and stores a successful fetch, so the 7TM count
    survives a transient RCSB outage. When the fetch still fails and no cached
    copy is available, the receptor count cannot be transmembrane-verified; the
    classification is then flagged low-confidence and routed (TM_DATA_UNAVAILABLE)
    rather than failing open into counting an unverified peptide/partner as a
    receptor protomer.
    """
    # 1. Build GPCR roster
    gpcr_roster = _build_gpcr_roster(enriched_entry)

    # 2. Scan all GPCR chains for 7TM. A cache (when supplied) lets the fetch
    # survive a transient outage; a genuine fetch failure is detected so the
    # count is not silently failed open below.
    tm_roster: dict[str, dict[str, Any]] = {}
    graphql_entry: dict[str, Any] | None = None
    tm_fetch_failed = False
    if gpcr_roster:
        tm_roster, graphql_entry = scan_all_chains_7tm(
            pdb_id, set(gpcr_roster.keys()), cache=polymer_features_cache
        )
        # The transmembrane gate could not run: there are GPCR-slug chains to
        # verify but the feature fetch returned nothing. Distinguish this from a
        # structure whose chains genuinely carry no TM annotation by re-probing
        # only when a cache is wired (the path that owns the fetch); without a
        # cache the legacy behavior is preserved byte-for-byte.
        if polymer_features_cache is not None and graphql_entry is None:
            tm_fetch_failed = True

    # 3. Refine fusion slugs using per-UniProt TM features
    _refine_fusion_slugs(gpcr_roster, enriched_entry, graphql_entry)

    # 3b. Validated protomer roster: a chain whose UniProt annotation is not 7TM
    # (a single-pass partner, a soluble ligand, etc. mis-mapped to a GPCR slug)
    # is not a protomer for classification or the missed-protomer check. Gate on
    # the annotated TM count so a truncated-but-real GPCR (few resolved TMs, full
    # annotation) is kept. Fall back to the full roster if the 7TM scan produced
    # nothing (e.g. no GraphQL) so missing data never over-prunes.
    classify_roster = {
        chain: info
        for chain, info in gpcr_roster.items()
        if (tm_roster.get(chain) or {}).get("total_tms", 0) >= GPCR_MIN_ANNOTATED_TM
    }
    if not classify_roster:
        classify_roster = dict(gpcr_roster)

    # 4. Classify (after refinement so slugs are correct)
    unique_slugs = {info["slug"] for info in classify_roster.values()}

    if len(classify_roster) == 0:
        classification = OLIGOMER_NO_GPCR
    elif len(classify_roster) == 1:
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
        classify_roster,
        tm_roster,
        classification,
        ai_chain,
        signaling_partners,
        ligands_data,
        coupling_chain=coupling_chain,
    )

    # 6. Alerts — use the validated roster so non-7TM partners don't trigger a
    # false MISSED_PROTOMER / inflate the heteromer chain list.
    alerts = _generate_alerts(
        classify_roster,
        classification,
        ai_chain,
        best_run_data,
    )

    # A GPCR-annotation tool that finds NO GPCR protomer must alarm the curator, not
    # pass silently: the empty roster can be a genuine non-GPCR structure OR an
    # unresolved/missing UniProt mapping (RCSB exposed no slug). Promoted to a gating
    # warning by inject_oligomer_alerts so one-click accept is disabled.
    if classification == OLIGOMER_NO_GPCR:
        alerts.append(
            {
                "type": ALERT_NO_GPCR,
                "message": (
                    f"[{ALERT_NO_GPCR}] at 'receptor_info': no GPCR protomer was found "
                    f"in the structure (the roster is empty). This may be a non-GPCR "
                    f"structure or an unresolved/missing UniProt-to-GPCRdb mapping; "
                    f"confirm the receptor identity manually."
                ),
            }
        )

    # When the transmembrane fetch failed, every GPCR chain falls back to UNKNOWN
    # status -- but that is a data gap, NOT evidence the chains are not GPCRs.
    # Suppress the "are you sure these are GPCRs?" accusation in that case;
    # ALERT_TM_DATA_UNAVAILABLE below is the only correct signal.
    unknown_chains = [c for c in all_gpcr_chains if c["7tm_status"] == TM_STATUS_UNKNOWN]
    if unknown_chains and not tm_fetch_failed:
        slugs = [f"Chain {c['chain_id']} ({c['slug']})" for c in unknown_chains]
        alerts.append(
            {
                "type": ALERT_SUSPICIOUS_7TM,
                "message": (
                    f"[{ALERT_SUSPICIOUS_7TM}] at 'oligomer_analysis': "
                    f"{', '.join(slugs)}: NO transmembrane helices detected. "
                    "Are you sure these are GPCRs? (e.g. they might be large soluble ligands, antibodies). "
                    "If NOT, notify authors to add their prefixes to GPCR_SLUG_NEGATIVE_PREFIXES in config.py."
                ),
            }
        )

    # The transmembrane gate could not run (feature fetch failed, no cached copy),
    # so the classification used the unfiltered roster: a peptide ligand or
    # single-pass partner carrying a GPCR slug may have been counted as a receptor
    # protomer. Rather than let that pass silently, route to a curator. Promoted to
    # a gating warning by inject_oligomer_alerts.
    if tm_fetch_failed:
        alerts.append(
            {
                "type": ALERT_TM_DATA_UNAVAILABLE,
                "message": (
                    f"[{ALERT_TM_DATA_UNAVAILABLE}] at 'receptor_info': transmembrane-helix "
                    f"data could not be fetched, so the receptor-chain count is unverified "
                    f"and the oligomer classification ({classification}) is low-confidence. "
                    f"A peptide ligand or partner chain may be miscounted as a receptor; "
                    f"confirm the receptor chains manually."
                ),
            }
        )

    # 6b. Evict a GPCR protomer mis-filed under auxiliary_proteins (the obligate
    # dimer partner of a Class C receptor). Tested against the transmembrane-gated
    # validated roster so a soluble partner mis-mapped to a receptor slug (an E3
    # ligase, an R-spondin ectodomain) is never deleted, and only a homo-/heteromer
    # can lose an entry — a single-protomer structure keeps every entry, and a
    # crystallization fusion on a protomer chain is kept. The recovered partner is
    # already in the structures.csv Partner columns via resolve_partner_protomer,
    # so eviction loses no data; an alert flags the move for review.
    reconcile_gpcr_in_auxiliary(best_run_data, classify_roster, classification, alerts)

    # 7. Smart override: correct chain_id when AI is objectively wrong
    override_info = _apply_chain_override(
        receptor_info,
        ai_chain,
        suggestion,
        gpcr_roster,
        tm_roster,
        alerts,
    )

    # 8. label_asym_id map + small-molecule instance index
    label_map = _build_label_asym_id_map(enriched_entry)
    nonpolymer_instance_index = build_nonpolymer_instance_index(enriched_entry)

    # 8b. Flag annotated ligands modelled in more than one copy, so a curator can
    # check whether the copies sit at distinct sites or play distinct roles --
    # something a single annotation row cannot carry.  Only components the model
    # actually annotated as ligands are flagged; repeated glycosylation or ions
    # the model did not call out stay silent.
    multi_copy = find_multi_copy_components(nonpolymer_instance_index)
    annotated_comp_ids = {
        (lig.get("chem_comp_id") or "").strip() for lig in ligands_data if isinstance(lig, dict)
    }
    for comp_id in sorted(multi_copy):
        if comp_id not in annotated_comp_ids:
            continue
        labels = ", ".join(rec["label_asym_id"] for rec in nonpolymer_instance_index[comp_id])
        alerts.append(
            {
                "type": ALERT_MULTI_COPY_LIGAND,
                "message": (
                    f"[{ALERT_MULTI_COPY_LIGAND}] at 'ligands[{comp_id}]': modelled in "
                    f"{multi_copy[comp_id]} copies (instances {labels}); one annotation row "
                    f"may hide copies at distinct sites or with distinct roles. "
                    f"Human review recommended."
                ),
            }
        )

    # 9. Assembly cross-check (informational only)
    assembly_info = _get_assembly_cross_check(enriched_entry)

    # 9b. Reconcile the GPCR-centric classification against the RCSB biological
    # assembly. The classification counts GPCR-slug chains only, so it can miss a
    # larger hetero-complex (MONOMER but the assembly is higher-order) or read a
    # homodimer into two crystallographic copies (HOMOMER but the assembly is a
    # monomer / hetero-complex). This surfaces that contradiction as a parallel
    # advisory + alert; it never changes the classification, and stays silent in
    # the overwhelming normal case (ordinary monomers are byte-identical to before).
    assembly_consistency, assembly_alert = _reconcile_assembly_consistency(
        classification,
        assembly_info,
    )
    if assembly_alert:
        alerts.append(assembly_alert)

    # 10. Write output. ``receptor_count`` and ``tm_data_available`` expose the
    # transmembrane-gated count and whether it could be verified, so the report
    # and any future receptor-level cross-check read one clean path.
    best_run_data["oligomer_analysis"] = {
        "classification": classification,
        "receptor_count": len(classify_roster),
        "tm_data_available": not tm_fetch_failed,
        "all_gpcr_chains": all_gpcr_chains,
        "primary_protomer_suggestion": suggestion,
        "assembly_cross_check": assembly_info,
        "assembly_consistency": assembly_consistency,
        "alerts": alerts,
        "chain_id_override": override_info,
        "label_asym_id_map": label_map,
        "nonpolymer_instance_index": nonpolymer_instance_index,
    }
