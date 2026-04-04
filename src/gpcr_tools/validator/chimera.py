"""G-protein identity verification via C-terminal tail sequence matching.

Compares the C-terminal tail of the G-alpha entity found in the PDB structure
against reference sequences of known G-alpha proteins fetched from UniProt.

Blood Lesson 1 — None-safety:
    ``(entity.get("rcsb_polymer_entity") or {}).get("pdbx_description") or ""``
Blood Lesson 4 — Magic strings:
    All status strings are constants from ``config.py``.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import requests

from gpcr_tools.config import (
    CHIMERA_STATUS_NO_G_PROTEIN,
    CHIMERA_STATUS_NO_VALID_COMPARISONS,
    CHIMERA_STATUS_SUCCESS,
    CHIMERA_STATUS_TOO_SHORT,
    CHIMERA_TAIL_LENGTH,
    FAMILY_LEADERS,
    FULL_G_ALPHA_CANDIDATES,
    G_ALPHA_EXCLUDE_KEYWORDS,
)
from gpcr_tools.validator.cache import SequenceCache

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def is_g_alpha_description(desc: str) -> bool:
    """Detect whether *desc* describes a G-alpha subunit.

    Uses a multi-tier heuristic covering standard names, family abbreviations,
    fusion constructs, and common OCR errors.
    """
    desc = desc.lower()

    # Exclude non-G proteins and beta/gamma subunits
    if any(kw in desc for kw in G_ALPHA_EXCLUDE_KEYWORDS) and "alpha" not in desc:
        return False

    # 1. Standard G-alpha names
    if any(x in desc for x in ("g alpha", "g-alpha", "galpha", "g_alpha")):
        return True

    # 2. Explicitly "alpha" with G protein context
    if "alpha" in desc and any(
        x in desc for x in ("g protein", "guanine", "g-protein", "g subunit")
    ):
        return True

    # 3. "subunit a" (OCR error or abbreviation)
    if "subunit a" in desc and ("g protein" in desc or "guanine" in desc):
        return True

    # 4. Specific family name (e.g. "Gq", "Gs")
    if ("guanine" in desc or "g protein" in desc) and (
        re.search(r"\bg[sioq]\b", desc) or re.search(r"\bg1[123]\b", desc)
    ):
        return True

    # 5. MiniG patterns
    if "minig" in desc.replace("-", ""):
        return True
    if "engineered g13" in desc:
        return True

    # 6. "guanine nucleotide-binding protein g(x)" terminal pattern
    if re.search(r"guanine nucleotide-binding protein g\([a-z]\)$", desc.strip()):
        return True

    # 7. Fusion catch
    return "guanine nucleotide-binding protein" in desc and (
        "subunit alpha" in desc or "alpha subunit" in desc
    )


def get_sequence_from_uniprot(
    accession: str,
    cache: SequenceCache,
) -> str | None:
    """Fetch a UniProt FASTA sequence, using *cache* to avoid repeat downloads.

    Returns the sequence string or ``None`` on failure.
    """
    cached = cache.get(accession)
    if cached is not None:
        return cached

    try:
        resp = requests.get(
            f"https://rest.uniprot.org/uniprotkb/{accession}.fasta",
            timeout=10,
        )
        if resp.status_code == 200:
            lines = resp.text.strip().split("\n")
            if len(lines) > 1:
                seq = "".join(lines[1:])
                cache.set(accession, seq)
                return seq
    except (requests.RequestException, OSError) as exc:
        logger.warning("UniProt FASTA fetch error for '%s': %s", accession, exc)

    return None


def calculate_match_score(seq1: str, seq2: str) -> int:
    """Count matching residues between two equal-length sequence tails.

    Returns 0 if either sequence is empty or lengths differ.
    """
    if not seq1 or not seq2 or len(seq1) != len(seq2):
        return 0
    return sum(1 for a, b in zip(seq1, seq2, strict=True) if a == b)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def get_chimera_analysis(
    pdb_id: str,
    enriched_entry: dict[str, Any],
    cache: SequenceCache,
) -> dict[str, Any]:
    """Run G-protein tail-matching analysis on *enriched_entry*.

    Returns a result dict with keys: ``status``, ``best_match``, ``score``,
    ``candidates_checked``, ``error``, ``tail_seq``, ``can_best``,
    ``max_score_matches``.
    """
    result: dict[str, Any] = {
        "status": CHIMERA_STATUS_NO_G_PROTEIN,
        "best_match": None,
        "score": 0,
        "candidates_checked": [],
        "error": None,
        "tail_seq": None,
    }

    # 1. Find G-alpha entity
    g_alpha_entity: dict[str, Any] | None = None
    entities = enriched_entry.get("polymer_entities") or []

    for entity in entities:
        if not isinstance(entity, dict):
            continue
        # BL1: (entity.get("rcsb_polymer_entity") or {}).get("pdbx_description") or ""
        desc = (entity.get("rcsb_polymer_entity") or {}).get("pdbx_description") or ""
        if is_g_alpha_description(desc):
            g_alpha_entity = entity
            break

    if g_alpha_entity is None:
        result["status"] = CHIMERA_STATUS_NO_G_PROTEIN
        return result

    # 2. Get structure sequence
    # BL1: (entity.get("entity_poly") or {})
    entity_poly = g_alpha_entity.get("entity_poly") or {}
    struct_seq: str | None = entity_poly.get("pdbx_seq_one_letter_code_can")
    if not struct_seq:
        struct_seq = entity_poly.get("pdbx_seq_one_letter_code")

    if not struct_seq or len(struct_seq) < CHIMERA_TAIL_LENGTH:
        result["status"] = CHIMERA_STATUS_TOO_SHORT
        return result

    # 3. Prepare candidates — mutable working copy of immutable constant
    candidates: dict[str, str] = dict(FULL_G_ALPHA_CANDIDATES)  # explicit copy
    uniprots = g_alpha_entity.get("uniprots") or []
    for u in uniprots:
        if not isinstance(u, dict):
            continue
        rcsb_id = u.get("rcsb_id")
        slug = u.get("gpcrdb_entry_name_slug")
        if rcsb_id and slug:
            candidates[rcsb_id] = slug

    # 4. Run comparison
    struct_tail = struct_seq[-CHIMERA_TAIL_LENGTH:]
    scores: dict[str, int] = {}

    for acc_id, slug in candidates.items():
        ref_seq = get_sequence_from_uniprot(acc_id, cache)
        if ref_seq and len(ref_seq) >= CHIMERA_TAIL_LENGTH:
            ref_tail = ref_seq[-CHIMERA_TAIL_LENGTH:]
            score = calculate_match_score(struct_tail, ref_tail)
            scores[slug] = score

    if not scores:
        result["status"] = CHIMERA_STATUS_NO_VALID_COMPARISONS
        return result

    best_score = max(scores.values())
    max_score_matches: list[str] = [slug for slug, score in scores.items() if score == best_score]

    # Tie-breaker: canonical priority via family leaders
    dynamic_canonical_map: dict[str, str] = {}
    for leader_slug in FAMILY_LEADERS:
        leader_acc = next(
            (k for k, v in FULL_G_ALPHA_CANDIDATES.items() if v == leader_slug),
            None,
        )
        if leader_acc:
            ref_seq = get_sequence_from_uniprot(leader_acc, cache)
            if ref_seq and len(ref_seq) >= CHIMERA_TAIL_LENGTH:
                ref_tail = ref_seq[-CHIMERA_TAIL_LENGTH:]
                if ref_tail not in dynamic_canonical_map:
                    dynamic_canonical_map[ref_tail] = leader_slug

    canonical_best = dynamic_canonical_map.get(struct_tail, max_score_matches[0])

    result["status"] = CHIMERA_STATUS_SUCCESS
    result["best_match"] = canonical_best
    result["score"] = best_score
    result["tail_seq"] = struct_tail
    result["can_best"] = canonical_best
    result["max_score_matches"] = max_score_matches
    result["candidates_checked"] = list(scores.keys())

    return result
