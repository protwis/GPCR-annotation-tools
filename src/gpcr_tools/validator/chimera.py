"""G-protein identity verification via alpha5-helix sequence matching.

The receptor-coupling determinant of a G-alpha subunit is its C-terminal alpha5
helix. This module compares the alpha5 window of the G-alpha entity found in the
PDB structure against reference sequences of known G-alpha proteins fetched from
UniProt, and reports the coupling family plus, where the sequence allows it, the
subtype.

Identity is taken from the alpha5 itself, including engineered chimeras: a
mini-G scaffold carrying a grafted alpha5 is reported by the family of that
grafted alpha5 (the coupling determinant), not the scaffold.

Several subtypes share an identical alpha5 (e.g. the transducins, or Gi1/Gi2)
and cannot be told apart by this window. In those cases the call stops at the
family and the subtype is routed to human review rather than forced to one
member.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

import requests

from gpcr_tools.config import (
    A5_INSEPARABLE_SUBTYPE_SETS,
    A5_SUBTYPE_FAMILY,
    API_MAX_RETRIES,
    CHIMERA_A5_ANCHOR_MIN_SCORE,
    CHIMERA_A5_WINDOW,
    CHIMERA_STATUS_NO_G_PROTEIN,
    CHIMERA_STATUS_NO_VALID_COMPARISONS,
    CHIMERA_STATUS_SUCCESS,
    CHIMERA_STATUS_TOO_SHORT,
    CHIMERA_SUBTYPE_FAMILY_ONLY,
    CHIMERA_SUBTYPE_INSEPARABLE_SET,
    CHIMERA_SUBTYPE_LOW_CONFIDENCE,
    CHIMERA_SUBTYPE_RESOLVED,
    FULL_G_ALPHA_CANDIDATES,
    G_ALPHA_EXCLUDE_KEYWORDS,
    SLEEP_VALIDATION_RETRY,
    TIMEOUT_UNIPROT_FASTA,
    UNIPROT_REST_URL,
    VALIDATION_RETRY_BACKOFF_FACTOR,
)
from gpcr_tools.validator.cache import SequenceCache

logger = logging.getLogger(__name__)


# The conserved C-terminus of the G-alpha alpha5 helix -- the receptor-coupling
# determinant. A synthetic peptide reproducing this tail (an "alpha5 mimetic") is a
# G-protein-derived fragment, not a receptor agonist, but it is sometimes deposited
# under a BARE SEQUENCE name (e.g. "ILENLKDVGLF peptide CT2") that carries no
# G-protein wording for the name-based tiers to catch. These motifs are the highly
# conserved alpha5 C-terminal tips of the Gi/Gt (transducin) family; they are long
# and specific enough that a genuine peptide-hormone or small-molecule ligand name
# will not contain them, so anchoring on them does not create false positives.
_G_ALPHA_A5_C_TERMINAL_MOTIFS: tuple[str, ...] = ("kdvglf", "kdcglf")


def is_alpha5_mimetic_description(desc: str) -> bool:
    """Detect a G-alpha alpha5 C-terminal mimetic peptide from its description.

    Catches the at-risk class that ``is_g_alpha_description`` misses: a G-alpha
    C-terminal ("alpha5") mimetic peptide deposited under a bare-sequence name with
    no G-protein wording (e.g. "ILENLKDVGLF peptide CT2"). Recognition is anchored
    solely on the conserved alpha5 C-terminal motif itself; it is long and specific
    enough that a genuine small-molecule or peptide-hormone ligand name will not
    contain it, so anchoring on it does not create false positives.
    """
    desc = desc.lower()
    return any(motif in desc for motif in _G_ALPHA_A5_C_TERMINAL_MOTIFS)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def is_g_alpha_description(desc: str) -> bool:
    """Detect whether *desc* describes a G-alpha subunit.

    Uses a multi-tier heuristic covering standard names, family abbreviations,
    fusion constructs, common OCR errors, and a G-alpha alpha5 C-terminal mimetic
    peptide deposited under a bare-sequence name (its conserved alpha5 motif).
    """
    if is_alpha5_mimetic_description(desc):
        return True

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
    if cache.is_unavailable(accession):
        # Already failed transiently this run -- don't re-hit the dead endpoint
        # once per PDB. Abstains (returns None) exactly as a fresh failure would.
        return None

    url = f"{UNIPROT_REST_URL}/{accession}.fasta"
    for attempt in range(API_MAX_RETRIES):
        try:
            resp = requests.get(url, timeout=TIMEOUT_UNIPROT_FASTA)
            if resp.status_code == 200:
                lines = resp.text.strip().split("\n")
                if len(lines) > 1:
                    seq = "".join(lines[1:])
                    cache.set(accession, seq)
                    return seq
                # 200 with an empty/header-only body: a real but unusable answer.
                return None
            if resp.status_code == 404:
                # Definitive: the accession does not exist. Abstain (no useful
                # "empty sequence" to cache), do not retry.
                return None
            if resp.status_code == 400:
                # Bad Request: a malformed accession, not a transient outage. Abstain
                # immediately without retry (and without marking unavailable -- the
                # service is fine, the query was not).
                logger.warning(
                    "UniProt FASTA rejected query (HTTP 400) for '%s' (malformed?)", accession
                )
                return None
            # 5xx / 429 / other: service unavailable, not a verdict. Retry, then abstain.
            if attempt == API_MAX_RETRIES - 1:
                logger.warning(
                    "UniProt FASTA unavailable (HTTP %s) for '%s'",
                    resp.status_code,
                    accession,
                )
                cache.mark_unavailable(accession)
                return None
            time.sleep(SLEEP_VALIDATION_RETRY * VALIDATION_RETRY_BACKOFF_FACTOR**attempt)
        except (requests.RequestException, OSError) as exc:
            if attempt == API_MAX_RETRIES - 1:
                logger.warning("UniProt FASTA fetch error for '%s': %s", accession, exc)
                cache.mark_unavailable(accession)
                return None
            time.sleep(SLEEP_VALIDATION_RETRY * VALIDATION_RETRY_BACKOFF_FACTOR**attempt)

    return None


def calculate_match_score(seq1: str, seq2: str) -> int:
    """Count matching residues between two equal-length sequence windows.

    Returns 0 if either sequence is empty or lengths differ.
    """
    if not seq1 or not seq2 or len(seq1) != len(seq2):
        return 0
    return sum(1 for a, b in zip(seq1, seq2, strict=True) if a == b)


def _best_alpha5_match(struct_seq: str, ref_tail: str, *, slide: bool) -> tuple[int, str]:
    """Score *ref_tail* against the structure's alpha5 window.

    The structure's alpha5 is almost always its C-terminus, so that window is
    tried first. When ``slide`` is set (the C-terminal window scored poorly
    against every reference, suggesting a fusion/tag/truncation has displaced
    the alpha5) every window of the structure is scanned for the best match.

    Returns ``(score, window)`` where *window* is the structure segment scored.
    """
    w = len(ref_tail)
    c_terminal = struct_seq[-w:]
    best_score = calculate_match_score(c_terminal, ref_tail)
    best_window = c_terminal
    if slide:
        for i in range(len(struct_seq) - w + 1):
            window = struct_seq[i : i + w]
            score = calculate_match_score(window, ref_tail)
            if score > best_score:
                best_score, best_window = score, window
    return best_score, best_window


def _has_hidden_tie_partner(winners: set[str], abstained: set[str]) -> bool:
    """True when an abstained reference could have tied the (lone) *winners*.

    The only way a fetch abstain can manufacture a false-unique winner is by
    removing a co-member of an inseparable set: within such a set every
    reference's alpha5 tail is identical, so a dropped member would have tied at
    the same best score. An abstain of any slug OUTSIDE the winner's inseparable
    set(s) could never have hidden a tie (its alpha5 differs), so it is no reason
    to withhold a confident subtype. This keeps the conservative downgrade scoped
    to the genuine hazard rather than firing on every partial outage.
    """
    if not abstained:
        return False
    return any(winners <= group and (group & abstained) for group in A5_INSEPARABLE_SUBTYPE_SETS)


def _resolve_subtype(
    winners: list[str],
    best_score: int,
    *,
    abstained: frozenset[str] = frozenset(),
) -> tuple[str | None, str]:
    """Map the set of equally-scoring slugs to (subtype, resolution).

    A single winner resolves to that subtype. A winner set contained in one of
    the inseparable subtype sets stops at the family. Anything else is reported
    family-only. A weak best score is low confidence regardless.

    *abstained* lists candidate references that could not be fetched this run. A
    lone winner is NOT promoted to a confident subtype when one of the abstained
    references is a co-member of the winner's inseparable set: the outage could
    have removed a tie-partner, making a survivor look unique. The call then stops
    at the family for review rather than emitting a confidently-wrong subtype.
    Abstains of unrelated references (whose alpha5 could never have tied) do not
    trigger this downgrade, so a genuinely-unique winner stays resolved even
    during a partial outage.
    """
    if best_score < CHIMERA_A5_ANCHOR_MIN_SCORE:
        return None, CHIMERA_SUBTYPE_LOW_CONFIDENCE
    unique = set(winners)
    if len(unique) == 1 and not _has_hidden_tie_partner(unique, set(abstained)):
        return winners[0], CHIMERA_SUBTYPE_RESOLVED
    if any(unique <= group for group in A5_INSEPARABLE_SUBTYPE_SETS):
        return None, CHIMERA_SUBTYPE_INSEPARABLE_SET
    return None, CHIMERA_SUBTYPE_FAMILY_ONLY


def _base_result() -> dict[str, Any]:
    """A result dict with every key defaulted, for the early-exit paths."""
    return {
        "status": CHIMERA_STATUS_NO_G_PROTEIN,
        "family": None,
        "family_confident": False,
        "subtype": None,
        "subtype_resolution": None,
        "candidate_set": [],
        "score": 0,
        "a5_window": CHIMERA_A5_WINDOW,
        "a5_tail": None,
        "candidates_checked": [],
        "backbone_family": None,
        "backbone_slug": None,
        "is_alpha5_graft": False,
        "transient_abstained": [],
        "error": None,
    }


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def get_chimera_analysis(
    pdb_id: str,
    enriched_entry: dict[str, Any],
    cache: SequenceCache,
) -> dict[str, Any]:
    """Identify the G-alpha coupling family (and subtype where resolvable).

    Returns a result dict with keys: ``status``, ``family``,
    ``family_confident``, ``subtype``, ``subtype_resolution``,
    ``candidate_set``, ``score``, ``a5_window``, ``a5_tail``,
    ``candidates_checked``, ``backbone_family``, ``backbone_slug``,
    ``is_alpha5_graft``, ``transient_abstained``, ``error``.
    """
    result = _base_result()
    w = CHIMERA_A5_WINDOW

    # 1. Find the G-alpha entity. None-safe access throughout.
    g_alpha_entity: dict[str, Any] | None = None
    for entity in enriched_entry.get("polymer_entities") or []:
        if not isinstance(entity, dict):
            continue
        desc = (entity.get("rcsb_polymer_entity") or {}).get("pdbx_description") or ""
        if is_g_alpha_description(desc):
            g_alpha_entity = entity
            break

    if g_alpha_entity is None:
        # status already defaults to no-G-protein in _base_result().
        return result

    # 2. Get the modelled sequence.
    entity_poly = g_alpha_entity.get("entity_poly") or {}
    struct_seq: str | None = entity_poly.get("pdbx_seq_one_letter_code_can")
    if not struct_seq:
        struct_seq = entity_poly.get("pdbx_seq_one_letter_code")

    if not struct_seq or len(struct_seq) < w:
        result["status"] = CHIMERA_STATUS_TOO_SHORT
        return result

    # 3. Candidates: the wild-type roster plus any UniProt entries the API
    #    attached to this entity.
    candidates: dict[str, str] = dict(FULL_G_ALPHA_CANDIDATES)
    for u in g_alpha_entity.get("uniprots") or []:
        if not isinstance(u, dict):
            continue
        rcsb_id = u.get("rcsb_id")
        slug = u.get("gpcrdb_entry_name_slug")
        if rcsb_id and slug:
            candidates[rcsb_id] = slug

    # 4. Fetch reference sequences once (cached across calls).
    ref_tails: dict[str, str] = {}
    abstained: list[str] = []
    for acc_id, slug in candidates.items():
        ref_seq = get_sequence_from_uniprot(acc_id, cache)
        if ref_seq is None:
            # Fetch abstained (transient outage or absent accession). Unlike a
            # fetched-but-too-short sequence, we have NO datum for this slug, so it
            # silently drops out of scoring -- which can turn an inseparable tie
            # into a lone "winner". Record it so resolution can stay conservative.
            abstained.append(slug)
            continue
        if len(ref_seq) >= w:
            ref_tails[slug] = ref_seq[-w:]

    # References that transiently failed THIS run (timeout/5xx, never a definitive
    # 404). Distinct from `abstained` above, which also counts 404s: only a
    # transient gap means a re-run could recover the datum, so only this set marks
    # the detect output incomplete (an all-404 roster is genuinely unresolvable,
    # not degraded, and must not trigger a perpetual re-run).
    transient_abstained = sorted(
        slug for acc_id, slug in candidates.items() if cache.is_unavailable(acc_id)
    )

    if not ref_tails:
        result["status"] = CHIMERA_STATUS_NO_VALID_COMPARISONS
        result["transient_abstained"] = transient_abstained
        return result

    # 5. Score the structure's alpha5 against each reference. Try the
    #    C-terminal window first; only if it matches nothing well do we pay for
    #    a full sliding scan to locate a displaced alpha5.
    def score_all(*, slide: bool) -> dict[str, tuple[int, str]]:
        return {
            slug: _best_alpha5_match(struct_seq, ref_tail, slide=slide)
            for slug, ref_tail in ref_tails.items()
        }

    scored = score_all(slide=False)
    best_score = max(score for score, _ in scored.values())
    if best_score < CHIMERA_A5_ANCHOR_MIN_SCORE:
        scored = score_all(slide=True)
        best_score = max(score for score, _ in scored.values())

    winners = sorted(slug for slug, (score, _) in scored.items() if score == best_score)
    a5_tail = scored[winners[0]][1]

    families = {A5_SUBTYPE_FAMILY[s] for s in winners if s in A5_SUBTYPE_FAMILY}
    family = next(iter(families)) if len(families) == 1 else None
    subtype, resolution = _resolve_subtype(winners, best_score, abstained=frozenset(abstained))

    # Backbone (scaffold) family from the entity's attached G-alpha slug. When it
    # differs from the alpha5-derived family this is an alpha5-graft chimera
    # (e.g. a mini-Gs scaffold carrying a grafted Gq alpha5). The alpha5
    # (functional) identity wins by convention; the backbone is informational, so
    # it can never corrupt `family` (computed above, independent of this block).
    # Detection is family-granular: an intra-family graft (e.g. a gnas2 scaffold
    # with a gnal alpha5, both Gs) is intentionally not flagged here -- that needs
    # the deferred full-length subtype refinement. The first attached G-alpha slug
    # wins; real entities carry one G-alpha annotation (a second uniprot, when
    # present, is the receptor of a fusion and is skipped by the family lookup).
    backbone_slug: str | None = None
    backbone_family: str | None = None
    for u in g_alpha_entity.get("uniprots") or []:
        if not isinstance(u, dict):
            continue
        slug = u.get("gpcrdb_entry_name_slug")
        if slug and slug in A5_SUBTYPE_FAMILY:
            backbone_slug = slug
            backbone_family = A5_SUBTYPE_FAMILY[slug]
            break
    is_graft = bool(backbone_family and family and backbone_family != family)

    result.update(
        status=CHIMERA_STATUS_SUCCESS,
        family=family,
        family_confident=len(families) == 1,
        subtype=subtype,
        subtype_resolution=resolution,
        candidate_set=winners,
        score=best_score,
        a5_tail=a5_tail,
        candidates_checked=list(scored.keys()),
        backbone_family=backbone_family,
        backbone_slug=backbone_slug,
        is_alpha5_graft=is_graft,
        transient_abstained=transient_abstained,
    )
    return result
