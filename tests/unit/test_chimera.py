"""Tests for G-protein alpha5 identity analysis.

Covers: is_g_alpha_description, calculate_match_score, and get_chimera_analysis
(resolved subtype, inseparable-set family-only, cross-family-only, low
confidence, sliding rescue of a displaced alpha5, and the no-G-protein /
too-short / no-comparison paths).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import requests

from gpcr_tools.config import (
    API_MAX_RETRIES,
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
)
from gpcr_tools.validator.cache import SequenceCache
from gpcr_tools.validator.chimera import (
    _resolve_subtype,
    calculate_match_score,
    get_chimera_analysis,
    get_sequence_from_uniprot,
    is_alpha5_mimetic_description,
    is_g_alpha_description,
)

# ===================================================================
# is_g_alpha_description
# ===================================================================


class TestIsGAlphaDescription:
    def test_standard_galpha(self) -> None:
        assert is_g_alpha_description("G alpha subunit") is True

    def test_g_dash_alpha(self) -> None:
        assert is_g_alpha_description("G-alpha protein") is True

    def test_g_protein_alpha(self) -> None:
        assert is_g_alpha_description("Guanine nucleotide-binding protein alpha subunit") is True

    def test_gq_family(self) -> None:
        assert is_g_alpha_description("Guanine nucleotide-binding protein Gq") is True

    def test_gs_family(self) -> None:
        assert is_g_alpha_description("G protein Gs subunit") is True

    def test_minig(self) -> None:
        assert is_g_alpha_description("miniGsq fusion") is True

    def test_engineered_g13(self) -> None:
        assert is_g_alpha_description("Engineered G13") is True

    def test_guanine_terminal_pattern(self) -> None:
        assert is_g_alpha_description("Guanine nucleotide-binding protein G(q)") is True

    def test_fusion_catch(self) -> None:
        assert is_g_alpha_description("Guanine nucleotide-binding protein subunit alpha") is True

    # Negative cases
    def test_receptor_excluded(self) -> None:
        assert is_g_alpha_description("Dopamine receptor D2") is False

    def test_antibody_excluded(self) -> None:
        assert is_g_alpha_description("Nanobody Nb35") is False

    def test_beta_subunit_excluded(self) -> None:
        assert is_g_alpha_description("G protein beta subunit") is False

    def test_gamma_subunit_excluded(self) -> None:
        assert is_g_alpha_description("G protein gamma subunit") is False

    def test_random_protein(self) -> None:
        assert is_g_alpha_description("Ubiquitin ligase") is False

    def test_empty_string(self) -> None:
        assert is_g_alpha_description("") is False

    def test_alpha_overrides_exclude(self) -> None:
        """If 'alpha' is present, exclude keywords are bypassed."""
        assert is_g_alpha_description("G-alpha receptor fusion") is True

    def test_bare_sequence_alpha5_mimetic_caught(self) -> None:
        """A G-alpha alpha5 C-terminal mimetic deposited under a bare-sequence
        name (no G-protein wording) is now recognised by its conserved motif."""
        assert is_g_alpha_description("ILENLKDVGLF peptide CT2") is True

    def test_gi_alpha5_mimetic_caught(self) -> None:
        assert is_g_alpha_description("IKENLKDCGLF peptide") is True

    def test_glp1_peptide_hormone_not_a_mimetic(self) -> None:
        # A genuine peptide-hormone agonist must not be mistaken for a G-alpha tail.
        assert is_g_alpha_description("Glucagon-like peptide-1") is False
        assert is_g_alpha_description("Glucagon") is False


class TestIsAlpha5MimeticDescription:
    """The standalone alpha5-mimetic recognizer anchors on the conserved G-alpha
    C-terminal motif, not on 'is a short peptide'."""

    def test_transducin_motif(self) -> None:
        assert is_alpha5_mimetic_description("ILENLKDVGLF peptide CT2") is True

    def test_gi_motif(self) -> None:
        assert is_alpha5_mimetic_description("IKENLKDCGLF") is True

    def test_genuine_peptide_hormone_not_matched(self) -> None:
        for name in ("Glucagon-like peptide-1", "Glucagon", "CXCL12 chemokine", "Substance P"):
            assert is_alpha5_mimetic_description(name) is False, name

    def test_small_molecule_not_matched(self) -> None:
        assert is_alpha5_mimetic_description("Adenosine") is False

    def test_empty_not_matched(self) -> None:
        assert is_alpha5_mimetic_description("") is False


# ===================================================================
# calculate_match_score
# ===================================================================


class TestCalculateMatchScore:
    def test_exact_match(self) -> None:
        assert calculate_match_score("ACDE", "ACDE") == 4

    def test_partial_match(self) -> None:
        assert calculate_match_score("ACDE", "ACDF") == 3

    def test_no_match(self) -> None:
        assert calculate_match_score("AAAA", "BBBB") == 0

    def test_empty_first(self) -> None:
        assert calculate_match_score("", "ACDE") == 0

    def test_empty_second(self) -> None:
        assert calculate_match_score("ACDE", "") == 0

    def test_different_lengths(self) -> None:
        assert calculate_match_score("AC", "ACDE") == 0

    def test_both_empty(self) -> None:
        assert calculate_match_score("", "") == 0

    def test_single_char_match(self) -> None:
        assert calculate_match_score("A", "A") == 1

    def test_single_char_mismatch(self) -> None:
        assert calculate_match_score("A", "B") == 0


# ===================================================================
# get_chimera_analysis
# ===================================================================

# The real transducin / Gi alpha5 tip. Its last residues (CGLF) are shared
# across Gi and the transducins, which is why a short window cannot tell them
# apart; the full 11-mer separates the transducin subgroup from Gi but still
# cannot pick one transducin over another.
TRANSDUCIN_A5 = "IKENLKDCGLF"
DISTINCT_A5 = "WWWWWWWWWWW"  # matches no real alpha5; used as the default ref tail

assert len(TRANSDUCIN_A5) == CHIMERA_A5_WINDOW
assert len(DISTINCT_A5) == CHIMERA_A5_WINDOW


def _make_enriched(
    *,
    desc: str = "G alpha subunit",
    sequence: str = "MMMMMMMMMM" + TRANSDUCIN_A5,
    uniprots: list[dict[str, Any]] | None = None,
    has_galpha: bool = True,
) -> dict[str, Any]:
    """Build an enriched_entry with a single polymer entity."""
    if not has_galpha:
        return {"polymer_entities": []}

    entity: dict[str, Any] = {
        "rcsb_polymer_entity": {"pdbx_description": desc},
        "entity_poly": {"pdbx_seq_one_letter_code_can": sequence},
    }
    if uniprots is not None:
        entity["uniprots"] = uniprots
    return {"polymer_entities": [entity]}


def _mock_refs(tail_by_slug: dict[str, str | None], default_tail: str = DISTINCT_A5) -> Any:
    """Return a get_sequence_from_uniprot stand-in.

    Each accession maps (via FULL_G_ALPHA_CANDIDATES) to a slug; the returned
    sequence ends with that slug's assigned alpha5 window, padded so it exceeds
    the window length. A slug whose entry is explicitly ``None`` simulates a
    fetch abstain (transient outage / absent accession) and returns ``None``;
    unlisted slugs fall back to *default_tail*.
    """

    def _fetch(accession: str, cache: Any) -> str | None:
        slug = FULL_G_ALPHA_CANDIDATES.get(accession)
        if slug is not None and slug in tail_by_slug:
            tail = tail_by_slug[slug]
            return None if tail is None else "GGGGG" + tail
        return "GGGGG" + default_tail

    return _fetch


class TestGetChimeraAnalysis:
    def test_resolved_subtype(self, tmp_path: Path) -> None:
        """A unique best match resolves to that subtype and its family."""
        cache = SequenceCache(tmp_path / "seq.json")
        target = "ACDEFGHIKLM"
        enriched = _make_enriched(sequence="MMMMMMMMMM" + target)
        with patch(
            "gpcr_tools.validator.chimera.get_sequence_from_uniprot",
            side_effect=_mock_refs({"gnas2_human": target}),
        ):
            result = get_chimera_analysis("TEST", enriched, cache)

        assert result["status"] == CHIMERA_STATUS_SUCCESS
        assert result["score"] == CHIMERA_A5_WINDOW
        assert result["subtype"] == "gnas2_human"
        assert result["subtype_resolution"] == CHIMERA_SUBTYPE_RESOLVED
        assert result["family"] == "Gs"
        assert result["family_confident"] is True
        assert result["candidate_set"] == ["gnas2_human"]
        assert result["a5_tail"] == target

    def test_transducin_alpha5_resolves_family_not_subtype(self, tmp_path: Path) -> None:
        """The transducin alpha5 is shared by gnat1/2/3.

        The family must come back as Gi/o, the subtype must NOT be forced to one
        member, and the indistinguishable set is reported for human review.
        """
        cache = SequenceCache(tmp_path / "seq.json")
        enriched = _make_enriched(sequence="MMMMMMMMMM" + TRANSDUCIN_A5)
        tails = {
            "gnat1_human": TRANSDUCIN_A5,
            "gnat2_human": TRANSDUCIN_A5,
            "gnat3_human": TRANSDUCIN_A5,
        }
        with patch(
            "gpcr_tools.validator.chimera.get_sequence_from_uniprot",
            side_effect=_mock_refs(tails),
        ):
            result = get_chimera_analysis("9IIX", enriched, cache)

        assert result["status"] == CHIMERA_STATUS_SUCCESS
        assert result["family"] == "Gi/o"
        assert result["family_confident"] is True
        assert result["subtype"] is None
        assert result["subtype_resolution"] == CHIMERA_SUBTYPE_INSEPARABLE_SET
        assert result["candidate_set"] == ["gnat1_human", "gnat2_human", "gnat3_human"]

    def test_partial_abstain_does_not_force_single_subtype(self, tmp_path: Path) -> None:
        """A transient fetch abstain that drops the co-members of an inseparable
        set must NOT leave a lone survivor that resolves to a confident subtype.

        The transducin structure ties gnat1/2/3. If gnat2/gnat3 abstain (fetch
        returned None this run) while gnat1 is fetched, gnat1 is the only scored
        member -- but the roster is known-incomplete, so the call stays at the
        family and routes the subtype to review rather than emitting a
        confidently-wrong gnat1.
        """
        cache = SequenceCache(tmp_path / "seq.json")
        enriched = _make_enriched(sequence="MMMMMMMMMM" + TRANSDUCIN_A5)
        tails: dict[str, str | None] = {
            "gnat1_human": TRANSDUCIN_A5,
            "gnat2_human": None,  # fetch abstained this run
            "gnat3_human": None,  # fetch abstained this run
        }
        with patch(
            "gpcr_tools.validator.chimera.get_sequence_from_uniprot",
            side_effect=_mock_refs(tails),
        ):
            result = get_chimera_analysis("TEST", enriched, cache)

        assert result["status"] == CHIMERA_STATUS_SUCCESS
        assert result["family"] == "Gi/o"
        assert result["subtype"] is None
        assert result["subtype_resolution"] == CHIMERA_SUBTYPE_INSEPARABLE_SET

    def test_genuine_resolve_unaffected_when_roster_complete(self, tmp_path: Path) -> None:
        """Guard against over-suppression: with NO abstains a unique winner still
        resolves to its subtype (the conservative downgrade only fires when a
        candidate reference could not be fetched)."""
        cache = SequenceCache(tmp_path / "seq.json")
        target = "ACDEFGHIKLM"
        enriched = _make_enriched(sequence="MMMMMMMMMM" + target)
        with patch(
            "gpcr_tools.validator.chimera.get_sequence_from_uniprot",
            side_effect=_mock_refs({"gnas2_human": target}),
        ):
            result = get_chimera_analysis("TEST", enriched, cache)

        assert result["status"] == CHIMERA_STATUS_SUCCESS
        assert result["subtype"] == "gnas2_human"
        assert result["subtype_resolution"] == CHIMERA_SUBTYPE_RESOLVED

    def test_unrelated_abstain_leaves_unique_winner_resolved(self, tmp_path: Path) -> None:
        """A partial outage that drops an UNRELATED reference must not downgrade a
        genuinely-unique winner. The structure uniquely matches gnas2; gnaz (no
        shared inseparable set) abstains. gnaz could never have tied gnas2, so the
        subtype stays resolved -- no false-review storm during a partial outage."""
        cache = SequenceCache(tmp_path / "seq.json")
        target = "ACDEFGHIKLM"
        enriched = _make_enriched(sequence="MMMMMMMMMM" + target)
        tails: dict[str, str | None] = {"gnas2_human": target, "gnaz_human": None}
        with patch(
            "gpcr_tools.validator.chimera.get_sequence_from_uniprot",
            side_effect=_mock_refs(tails),
        ):
            result = get_chimera_analysis("TEST", enriched, cache)

        assert result["status"] == CHIMERA_STATUS_SUCCESS
        assert result["subtype"] == "gnas2_human"
        assert result["subtype_resolution"] == CHIMERA_SUBTYPE_RESOLVED

    def test_cross_member_tie_is_family_only(self, tmp_path: Path) -> None:
        """A tie across same-family members that are not a defined inseparable
        set stops at the family with a family_only resolution."""
        cache = SequenceCache(tmp_path / "seq.json")
        enriched = _make_enriched(sequence="MMMMMMMMMM" + TRANSDUCIN_A5)
        tails = {"gnai1_human": TRANSDUCIN_A5, "gnai3_human": TRANSDUCIN_A5}
        with patch(
            "gpcr_tools.validator.chimera.get_sequence_from_uniprot",
            side_effect=_mock_refs(tails),
        ):
            result = get_chimera_analysis("TEST", enriched, cache)

        assert result["status"] == CHIMERA_STATUS_SUCCESS
        assert result["family"] == "Gi/o"
        assert result["subtype"] is None
        assert result["subtype_resolution"] == CHIMERA_SUBTYPE_FAMILY_ONLY
        assert result["candidate_set"] == ["gnai1_human", "gnai3_human"]

    def test_low_confidence_when_nothing_matches(self, tmp_path: Path) -> None:
        """A weak best score is reported as low confidence, not a forced call."""
        cache = SequenceCache(tmp_path / "seq.json")
        enriched = _make_enriched(sequence="A" * 25)  # matches no real alpha5
        with patch(
            "gpcr_tools.validator.chimera.get_sequence_from_uniprot",
            side_effect=_mock_refs({}),  # every ref gets DISTINCT_A5 (no 'A')
        ):
            result = get_chimera_analysis("TEST", enriched, cache)

        assert result["status"] == CHIMERA_STATUS_SUCCESS
        assert result["subtype"] is None
        assert result["subtype_resolution"] == CHIMERA_SUBTYPE_LOW_CONFIDENCE

    def test_sliding_rescues_displaced_alpha5(self, tmp_path: Path) -> None:
        """When the alpha5 is not at the C-terminus (fusion/extra residues), the
        sliding scan still locates it and resolves the subtype."""
        cache = SequenceCache(tmp_path / "seq.json")
        # alpha5 sits in the middle, followed by a non-matching C-terminal tag.
        enriched = _make_enriched(sequence="GGGGG" + TRANSDUCIN_A5 + "PPPPPPPPPP")
        with patch(
            "gpcr_tools.validator.chimera.get_sequence_from_uniprot",
            side_effect=_mock_refs({"gnao_human": TRANSDUCIN_A5}),
        ):
            result = get_chimera_analysis("TEST", enriched, cache)

        assert result["status"] == CHIMERA_STATUS_SUCCESS
        assert result["subtype"] == "gnao_human"
        assert result["subtype_resolution"] == CHIMERA_SUBTYPE_RESOLVED
        assert result["a5_tail"] == TRANSDUCIN_A5

    def test_no_g_protein(self, tmp_path: Path) -> None:
        cache = SequenceCache(tmp_path / "seq.json")
        enriched = _make_enriched(has_galpha=False)
        result = get_chimera_analysis("TEST", enriched, cache)
        assert result["status"] == CHIMERA_STATUS_NO_G_PROTEIN

    def test_sequence_too_short(self, tmp_path: Path) -> None:
        cache = SequenceCache(tmp_path / "seq.json")
        enriched = _make_enriched(sequence="AB")  # shorter than the window
        result = get_chimera_analysis("TEST", enriched, cache)
        assert result["status"] == CHIMERA_STATUS_TOO_SHORT

    def test_no_valid_comparisons(self, tmp_path: Path) -> None:
        cache = SequenceCache(tmp_path / "seq.json")
        enriched = _make_enriched(sequence="MMMMMMMMMM" + TRANSDUCIN_A5)
        with patch(
            "gpcr_tools.validator.chimera.get_sequence_from_uniprot",
            return_value=None,
        ):
            result = get_chimera_analysis("TEST", enriched, cache)
        assert result["status"] == CHIMERA_STATUS_NO_VALID_COMPARISONS

    def test_total_transient_outage_surfaces_transient_abstained(self, tmp_path: Path) -> None:
        # Every reference times out: no comparison is possible AND the gap is
        # transient, so transient_abstained is populated -- this is what drives
        # the detect stage to mark the output for a later re-run.
        cache = SequenceCache(tmp_path / "seq.json")
        enriched = _make_enriched(sequence="MMMMMMMMMM" + TRANSDUCIN_A5)
        with (
            patch("gpcr_tools.validator.chimera.requests.get", return_value=_resp(503)),
            patch("gpcr_tools.validator.chimera.time.sleep"),
        ):
            result = get_chimera_analysis("TEST", enriched, cache)
        assert result["status"] == CHIMERA_STATUS_NO_VALID_COMPARISONS
        assert result["transient_abstained"]  # non-empty: recoverable on re-run

    def test_all_404_references_not_transient_abstained(self, tmp_path: Path) -> None:
        # A definitive-404 roster is genuinely unresolvable, not a transient gap:
        # no comparison, but transient_abstained stays empty so the detect output
        # is NOT marked (it must not be re-run forever).
        cache = SequenceCache(tmp_path / "seq.json")
        enriched = _make_enriched(sequence="MMMMMMMMMM" + TRANSDUCIN_A5)
        with (
            patch("gpcr_tools.validator.chimera.requests.get", return_value=_resp(404)),
            patch("gpcr_tools.validator.chimera.time.sleep"),
        ):
            result = get_chimera_analysis("TEST", enriched, cache)
        assert result["status"] == CHIMERA_STATUS_NO_VALID_COMPARISONS
        assert result["transient_abstained"] == []

    def test_null_pdbx_description(self, tmp_path: Path) -> None:
        """A null pdbx_description must not crash."""
        cache = SequenceCache(tmp_path / "seq.json")
        enriched: dict[str, Any] = {
            "polymer_entities": [
                {
                    "rcsb_polymer_entity": {"pdbx_description": None},
                    "entity_poly": {"pdbx_seq_one_letter_code_can": "MDEF"},
                }
            ]
        }
        result = get_chimera_analysis("TEST", enriched, cache)
        assert result["status"] == CHIMERA_STATUS_NO_G_PROTEIN

    def test_null_entity_poly(self, tmp_path: Path) -> None:
        """A null entity_poly must not crash."""
        cache = SequenceCache(tmp_path / "seq.json")
        enriched: dict[str, Any] = {
            "polymer_entities": [
                {
                    "rcsb_polymer_entity": {"pdbx_description": "G alpha subunit"},
                    "entity_poly": None,
                }
            ]
        }
        result = get_chimera_analysis("TEST", enriched, cache)
        assert result["status"] == CHIMERA_STATUS_TOO_SHORT

    def test_null_rcsb_polymer_entity(self, tmp_path: Path) -> None:
        """A null rcsb_polymer_entity must not crash."""
        cache = SequenceCache(tmp_path / "seq.json")
        enriched: dict[str, Any] = {
            "polymer_entities": [
                {
                    "rcsb_polymer_entity": None,
                    "entity_poly": {"pdbx_seq_one_letter_code_can": "MDEF"},
                }
            ]
        }
        result = get_chimera_analysis("TEST", enriched, cache)
        assert result["status"] == CHIMERA_STATUS_NO_G_PROTEIN

    def test_empty_enriched(self, tmp_path: Path) -> None:
        cache = SequenceCache(tmp_path / "seq.json")
        result = get_chimera_analysis("TEST", {}, cache)
        assert result["status"] == CHIMERA_STATUS_NO_G_PROTEIN

    def test_fallback_to_pdbx_seq(self, tmp_path: Path) -> None:
        """Falls back to pdbx_seq_one_letter_code when _can is missing."""
        cache = SequenceCache(tmp_path / "seq.json")
        enriched: dict[str, Any] = {
            "polymer_entities": [
                {
                    "rcsb_polymer_entity": {"pdbx_description": "G alpha subunit"},
                    "entity_poly": {"pdbx_seq_one_letter_code": "MDEFGHIJKLM"},
                }
            ]
        }
        with patch(
            "gpcr_tools.validator.chimera.get_sequence_from_uniprot",
            return_value=None,
        ):
            result = get_chimera_analysis("TEST", enriched, cache)
        # The sequence was found via the fallback field, but no references
        # could be fetched, so no comparison was possible.
        assert result["status"] == CHIMERA_STATUS_NO_VALID_COMPARISONS

    def test_result_keys(self, tmp_path: Path) -> None:
        """Every documented key is present even on the early-exit paths."""
        cache = SequenceCache(tmp_path / "seq.json")
        result = get_chimera_analysis("TEST", {}, cache)
        for key in (
            "status",
            "family",
            "family_confident",
            "subtype",
            "subtype_resolution",
            "candidate_set",
            "score",
            "a5_window",
            "a5_tail",
            "candidates_checked",
            "backbone_family",
            "backbone_slug",
            "is_alpha5_graft",
            "error",
        ):
            assert key in result


class TestAlpha5Graft:
    def test_graft_detected_when_backbone_differs_from_alpha5(self, tmp_path: Path) -> None:
        # Scaffold is gnas2 (Gs) but the modelled alpha5 is a Gq tail = a graft.
        cache = SequenceCache(tmp_path / "seq.json")
        gq_tail = "LQMNLREYNLV"
        enriched = _make_enriched(
            sequence="MMMMMMMMMM" + gq_tail,
            uniprots=[{"rcsb_id": "P63092", "gpcrdb_entry_name_slug": "gnas2_human"}],
        )
        tails = {"gnaq_human": gq_tail, "gnas2_human": "ACDEFGHIKLM"}
        with patch(
            "gpcr_tools.validator.chimera.get_sequence_from_uniprot",
            side_effect=_mock_refs(tails),
        ):
            result = get_chimera_analysis("TEST", enriched, cache)
        assert result["family"] == "Gq/11"  # functional alpha5 wins
        assert result["backbone_slug"] == "gnas2_human"
        assert result["backbone_family"] == "Gs"  # scaffold
        assert result["is_alpha5_graft"] is True

    def test_no_graft_when_backbone_matches_alpha5(self, tmp_path: Path) -> None:
        cache = SequenceCache(tmp_path / "seq.json")
        gs_tail = "ACDEFGHIKLM"
        enriched = _make_enriched(
            sequence="MMMMMMMMMM" + gs_tail,
            uniprots=[{"rcsb_id": "P63092", "gpcrdb_entry_name_slug": "gnas2_human"}],
        )
        with patch(
            "gpcr_tools.validator.chimera.get_sequence_from_uniprot",
            side_effect=_mock_refs({"gnas2_human": gs_tail}),
        ):
            result = get_chimera_analysis("TEST", enriched, cache)
        assert result["family"] == "Gs"
        assert result["backbone_family"] == "Gs"
        assert result["is_alpha5_graft"] is False

    def test_no_backbone_slug_means_no_graft(self, tmp_path: Path) -> None:
        # No attached G-alpha slug -> backbone unknown -> never a graft.
        cache = SequenceCache(tmp_path / "seq.json")
        enriched = _make_enriched(sequence="MMMMMMMMMM" + "ACDEFGHIKLM")
        with patch(
            "gpcr_tools.validator.chimera.get_sequence_from_uniprot",
            side_effect=_mock_refs({"gnas2_human": "ACDEFGHIKLM"}),
        ):
            result = get_chimera_analysis("TEST", enriched, cache)
        assert result["backbone_slug"] is None
        assert result["is_alpha5_graft"] is False


# ===================================================================
# _resolve_subtype (roster-incomplete behaviour)
# ===================================================================


class TestResolveSubtype:
    def test_lone_winner_resolves_when_no_abstain(self) -> None:
        assert _resolve_subtype(["gnat1_human"], 11) == (
            "gnat1_human",
            CHIMERA_SUBTYPE_RESOLVED,
        )

    def test_lone_winner_not_promoted_when_tie_partner_abstained(self) -> None:
        # gnat2/gnat3 are co-members of gnat1's inseparable set; their abstain
        # could have hidden a tie, so the lone gnat1 survivor is downgraded to the
        # inseparable-set family review -- not a confident subtype. gnat1 is a
        # member of a known inseparable set, so the outcome is INSEPARABLE_SET.
        subtype, resolution = _resolve_subtype(
            ["gnat1_human"], 11, abstained=frozenset({"gnat2_human", "gnat3_human"})
        )
        assert subtype is None
        assert resolution == CHIMERA_SUBTYPE_INSEPARABLE_SET

    def test_lone_winner_resolves_when_unrelated_slug_abstained(self) -> None:
        # gnas2 shares no inseparable set with the abstained gnaz, so gnaz could
        # never have tied it: a genuinely-unique winner stays RESOLVED even when
        # an unrelated reference could not be fetched (no false-review storm).
        assert _resolve_subtype(["gnas2_human"], 11, abstained=frozenset({"gnaz_human"})) == (
            "gnas2_human",
            CHIMERA_SUBTYPE_RESOLVED,
        )

    def test_lone_inseparable_member_resolves_when_only_unrelated_abstained(self) -> None:
        # gnat1's tie-partners (gnat2/gnat3) are present; only an unrelated slug
        # abstained, so there is no hidden tie and gnat1 still resolves.
        assert _resolve_subtype(["gnat1_human"], 11, abstained=frozenset({"gnaq_human"})) == (
            "gnat1_human",
            CHIMERA_SUBTYPE_RESOLVED,
        )


# ===================================================================
# get_sequence_from_uniprot (resilience: 200 / 404 / transient)
# ===================================================================


def _resp(status_code: int, text: str = "") -> MagicMock:
    """Build a stand-in requests.Response with the given status and body."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    return resp


class TestGetSequenceFromUniprot:
    _ACC = "P63092"
    _FASTA = ">sp|P63092|GNAS_HUMAN\nMGCLGNSKTEDQRNEEKAQR"

    def test_200_caches_and_returns(self, tmp_path: Path) -> None:
        cache = SequenceCache(tmp_path / "seq.json")
        with patch(
            "gpcr_tools.validator.chimera.requests.get",
            return_value=_resp(200, self._FASTA),
        ) as mock_get:
            seq = get_sequence_from_uniprot(self._ACC, cache)
        assert seq == "MGCLGNSKTEDQRNEEKAQR"
        assert mock_get.call_count == 1
        assert cache.get(self._ACC) == "MGCLGNSKTEDQRNEEKAQR"

    def test_transient_5xx_retries_then_abstains_no_cache(self, tmp_path: Path) -> None:
        cache = SequenceCache(tmp_path / "seq.json")
        with (
            patch(
                "gpcr_tools.validator.chimera.requests.get",
                return_value=_resp(503),
            ) as mock_get,
            patch("gpcr_tools.validator.chimera.time.sleep"),
        ):
            seq = get_sequence_from_uniprot(self._ACC, cache)
        assert seq is None
        assert mock_get.call_count == API_MAX_RETRIES
        assert self._ACC not in cache

    def test_404_abstains_immediately_no_retry_no_cache(self, tmp_path: Path) -> None:
        cache = SequenceCache(tmp_path / "seq.json")
        with (
            patch(
                "gpcr_tools.validator.chimera.requests.get",
                return_value=_resp(404),
            ) as mock_get,
            patch("gpcr_tools.validator.chimera.time.sleep"),
        ):
            seq = get_sequence_from_uniprot(self._ACC, cache)
        assert seq is None
        assert mock_get.call_count == 1
        assert self._ACC not in cache

    def test_transient_then_success(self, tmp_path: Path) -> None:
        cache = SequenceCache(tmp_path / "seq.json")
        with (
            patch(
                "gpcr_tools.validator.chimera.requests.get",
                side_effect=[_resp(503), _resp(200, self._FASTA)],
            ) as mock_get,
            patch("gpcr_tools.validator.chimera.time.sleep"),
        ):
            seq = get_sequence_from_uniprot(self._ACC, cache)
        assert seq == "MGCLGNSKTEDQRNEEKAQR"
        assert mock_get.call_count == 2
        assert cache.get(self._ACC) == "MGCLGNSKTEDQRNEEKAQR"

    def test_network_error_retries_then_abstains(self, tmp_path: Path) -> None:
        cache = SequenceCache(tmp_path / "seq.json")
        with (
            patch(
                "gpcr_tools.validator.chimera.requests.get",
                side_effect=requests.RequestException("boom"),
            ) as mock_get,
            patch("gpcr_tools.validator.chimera.time.sleep"),
        ):
            seq = get_sequence_from_uniprot(self._ACC, cache)
        assert seq is None
        assert mock_get.call_count == API_MAX_RETRIES
        assert self._ACC not in cache

    def test_200_empty_body_returns_none_no_cache(self, tmp_path: Path) -> None:
        cache = SequenceCache(tmp_path / "seq.json")
        with patch(
            "gpcr_tools.validator.chimera.requests.get",
            return_value=_resp(200, ">sp|P63092|GNAS_HUMAN"),
        ) as mock_get:
            seq = get_sequence_from_uniprot(self._ACC, cache)
        assert seq is None
        assert mock_get.call_count == 1
        assert self._ACC not in cache

    def test_transient_failure_marks_accession_unavailable(self, tmp_path: Path) -> None:
        cache = SequenceCache(tmp_path / "seq.json")
        with (
            patch("gpcr_tools.validator.chimera.requests.get", return_value=_resp(503)),
            patch("gpcr_tools.validator.chimera.time.sleep"),
        ):
            get_sequence_from_uniprot(self._ACC, cache)
        assert cache.is_unavailable(self._ACC)  # remembered for this run
        assert self._ACC not in cache  # but never frozen as a cached fact

    def test_404_does_not_mark_unavailable(self, tmp_path: Path) -> None:
        # A 404 is a definitive verdict, not a transient outage, so it must not
        # join the per-run unavailable guard (which drives the detect "degraded"
        # marker and the don't-re-request short-circuit).
        cache = SequenceCache(tmp_path / "seq.json")
        with (
            patch("gpcr_tools.validator.chimera.requests.get", return_value=_resp(404)),
            patch("gpcr_tools.validator.chimera.time.sleep"),
        ):
            get_sequence_from_uniprot(self._ACC, cache)
        assert not cache.is_unavailable(self._ACC)

    def test_unavailable_accession_short_circuits_without_request(self, tmp_path: Path) -> None:
        # After a transient failure this run, a later call abstains immediately
        # instead of re-hitting the dead endpoint once per PDB across the batch.
        cache = SequenceCache(tmp_path / "seq.json")
        with (
            patch("gpcr_tools.validator.chimera.requests.get", return_value=_resp(503)) as mock_get,
            patch("gpcr_tools.validator.chimera.time.sleep"),
        ):
            get_sequence_from_uniprot(self._ACC, cache)
            count_after_first = mock_get.call_count
            seq = get_sequence_from_uniprot(self._ACC, cache)
        assert seq is None
        assert mock_get.call_count == count_after_first  # no further requests
