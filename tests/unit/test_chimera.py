"""Tests for G-protein chimera analysis (Epic 5).

Covers: is_g_alpha_description, calculate_match_score, get_chimera_analysis
(success, tie-breaker, no G-protein, sequence too short, no valid comparisons).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

from gpcr_tools.config import (
    CHIMERA_STATUS_NO_G_PROTEIN,
    CHIMERA_STATUS_NO_VALID_COMPARISONS,
    CHIMERA_STATUS_SUCCESS,
    CHIMERA_STATUS_TOO_SHORT,
    CHIMERA_TAIL_LENGTH,
)
from gpcr_tools.validator.cache import SequenceCache
from gpcr_tools.validator.chimera import (
    calculate_match_score,
    get_chimera_analysis,
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


def _make_enriched(
    *,
    desc: str = "G alpha subunit",
    sequence: str = "MDEFGHIJKLMNOPQRSTUVWXYZABCD",
    uniprots: list[dict[str, Any]] | None = None,
    has_galpha: bool = True,
) -> dict[str, Any]:
    """Build an enriched_entry with a polymer entity."""
    if not has_galpha:
        return {"polymer_entities": []}

    entity: dict[str, Any] = {
        "rcsb_polymer_entity": {"pdbx_description": desc},
        "entity_poly": {"pdbx_seq_one_letter_code_can": sequence},
    }
    if uniprots is not None:
        entity["uniprots"] = uniprots
    return {"polymer_entities": [entity]}


def _mock_sequence_fetcher(sequences: dict[str, str]) -> Any:
    """Return a side_effect function for patching get_sequence_from_uniprot."""

    def _fetch(accession: str, cache: Any) -> str | None:
        return sequences.get(accession)

    return _fetch


class TestGetChimeraAnalysis:
    def test_success(self, tmp_path: Path) -> None:
        cache = SequenceCache(tmp_path / "seq.json")
        enriched = _make_enriched(sequence="MDEFGHIJKLMNQYFL")
        # All candidates return the same tail "QYFL" -> perfect match
        mock_seqs = {
            acc: f"AAAAA{acc}QYFL"
            for acc in [
                "P63092",
                "P38405",
                "P63096",
                "P04899",
                "P08754",
                "P09471",
                "P19086",
                "P11488",
                "P19087",
                "A8MTJ3",
                "P50148",
                "P29992",
                "O95837",
                "P30679",
                "Q03113",
                "Q14344",
            ]
        }
        with patch(
            "gpcr_tools.validator.chimera.get_sequence_from_uniprot",
            side_effect=_mock_sequence_fetcher(mock_seqs),
        ):
            result = get_chimera_analysis("TEST", enriched, cache)

        assert result["status"] == CHIMERA_STATUS_SUCCESS
        assert result["score"] == CHIMERA_TAIL_LENGTH
        assert result["tail_seq"] == "QYFL"
        assert result["best_match"] is not None

    def test_no_g_protein(self, tmp_path: Path) -> None:
        cache = SequenceCache(tmp_path / "seq.json")
        enriched = _make_enriched(has_galpha=False)
        result = get_chimera_analysis("TEST", enriched, cache)
        assert result["status"] == CHIMERA_STATUS_NO_G_PROTEIN

    def test_sequence_too_short(self, tmp_path: Path) -> None:
        cache = SequenceCache(tmp_path / "seq.json")
        enriched = _make_enriched(sequence="AB")  # shorter than CHIMERA_TAIL_LENGTH
        result = get_chimera_analysis("TEST", enriched, cache)
        assert result["status"] == CHIMERA_STATUS_TOO_SHORT

    def test_no_valid_comparisons(self, tmp_path: Path) -> None:
        cache = SequenceCache(tmp_path / "seq.json")
        enriched = _make_enriched(sequence="MDEFGHIJ")
        # All fetches return None
        with patch(
            "gpcr_tools.validator.chimera.get_sequence_from_uniprot",
            return_value=None,
        ):
            result = get_chimera_analysis("TEST", enriched, cache)
        assert result["status"] == CHIMERA_STATUS_NO_VALID_COMPARISONS

    def test_tie_breaker_picks_family_leader(self, tmp_path: Path) -> None:
        """When multiple candidates tie, the canonical family leader wins."""
        cache = SequenceCache(tmp_path / "seq.json")
        enriched = _make_enriched(sequence="MDEFXYZW")
        # All candidates get the same tail -> all score equally
        mock_seqs = {
            acc: "AAAAAXYZW"
            for acc in [
                "P63092",
                "P38405",
                "P63096",
                "P04899",
                "P08754",
                "P09471",
                "P19086",
                "P11488",
                "P19087",
                "A8MTJ3",
                "P50148",
                "P29992",
                "O95837",
                "P30679",
                "Q03113",
                "Q14344",
            ]
        }
        with patch(
            "gpcr_tools.validator.chimera.get_sequence_from_uniprot",
            side_effect=_mock_sequence_fetcher(mock_seqs),
        ):
            result = get_chimera_analysis("TEST", enriched, cache)

        assert result["status"] == CHIMERA_STATUS_SUCCESS
        # canonical_best should be a family leader since all tails match
        assert result["can_best"] in (
            "gnas2_human",
            "gnai1_human",
            "gnaq_human",
            "gna13_human",
        )

    def test_null_pdbx_description(self, tmp_path: Path) -> None:
        """BL1: null pdbx_description must not crash."""
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
        """BL1: null entity_poly must not crash."""
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
        """BL1: null rcsb_polymer_entity must not crash."""
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
        # Should not crash; entity has no description -> no G-alpha found
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
                    "entity_poly": {"pdbx_seq_one_letter_code": "MDEFGHIJ"},
                }
            ]
        }
        with patch(
            "gpcr_tools.validator.chimera.get_sequence_from_uniprot",
            return_value=None,
        ):
            result = get_chimera_analysis("TEST", enriched, cache)
        # No comparisons possible, but sequence was found
        assert result["status"] == CHIMERA_STATUS_NO_VALID_COMPARISONS

    def test_result_keys(self, tmp_path: Path) -> None:
        """Verify all expected keys are present in the result dict."""
        cache = SequenceCache(tmp_path / "seq.json")
        result = get_chimera_analysis("TEST", {}, cache)
        assert "status" in result
        assert "best_match" in result
        assert "score" in result
        assert "candidates_checked" in result
        assert "error" in result
        assert "tail_seq" in result
