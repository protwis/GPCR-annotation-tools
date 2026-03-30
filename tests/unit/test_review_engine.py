"""Tests for the review engine — pure logic, no interactive prompts.

Tests focus on controversy detection, auto-resolve, and significance checks.
"""

from gpcr_tools.csv_generator.review_engine import (
    get_verified_paths,
    has_downstream_controversy,
    is_controversy_significant,
)


class TestHasDownstreamControversy:
    def test_empty_prefix_with_controversies(self):
        controversies = {"receptor_info.chain_id": {}}
        assert has_downstream_controversy("", controversies) is True

    def test_empty_prefix_no_controversies(self):
        assert has_downstream_controversy("", {}) is False

    def test_exact_match(self):
        controversies = {"receptor_info.chain_id": {}}
        assert has_downstream_controversy("receptor_info.chain_id", controversies) is True

    def test_nested_match(self):
        controversies = {"receptor_info.chain_id": {}}
        assert has_downstream_controversy("receptor_info", controversies) is True

    def test_no_match(self):
        controversies = {"receptor_info.chain_id": {}}
        assert has_downstream_controversy("signaling_partners", controversies) is False

    def test_list_path(self):
        controversies = {"ligands[ADN].role.value": {}}
        assert has_downstream_controversy("ligands", controversies) is True
        assert has_downstream_controversy("ligands[ADN]", controversies) is True
        assert has_downstream_controversy("ligands[ZMA]", controversies) is False


class TestIsControversySignificant:
    def test_trivial_only(self):
        """Controversies in AUTO_RESOLVE_KEYS should be non-significant."""
        controversies = {"receptor_info.confidence": {}}
        validation_data = {"critical_warnings": [], "algo_conflicts": []}
        assert is_controversy_significant("receptor_info", controversies, validation_data) is False

    def test_significant_key(self):
        """Controversies in non-trivial keys should be significant."""
        controversies = {"receptor_info.chain_id": {}}
        validation_data = {"critical_warnings": [], "algo_conflicts": []}
        assert is_controversy_significant("receptor_info", controversies, validation_data) is True

    def test_validation_warning_makes_significant(self):
        """Even trivial keys become significant with validation warnings."""
        controversies = {"receptor_info.confidence": {}}
        validation_data = {
            "critical_warnings": ["Ghost Chain at 'receptor_info': 'Z' not in PDB Source."],
            "algo_conflicts": [],
        }
        assert is_controversy_significant("receptor_info", controversies, validation_data) is True

    def test_empty_path(self):
        """Empty path prefix with non-trivial controversies."""
        controversies = {"structure_info.method": {}}
        validation_data = {"critical_warnings": [], "algo_conflicts": []}
        assert is_controversy_significant("", controversies, validation_data) is True


class TestGetVerifiedPaths:
    def test_extracts_verified_fields(self):
        data = {
            "receptor_info": {
                "chain_id": "A",
                "uniprot_entry_name": "test_human",
                "_verified_fields": ["chain_id", "uniprot_entry_name"],
            },
            "structure_info": {"method": "ELECTRON MICROSCOPY"},
        }
        verified = get_verified_paths(data)
        assert "receptor_info.chain_id" in verified
        assert "receptor_info.uniprot_entry_name" in verified
        assert len(verified) == 2

    def test_no_verified_fields(self):
        data = {"structure_info": {"method": "ELECTRON MICROSCOPY"}}
        verified = get_verified_paths(data)
        assert len(verified) == 0

    def test_empty_data(self):
        assert get_verified_paths({}) == set()
