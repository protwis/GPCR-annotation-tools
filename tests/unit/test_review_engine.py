"""Tests for the review engine — pure logic, no interactive prompts.

Tests focus on controversy detection, auto-resolve, and significance checks.
"""

import pytest

from gpcr_tools import config
from gpcr_tools.csv_generator.review_engine import (
    _list_item_path,
    _resolve_list_key_field,
    get_verified_paths,
    has_downstream_controversy,
    has_gating_controversy,
    is_controversy_significant,
)


class TestListKeyFieldResolution:
    @pytest.mark.parametrize("segment,field", list(config.LIST_ITEM_KEY_FIELDS.items()))
    def test_resolves_from_shared_config(self, segment, field):
        # List-item review keys must come from the shared config map (not a
        # private hardcoded copy) so review paths line up with vote aggregation.
        assert _resolve_list_key_field(f"root.{segment}.child") == field

    def test_follows_config_changes(self, monkeypatch):
        # Proves resolution reads config rather than a hardcoded literal.
        monkeypatch.setattr(
            "gpcr_tools.csv_generator.review_engine.LIST_ITEM_KEY_FIELDS",
            {"widgets": "widget_id"},
        )
        assert _resolve_list_key_field("root.widgets.x") == "widget_id"
        assert _resolve_list_key_field("root.ligands.x") is None

    def test_none_for_unknown_path(self):
        assert _resolve_list_key_field("root.unknown.child") is None


class TestListItemPathMatchesAggregatorIdentity:
    """The review navigation path for a list item must equal the identity vote
    aggregation stored its controversy/flag under — otherwise the curator never
    sees contested keyless calls (protein/Apo ligands with chem_comp_id="None").
    """

    def test_keyless_ligand_path_matches_aggregator(self):
        # chem_comp_id="None" is the placeholder the schema injects for protein
        # and Apo ligands — the routine GPCR case, not an edge case.
        item = {"chem_comp_id": "None", "name": "GLP-1"}
        review_path = _list_item_path("ligands", item, "chem_comp_id", 0)
        agg_path = f"ligands[{config.list_item_identity(item, 'chem_comp_id', 0)}]"
        assert review_path == agg_path
        # The keyless name is SAFE-normalized (casefold + separator collapse), so
        # review navigation and aggregation share the same normalized identity.
        assert review_path == "ligands[__keyless__:glp 1]"

    def test_real_key_path_unchanged(self):
        item = {"chem_comp_id": "ATP", "name": "x"}
        assert _list_item_path("ligands", item, "chem_comp_id", 0) == "ligands[ATP]"

    def test_non_dict_item_uses_index(self):
        assert (
            _list_item_path("ligands[X].synonyms", "a-string", "chem_comp_id", 2)
            == "ligands[X].synonyms[2]"
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


class TestHasGatingControversy:
    """The one-click accept-all gate is driven by this helper. Minority-omission
    advisories (gating=False) stay in the controversy map for display but must
    not disable accept-all; near-tie / real disagreements still gate."""

    def _omission_advisory(self, path: str) -> dict:
        # Same shape voting.py emits for an entity the chosen run omitted.
        return {
            "path": path,
            "best_run_value": None,
            "majority_vote_value": {"chem_comp_id": "RET"},
            "all_votes": {"role": {"agonist": 2}},
            "needs_review": True,
            "gating": False,
        }

    def _near_tie(self, path: str) -> dict:
        # Same shape voting.py emits for a near-tie disagreement (no gating key).
        return {
            "path": path,
            "best_run_value": "agonist",
            "majority_vote_value": "antagonist",
            "all_votes": {"agonist": 5, "antagonist": 5},
            "needs_review": True,
            "vote_margin": 0,
        }

    def test_empty_map_does_not_gate(self):
        assert has_gating_controversy({}) is False

    def test_only_omission_advisories_do_not_gate(self):
        # A PDB whose sole controversy is a minority-omission advisory must keep
        # one-click accept-all enabled.
        controversies = {c["path"]: c for c in [self._omission_advisory("ligands[RET]")]}
        assert has_gating_controversy(controversies) is False

    def test_near_tie_still_gates(self):
        # A genuine near-tie disagreement disables accept-all exactly as before.
        controversies = {c["path"]: c for c in [self._near_tie("ligands[ADN].role.value")]}
        assert has_gating_controversy(controversies) is True

    def test_mixed_gates_on_the_near_tie(self):
        controversies = {
            c["path"]: c
            for c in [
                self._omission_advisory("ligands[RET]"),
                self._near_tie("ligands[ADN].role.value"),
            ]
        }
        # The advisory is still present (visible for review)...
        assert "ligands[RET]" in controversies
        # ...but the near-tie is what gates.
        assert has_gating_controversy(controversies) is True


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


class TestConfidenceStyle:
    def test_high_is_success(self):
        from gpcr_tools.csv_generator.review_engine import _confidence_style

        assert _confidence_style("High") == "success"

    def test_low_is_warning(self):
        from gpcr_tools.csv_generator.review_engine import _confidence_style

        assert _confidence_style("Low") == "warning"

    def test_medium_is_warning(self):
        from gpcr_tools.csv_generator.review_engine import _confidence_style

        assert _confidence_style("Medium") == "warning"
