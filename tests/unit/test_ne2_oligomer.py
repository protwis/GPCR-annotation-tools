"""Tests for New Era Epic 2: Oligomer Analysis Integration.

Covers config constants update (NE-2.1), inject_oligomer_alerts (NE-2.2),
_should_highlight_oligomer and display_oligomer_analysis_panel (NE-2.3).
"""

from gpcr_tools.config import BLACKLISTED_KEYS
from gpcr_tools.csv_generator.ui import _should_highlight_oligomer
from gpcr_tools.csv_generator.validation_display import inject_oligomer_alerts

# ── NE-2.1: Config Constants ──────────────────────────────────────────


class TestBlacklistedKeysUpdate:
    def test_oligomer_analysis_blacklisted(self):
        assert "oligomer_analysis" in BLACKLISTED_KEYS

    def test_legacy_keys_removed(self):
        assert "heteromer_resolution" not in BLACKLISTED_KEYS
        assert "tm_completeness" not in BLACKLISTED_KEYS


# ── NE-2.2: inject_oligomer_alerts ────────────────────────────────────


class TestInjectOligomerAlerts:
    def test_empty_oligo_no_warnings(self):
        validation_data: dict = {}
        inject_oligomer_alerts({}, validation_data)
        assert validation_data.get("critical_warnings", []) == []

    def test_chain_override_applied(self):
        oligo = {
            "chain_id_override": {
                "applied": True,
                "original_chain_id": "G",
                "corrected_chain_id": "R",
                "trigger": "HALLUCINATION",
            },
            "alerts": [],
            "all_gpcr_chains": [],
        }
        validation_data: dict = {}
        inject_oligomer_alerts(oligo, validation_data)
        warnings = validation_data["critical_warnings"]
        assert len(warnings) == 1
        assert "CHAIN_ID CORRECTED" in warnings[0]
        assert "G -> R" in warnings[0]
        assert "HALLUCINATION" in warnings[0]

    def test_hallucination_alert(self):
        oligo = {
            "chain_id_override": {"applied": False},
            "alerts": [{"type": "HALLUCINATION", "message": "Chain G not in GPCR roster"}],
            "all_gpcr_chains": [],
        }
        validation_data: dict = {}
        inject_oligomer_alerts(oligo, validation_data)
        warnings = validation_data["critical_warnings"]
        assert len(warnings) == 1
        assert "[HALLUCINATION]" in warnings[0]
        assert "receptor_info" in warnings[0]

    def test_missed_protomer_alert(self):
        oligo = {
            "chain_id_override": {"applied": False},
            "alerts": [{"type": "MISSED_PROTOMER", "message": "Missed chain B"}],
            "all_gpcr_chains": [],
        }
        validation_data: dict = {}
        inject_oligomer_alerts(oligo, validation_data)
        warnings = validation_data["critical_warnings"]
        assert len(warnings) == 1
        assert "[MISSED_PROTOMER]" in warnings[0]

    def test_incomplete_7tm(self):
        oligo = {
            "chain_id_override": {"applied": False},
            "alerts": [],
            "all_gpcr_chains": [
                {"chain_id": "A", "7tm_status": "INCOMPLETE_7TM"},
            ],
        }
        validation_data: dict = {}
        inject_oligomer_alerts(oligo, validation_data)
        warnings = validation_data["critical_warnings"]
        assert len(warnings) == 1
        assert "INCOMPLETE 7TM" in warnings[0]

    def test_clean_oligo_no_warnings(self):
        """No alerts, no override, all COMPLETE → no warnings injected."""
        oligo = {
            "chain_id_override": {"applied": False},
            "alerts": [{"type": "CONFIRMED_OLIGOMER", "message": "All good"}],
            "all_gpcr_chains": [
                {"chain_id": "A", "7tm_status": "COMPLETE"},
            ],
        }
        validation_data: dict = {}
        inject_oligomer_alerts(oligo, validation_data)
        assert validation_data.get("critical_warnings", []) == []

    def test_preserves_existing_warnings(self):
        """New warnings are appended, not replacing existing ones."""
        oligo = {
            "chain_id_override": {"applied": False},
            "alerts": [{"type": "HALLUCINATION", "message": "bad chain"}],
            "all_gpcr_chains": [],
        }
        validation_data = {"critical_warnings": ["existing warning"]}
        inject_oligomer_alerts(oligo, validation_data)
        warnings = validation_data["critical_warnings"]
        assert len(warnings) == 2
        assert warnings[0] == "existing warning"
        assert "[HALLUCINATION]" in warnings[1]

    def test_multiple_alerts_combined(self):
        """Override + MISSED_PROTOMER + INCOMPLETE_7TM → 3 warnings."""
        oligo = {
            "chain_id_override": {
                "applied": True,
                "original_chain_id": "X",
                "corrected_chain_id": "A",
                "trigger": "7TM_UPGRADE",
            },
            "alerts": [
                {"type": "MISSED_PROTOMER", "message": "Missed B"},
            ],
            "all_gpcr_chains": [
                {"chain_id": "A", "7tm_status": "COMPLETE"},
                {"chain_id": "B", "7tm_status": "INCOMPLETE_7TM"},
            ],
        }
        validation_data: dict = {}
        inject_oligomer_alerts(oligo, validation_data)
        warnings = validation_data["critical_warnings"]
        assert len(warnings) == 3
        assert any("CHAIN_ID CORRECTED" in w for w in warnings)
        assert any("[MISSED_PROTOMER]" in w for w in warnings)
        assert any("INCOMPLETE 7TM" in w for w in warnings)


# ── NE-2.3: _should_highlight_oligomer ────────────────────────────────


class TestShouldHighlightOligomer:
    def test_empty_oligo(self):
        assert _should_highlight_oligomer({}, "") is False

    def test_chain_override_applied(self):
        oligo = {"chain_id_override": {"applied": True}, "alerts": [], "all_gpcr_chains": []}
        assert _should_highlight_oligomer(oligo, "A") is True

    def test_hallucination_alert(self):
        oligo = {
            "chain_id_override": {"applied": False},
            "alerts": [{"type": "HALLUCINATION"}],
            "all_gpcr_chains": [],
        }
        assert _should_highlight_oligomer(oligo, "A") is True

    def test_missed_protomer_alert(self):
        oligo = {
            "chain_id_override": {"applied": False},
            "alerts": [{"type": "MISSED_PROTOMER"}],
            "all_gpcr_chains": [],
        }
        assert _should_highlight_oligomer(oligo, "A") is True

    def test_homomer_classification(self):
        oligo = {
            "classification": "HOMOMER",
            "chain_id_override": {"applied": False},
            "alerts": [],
            "all_gpcr_chains": [],
        }
        assert _should_highlight_oligomer(oligo, "A") is True

    def test_heteromer_classification(self):
        oligo = {
            "classification": "HETEROMER",
            "chain_id_override": {"applied": False},
            "alerts": [],
            "all_gpcr_chains": [],
        }
        assert _should_highlight_oligomer(oligo, "A") is True

    def test_comma_in_chain(self):
        oligo = {
            "classification": "MONOMER",
            "chain_id_override": {"applied": False},
            "alerts": [],
            "all_gpcr_chains": [],
        }
        assert _should_highlight_oligomer(oligo, "A, B") is True

    def test_incomplete_7tm(self):
        oligo = {
            "classification": "MONOMER",
            "chain_id_override": {"applied": False},
            "alerts": [],
            "all_gpcr_chains": [{"7tm_status": "INCOMPLETE_7TM"}],
        }
        assert _should_highlight_oligomer(oligo, "A") is True

    def test_clean_monomer(self):
        """MONOMER, no alerts, all COMPLETE, single chain → no highlight."""
        oligo = {
            "classification": "MONOMER",
            "chain_id_override": {"applied": False},
            "alerts": [],
            "all_gpcr_chains": [{"7tm_status": "COMPLETE"}],
        }
        assert _should_highlight_oligomer(oligo, "A") is False


# ── NE-2.3: display_oligomer_analysis_panel (smoke tests) ─────────────


class TestDisplayOligomerPanel:
    def test_no_oligo_key_no_error(self):
        """Calling with data missing oligomer_analysis should not raise."""
        from gpcr_tools.csv_generator.ui import display_oligomer_analysis_panel

        display_oligomer_analysis_panel({"receptor_info": {"chain_id": "A"}})

    def test_full_oligo_data_no_error(self, sample_oligomer_data):
        """Calling with full oligomer fixture should render without error."""
        from gpcr_tools.csv_generator.ui import display_oligomer_analysis_panel

        display_oligomer_analysis_panel(sample_oligomer_data)
