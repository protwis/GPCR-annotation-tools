"""Tests for validation display and analysis logic."""

from gpcr_tools.csv_generator.validation_display import (
    analyze_validation_impact,
    canonicalize_path,
    display_detector_notes,
    extract_validation_entries,
    get_relevant_validation_warnings,
    warning_matches_block,
)


class TestDisplayDetectorNotes:
    def test_renders_notes(self, capsys):
        display_detector_notes({"detector_notes": ["BRIL fusion at the N-terminus"]})
        out = capsys.readouterr().out
        assert "BRIL fusion" in out
        assert "DETECTOR NOTES" in out

    def test_silent_when_empty(self, capsys):
        display_detector_notes({"detector_notes": []})
        display_detector_notes({})
        assert capsys.readouterr().out == ""


class TestLigandDetectorNotes:
    """site_ref surfacing and the incidental_candidate verdict ternary are logic, not aesthetics."""

    def test_site_ref_note(self):
        from gpcr_tools.csv_generator.ui import ligand_detector_notes

        assert ligand_detector_notes({"site_ref": "allosteric"}) == ["Site: allosteric"]

    def test_incidental_candidate_functional_verdict(self):
        from gpcr_tools.csv_generator.ui import ligand_detector_notes

        notes = ligand_detector_notes(
            {"pharmacological_role_check": {"is_functional_ligand": True}}
        )
        assert any("functional ligand" in n for n in notes)

    def test_incidental_candidate_incidental_verdict(self):
        from gpcr_tools.csv_generator.ui import ligand_detector_notes

        notes = ligand_detector_notes(
            {"pharmacological_role_check": {"is_functional_ligand": False}}
        )
        assert any("incidental / structural" in n for n in notes)

    def test_no_notes_for_plain_ligand(self):
        from gpcr_tools.csv_generator.ui import ligand_detector_notes

        assert ligand_detector_notes({"name": "Adenosine"}) == []


class TestDisplayPdbFooter:
    def test_shows_pdb_id(self, capsys):
        from gpcr_tools.csv_generator.ui import display_pdb_footer

        display_pdb_footer("6CMO")
        out = capsys.readouterr().out
        assert "6CMO" in out
        assert "PDB" in out


class TestOligomerAlertRendering:
    """The oligomer analysis panel must render each alert's type prefix exactly
    once. Current validator messages already carry their own "[TYPE]" prefix, so
    the renderer must not prepend it again; older recorded data has no prefix, so
    the renderer must add it. The rendering is idempotent either way.
    """

    def test_multi_copy_ligand_prefix_not_duplicated(self, capsys):
        from gpcr_tools.config import ALERT_MULTI_COPY_LIGAND
        from gpcr_tools.csv_generator.ui import display_oligomer_analysis_panel

        msg = (
            f"[{ALERT_MULTI_COPY_LIGAND}] at 'ligands[CLR]': modelled in 2 copies "
            f"(instances D, E); one annotation row may hide copies at distinct "
            f"sites or with distinct roles. Human review recommended."
        )
        main_data = {
            "oligomer_analysis": {
                "classification": "MONOMER",
                "alerts": [{"type": ALERT_MULTI_COPY_LIGAND, "message": msg}],
            }
        }
        display_oligomer_analysis_panel(main_data)
        out = capsys.readouterr().out
        # The type prefix must appear exactly once, never doubled.
        assert out.count(f"[{ALERT_MULTI_COPY_LIGAND}]") == 1
        assert f"[{ALERT_MULTI_COPY_LIGAND}] [{ALERT_MULTI_COPY_LIGAND}]" not in out

    def test_hallucination_prefix_not_duplicated(self, capsys):
        from gpcr_tools.config import ALERT_HALLUCINATION
        from gpcr_tools.csv_generator.ui import display_oligomer_analysis_panel

        msg = f"[{ALERT_HALLUCINATION}] at 'oligomer_analysis': chain Z not in source."
        main_data = {
            "oligomer_analysis": {
                "classification": "MONOMER",
                "alerts": [{"type": ALERT_HALLUCINATION, "message": msg}],
            }
        }
        display_oligomer_analysis_panel(main_data)
        out = capsys.readouterr().out
        assert out.count(f"[{ALERT_HALLUCINATION}]") == 1

    def test_bare_message_gets_single_prefix(self, capsys):
        # Older recorded data stored the bare description (no "[TYPE]" prefix);
        # the renderer must add it exactly once so the type stays visible.
        from gpcr_tools.config import ALERT_MISSED_PROTOMER
        from gpcr_tools.csv_generator.ui import display_oligomer_analysis_panel

        main_data = {
            "oligomer_analysis": {
                "classification": "MONOMER",
                "alerts": [
                    {"type": ALERT_MISSED_PROTOMER, "message": "GPCR roster has chains A, B."}
                ],
            }
        }
        display_oligomer_analysis_panel(main_data)
        out = capsys.readouterr().out
        assert out.count(f"[{ALERT_MISSED_PROTOMER}]") == 1


class TestDisplayCriticalWarningsSummary:
    """Critical validation findings must be surfaced on the Initial Summary so
    the curator sees *why* a PDB is gated without entering review mode.
    """

    def test_renders_critical_warnings(self, capsys):
        from gpcr_tools.csv_generator.ui import display_critical_warnings_summary

        rendered = display_critical_warnings_summary(
            {
                "critical_warnings": [
                    "APO_WITH_LIGANDS at 'ligands': structure is apo but ligands annotated.",
                    "CHIMERIC G-PROTEIN at 'signaling_partners': fusion detected.",
                ],
                "algo_conflicts": ["CONFLICT! AI: 'gnai1' vs Algo: 'gnas2'"],
            }
        )
        out = capsys.readouterr().out
        assert rendered is True
        assert "APO_WITH_LIGANDS" in out
        assert "CHIMERIC G-PROTEIN" in out
        assert "gnai1" in out
        assert "CRITICAL VALIDATION FINDINGS" in out

    def test_silent_when_no_findings(self, capsys):
        from gpcr_tools.csv_generator.ui import display_critical_warnings_summary

        assert display_critical_warnings_summary({}) is False
        assert (
            display_critical_warnings_summary({"critical_warnings": [], "algo_conflicts": []})
            is False
        )
        assert capsys.readouterr().out == ""


class TestCanonicalizePath:
    def test_dots_stripped(self):
        assert canonicalize_path(".receptor_info.chain_id") == "receptor_info.chain_id"

    def test_none(self):
        assert canonicalize_path(None) == ""

    def test_empty(self):
        assert canonicalize_path("") == ""

    def test_clean_path(self):
        assert canonicalize_path("receptor_info") == "receptor_info"


class TestGetRelevantValidationWarnings:
    def test_no_warnings(self):
        validation = {"critical_warnings": [], "algo_conflicts": []}
        assert get_relevant_validation_warnings("receptor_info", validation) == []

    def test_algo_conflict_for_signaling(self):
        validation = {
            "critical_warnings": [],
            "algo_conflicts": ["CONFLICT! AI: 'gnai1' vs Algo: 'gnas2'"],
        }
        result = get_relevant_validation_warnings("signaling_partners", validation)
        assert len(result) == 1

    def test_critical_warning_matching_path(self):
        validation = {
            "critical_warnings": ["Ghost Chain at 'receptor_info': 'Z' not in PDB Source."],
            "algo_conflicts": [],
        }
        result = get_relevant_validation_warnings("receptor_info", validation)
        assert len(result) == 1


class TestWarningMatchesBlock:
    def test_exact_path_match(self):
        entry = {"path": "receptor_info", "is_hallucination": False, "text": "test"}
        assert warning_matches_block(entry, "receptor_info") is True

    def test_nested_path_match(self):
        entry = {"path": "receptor_info.chain_id", "is_hallucination": False, "text": "test"}
        assert warning_matches_block(entry, "receptor_info") is True

    def test_no_match(self):
        entry = {"path": "structure_info.method", "is_hallucination": False, "text": "test"}
        assert warning_matches_block(entry, "receptor_info") is False

    def test_hallucination_text_match(self):
        entry = {
            "path": None,
            "is_hallucination": True,
            "text": "HALLUCINATION ALERT for signaling_partners g-protein",
        }
        assert warning_matches_block(entry, "signaling_partners") is True


class TestExtractValidationEntries:
    def test_empty(self):
        assert extract_validation_entries(None) == []
        assert extract_validation_entries({}) == []

    def test_handles_null_buckets(self):
        validation = {"critical_warnings": None, "algo_conflicts": None}
        assert extract_validation_entries(validation) == []

    def test_extracts_entries(self):
        validation = {
            "critical_warnings": ["Ghost Chain at 'receptor_info': 'Z' not in PDB Source."],
            "algo_conflicts": ["CONFLICT! AI: 'gnai1' vs Algo: 'gnas2'"],
        }
        entries = extract_validation_entries(validation)
        assert len(entries) == 2
        assert entries[0]["bucket"] == "critical_warnings"
        assert entries[1]["bucket"] == "algo_conflicts"


class TestInjectOligomerAlerts:
    def test_handles_null_critical_warnings_bucket(self):
        from gpcr_tools.csv_generator.validation_display import inject_oligomer_alerts

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
        validation = {"critical_warnings": None}

        inject_oligomer_alerts(oligo, validation)

        assert validation["critical_warnings"] == [
            "CHAIN_ID CORRECTED at 'receptor_info': G -> R (HALLUCINATION). Human confirmation required."
        ]

    def test_promotes_multi_copy_ligand_with_ligand_path(self):
        from gpcr_tools.config import ALERT_MULTI_COPY_LIGAND
        from gpcr_tools.csv_generator.validation_display import inject_oligomer_alerts

        msg = (
            "[MULTI_COPY_LIGAND] at 'ligands[CA]': modelled in 2 copies (instances D, E); "
            "one annotation row may hide copies at distinct sites or with distinct roles. "
            "Human review recommended."
        )
        oligo = {
            "alerts": [{"type": ALERT_MULTI_COPY_LIGAND, "message": msg}],
            "all_gpcr_chains": [],
        }
        validation: dict = {}
        inject_oligomer_alerts(oligo, validation)
        # Promoted verbatim, keeping its own 'ligands[...]' path (not wrapped as
        # a receptor_info alert) so it buckets with the ligand block.
        assert msg in validation["critical_warnings"]


class TestAnalyzeValidationImpact:
    def test_no_warnings_returns_none(self):
        result = analyze_validation_impact("ligands", [{"name": "test"}], {})
        assert result is None

    def test_ghost_ligand_with_underscore_is_fatal(self):
        validation = {
            "critical_warnings": ["GHOST_LIGAND at 'ligands[0]': 'XYZ' not found in API entities."],
            "algo_conflicts": [],
        }
        result = analyze_validation_impact("ligands", [{"name": "bad"}], validation)
        assert result is not None
        assert result["action"] == "DELETE_BLOCK"

    def test_hallucination_dict_delete(self):
        validation = {
            "critical_warnings": [
                "HALLUCINATION ALERT at 'signaling_partners': g-protein does not exist."
            ],
            "algo_conflicts": [],
        }
        result = analyze_validation_impact("signaling_partners", {"g_protein": {}}, validation)
        assert result is not None
        assert result["action"] == "DELETE_BLOCK"

    def test_list_clean_entries(self):
        validation = {
            "critical_warnings": [
                "Ghost Ligand ID at 'ligands[0]': 'XYZ' not found in API entities."
            ],
            "algo_conflicts": [],
        }
        data = [{"name": "bad"}, {"name": "good"}]
        result = analyze_validation_impact("ligands", data, validation)
        assert result is not None
        assert result["action"] == "CLEAN_ENTRIES"
        assert 0 in result["invalid_indices"]

    def test_none_data_returns_none(self):
        result = analyze_validation_impact("test", None, {"critical_warnings": ["test"]})
        assert result is None
