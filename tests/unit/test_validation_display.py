"""Tests for validation display and analysis logic."""

from gpcr_tools.csv_generator.validation_display import (
    analyze_validation_impact,
    canonicalize_path,
    extract_validation_entries,
    get_relevant_validation_warnings,
    warning_matches_block,
)


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
