"""Tests for New Era Epic 3: Scientific Transformation Layer (logic.py).

Covers all 4 pure functions: map_label_asym_id, collect_ligand_chains,
apply_db_truncation, build_structure_note.
"""

from gpcr_tools.csv_generator.logic import (
    apply_db_truncation,
    build_structure_note,
    collect_ligand_chains,
    map_label_asym_id,
)

# ── map_label_asym_id ────────────────────────────────────────────────


class TestMapLabelAsymId:
    def test_identity(self):
        assert map_label_asym_id("A", {"A": "A"}) == "A"

    def test_remap(self):
        assert map_label_asym_id("R", {"R": "E"}) == "E"

    def test_comma_separated(self):
        assert map_label_asym_id("A, B", {"A": "A", "B": "C"}) == "A, C"

    def test_empty_string(self):
        assert map_label_asym_id("", {"A": "B"}) == ""

    def test_missing_key_fallback(self):
        """Keys not in the map fall through unchanged."""
        assert map_label_asym_id("X", {"A": "A"}) == "X"

    def test_empty_map(self):
        assert map_label_asym_id("A", {}) == "A"

    def test_multiple_comma_remap(self):
        label_map = {"A": "X", "B": "Y", "C": "Z"}
        assert map_label_asym_id("A, B, C", label_map) == "X, Y, Z"


# ── collect_ligand_chains ────────────────────────────────────────────


class TestCollectLigandChains:
    def test_basic(self):
        ligands = [{"chain_id": "A"}, {"chain_id": "B"}]
        assert collect_ligand_chains(ligands) == {"A", "B"}

    def test_skip_null_sentinels(self):
        ligands = [
            {"chain_id": "A"},
            {"chain_id": "None"},
            {"chain_id": "null"},
            {"chain_id": "n/a"},
        ]
        assert collect_ligand_chains(ligands) == {"A"}

    def test_comma_separated(self):
        ligands = [{"chain_id": "A, B"}]
        assert collect_ligand_chains(ligands) == {"A", "B"}

    def test_empty_chain_id(self):
        ligands = [{"chain_id": ""}]
        assert collect_ligand_chains(ligands) == set()

    def test_missing_chain_id(self):
        ligands = [{"name": "ligand without chain"}]
        assert collect_ligand_chains(ligands) == set()

    def test_deduplication(self):
        ligands = [{"chain_id": "A"}, {"chain_id": "A"}]
        assert collect_ligand_chains(ligands) == {"A"}

    def test_empty_list(self):
        assert collect_ligand_chains([]) == set()


# ── apply_db_truncation ─────────────────────────────────────────────


class TestApplyDbTruncation:
    def test_single_chain_no_truncation(self):
        chain, uniprot, note = apply_db_truncation("A", "aa2ar_human", {}, set())
        assert chain == "A"
        assert uniprot == "aa2ar_human"
        assert note == ""

    def test_multi_chain_with_suggestion(self):
        oligo = {
            "primary_protomer_suggestion": {
                "chain_id": "A",
                "reason": "G-protein bound",
            },
            "all_gpcr_chains": [
                {"chain_id": "A", "slug": "aa2ar_human"},
                {"chain_id": "B", "slug": "aa2ar_human"},
            ],
        }
        chain, uniprot, note = apply_db_truncation("A, B", "aa2ar_human", oligo, set())
        assert chain == "A"
        assert uniprot == "aa2ar_human"
        assert "[DB TRUNCATION:" in note
        assert "primary chain A" in note

    def test_preserves_uniprot_from_chain_info(self):
        oligo = {
            "primary_protomer_suggestion": {"chain_id": "B", "reason": "test"},
            "all_gpcr_chains": [
                {"chain_id": "A", "slug": "drd2_human"},
                {"chain_id": "B", "slug": "oprm_human"},
            ],
        }
        chain, uniprot, _note = apply_db_truncation("A, B", "drd2_human", oligo, set())
        assert chain == "B"
        assert uniprot == "oprm_human"

    def test_orphaned_ligand_warning(self):
        oligo = {
            "primary_protomer_suggestion": {"chain_id": "A", "reason": "test"},
            "all_gpcr_chains": [
                {"chain_id": "A", "slug": "aa2ar_human"},
                {"chain_id": "B", "slug": "aa2ar_human"},
            ],
        }
        chain, _uniprot, note = apply_db_truncation("A, B", "aa2ar_human", oligo, {"A", "B"})
        assert chain == "A"
        assert "[WARNING: Ligands are bound to truncated chains B!]" in note

    def test_no_orphan_when_ligand_on_primary(self):
        oligo = {
            "primary_protomer_suggestion": {"chain_id": "A", "reason": "test"},
            "all_gpcr_chains": [
                {"chain_id": "A", "slug": "aa2ar_human"},
                {"chain_id": "B", "slug": "aa2ar_human"},
            ],
        }
        _chain, _uniprot, note = apply_db_truncation("A, B", "aa2ar_human", oligo, {"A"})
        assert "WARNING" not in note
        assert "[DB TRUNCATION:" in note

    def test_no_suggestion_returns_original(self):
        """Multi-chain but no primary_protomer_suggestion → no truncation."""
        chain, _uniprot, note = apply_db_truncation("A, B", "aa2ar_human", {}, set())
        assert chain == "A, B"
        assert note == ""

    def test_suggestion_missing_chain_id(self):
        oligo = {"primary_protomer_suggestion": {"reason": "test"}}
        chain, _uniprot, note = apply_db_truncation("A, B", "aa2ar_human", oligo, set())
        assert chain == "A, B"
        assert note == ""


# ── build_structure_note ─────────────────────────────────────────────


class TestBuildStructureNote:
    def test_empty_oligo(self):
        result = build_structure_note({"note": "Base note"}, {})
        assert result == "Base note"

    def test_no_note_no_oligo(self):
        result = build_structure_note({}, {})
        assert result == ""

    def test_chain_corrected(self):
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
        result = build_structure_note({"note": ""}, oligo)
        assert "[CHAIN CORRECTED: G -> R" in result

    def test_homomer_classification(self):
        oligo = {
            "classification": "HOMOMER",
            "chain_id_override": {"applied": False},
            "alerts": [],
            "all_gpcr_chains": [
                {"chain_id": "A"},
                {"chain_id": "B"},
            ],
        }
        result = build_structure_note({"note": ""}, oligo)
        assert "[HOMOMER: chains A, B]" in result

    def test_heteromer_classification(self):
        oligo = {
            "classification": "HETEROMER",
            "chain_id_override": {"applied": False},
            "alerts": [],
            "all_gpcr_chains": [
                {"chain_id": "R"},
                {"chain_id": "S"},
            ],
        }
        result = build_structure_note({"note": ""}, oligo)
        assert "[HETEROMER: chains R, S]" in result

    def test_missed_protomer_alert(self):
        oligo = {
            "classification": "MONOMER",
            "chain_id_override": {"applied": False},
            "alerts": [{"type": "MISSED_PROTOMER", "message": "Missed B"}],
            "all_gpcr_chains": [],
        }
        result = build_structure_note({"note": ""}, oligo)
        assert "[MISSED_PROTOMER: Missed B]" in result

    def test_hallucination_alert(self):
        oligo = {
            "chain_id_override": {"applied": False},
            "alerts": [{"type": "HALLUCINATION", "message": "Chain G fake"}],
            "all_gpcr_chains": [],
        }
        result = build_structure_note({"note": ""}, oligo)
        assert "[HALLUCINATION: Chain G fake]" in result

    def test_confirmed_oligomer_not_included(self):
        """CONFIRMED_OLIGOMER alerts should NOT appear in the note."""
        oligo = {
            "classification": "MONOMER",
            "chain_id_override": {"applied": False},
            "alerts": [{"type": "CONFIRMED_OLIGOMER", "message": "All good"}],
            "all_gpcr_chains": [],
        }
        result = build_structure_note({"note": ""}, oligo)
        assert "CONFIRMED_OLIGOMER" not in result

    def test_with_truncation_note(self):
        result = build_structure_note(
            {"note": "Base"},
            {},
            truncation_note="[DB TRUNCATION: test]",
        )
        assert result == "Base [DB TRUNCATION: test]"

    def test_combined(self):
        """Base note + override + classification + truncation → all present."""
        oligo = {
            "classification": "HOMOMER",
            "chain_id_override": {
                "applied": True,
                "original_chain_id": "X",
                "corrected_chain_id": "A",
                "trigger": "7TM_UPGRADE",
            },
            "alerts": [],
            "all_gpcr_chains": [{"chain_id": "A"}, {"chain_id": "B"}],
        }
        result = build_structure_note(
            {"note": "Cryo-EM complex"},
            oligo,
            truncation_note="[DB TRUNCATION: test]",
        )
        assert "Cryo-EM complex" in result
        assert "[CHAIN CORRECTED:" in result
        assert "[HOMOMER:" in result
        assert "[DB TRUNCATION:" in result

    def test_non_string_note(self):
        """Non-string note values should be handled gracefully."""
        result = build_structure_note({"note": 42}, {})
        assert result == "42"
