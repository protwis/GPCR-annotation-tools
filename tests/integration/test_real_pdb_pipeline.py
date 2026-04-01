"""Function-level real-data regression tests for the CSV generator pipeline.

Covers load_pdb_data, inject_oligomer_alerts, transform_for_csv, and
append_to_csvs using committed real PDB fixtures.
"""

from pathlib import Path

import pytest

from tests.conftest import REAL_PDB_IDS

# ── Helpers ──────────────────────────────────────────────────────────────


def _load_and_inject(pdb_id: str) -> tuple[dict, dict, dict]:
    """Load a real PDB and inject oligomer alerts.  Returns (main_data, controversies, validation_data)."""
    from gpcr_tools.csv_generator.data_loader import load_pdb_data
    from gpcr_tools.csv_generator.validation_display import inject_oligomer_alerts

    main_data, controversies, validation_data = load_pdb_data(pdb_id)
    assert main_data is not None
    oligo = main_data.get("oligomer_analysis") or {}
    inject_oligomer_alerts(oligo, validation_data)
    return main_data, controversies, validation_data


# ── RP-2.1: Parametrized Smoke Tests ────────────────────────────────────

# Expected non-empty CSV files per fixture, from PoC verification.
EXPECTED_CSV_FILES: dict[str, set[str]] = {
    "5G53": {"structures.csv", "ligands.csv", "g_proteins.csv"},
    "8TII": {"structures.csv", "ligands.csv", "arrestins.csv", "nanobodies.csv", "antibodies.csv"},
    "9AS1": {"structures.csv", "ligands.csv", "g_proteins.csv", "arrestins.csv"},
    "9BLW": {"structures.csv", "ligands.csv", "g_proteins.csv", "nanobodies.csv", "ramp.csv"},
    "9EJZ": {"structures.csv", "ligands.csv", "g_proteins.csv", "nanobodies.csv", "scfv.csv"},
    "9IQS": {
        "structures.csv",
        "ligands.csv",
        "fusion_proteins.csv",
        "other_aux_proteins.csv",
    },
    "9M88": {
        "structures.csv",
        "ligands.csv",
        "g_proteins.csv",
        "arrestins.csv",
        "fusion_proteins.csv",
    },
    "9NOR": {"structures.csv", "ligands.csv", "g_proteins.csv"},
    "9O38": {"structures.csv", "ligands.csv", "g_proteins.csv", "nanobodies.csv"},
}


class TestSmokeLoadTransformAppend:
    """RP-2.1: end-to-end smoke test per fixture."""

    @pytest.mark.parametrize("pdb_id", REAL_PDB_IDS)
    def test_full_pipeline_no_crash(self, pdb_id: str, real_pdb_workspace: Path) -> None:
        from gpcr_tools.csv_generator.csv_writer import append_to_csvs, transform_for_csv

        main_data, _, _ = _load_and_inject(pdb_id)
        csv_data = transform_for_csv(pdb_id, main_data)
        append_to_csvs(csv_data)

    @pytest.mark.parametrize("pdb_id", REAL_PDB_IDS)
    def test_csv_file_sets_match_expected(self, pdb_id: str, real_pdb_workspace: Path) -> None:
        from gpcr_tools.csv_generator.csv_writer import transform_for_csv

        main_data, _, _ = _load_and_inject(pdb_id)
        csv_data = transform_for_csv(pdb_id, main_data)
        non_empty = {k for k, v in csv_data.items() if v}
        assert non_empty == EXPECTED_CSV_FILES[pdb_id], (
            f"{pdb_id}: expected {EXPECTED_CSV_FILES[pdb_id]}, got {non_empty}"
        )


# ── RP-2.2: Focused Loader Tests ────────────────────────────────────────


class TestLoaderSidecarPrecision:
    """RP-2.2: exact controversy key sets and warning substring precision."""

    def test_9m88_controversy_keys(self, real_pdb_workspace: Path) -> None:
        from gpcr_tools.csv_generator.data_loader import load_pdb_data

        _, controversies, _ = load_pdb_data("9M88")
        assert set(controversies.keys()) == {
            "ligands[5YM].name",
            "ligands[A1EM3].name",
            "ligands[A1EM3].type",
            "receptor_info.chain_id",
            "receptor_info.uniprot_entry_name",
        }

    def test_9as1_controversy_keys(self, real_pdb_workspace: Path) -> None:
        from gpcr_tools.csv_generator.data_loader import load_pdb_data

        _, controversies, _ = load_pdb_data("9AS1")
        assert set(controversies.keys()) == {
            "ligands[A1AFV].name",
            "structure_info.state.evidence.source",
        }

    def test_9blw_controversy_keys(self, real_pdb_workspace: Path) -> None:
        from gpcr_tools.csv_generator.data_loader import load_pdb_data

        _, controversies, _ = load_pdb_data("9BLW")
        assert set(controversies.keys()) == {
            "auxiliary_proteins[Nanobody-35].type.evidence.source",
            "ligands[None].name",
        }

    def test_9iqs_controversy_keys(self, real_pdb_workspace: Path) -> None:
        from gpcr_tools.csv_generator.data_loader import load_pdb_data

        _, controversies, _ = load_pdb_data("9IQS")
        assert set(controversies.keys()) == {
            "auxiliary_proteins[Soluble cytochrome b562].type.value",
            "ligands[None].type",
        }

    def test_9o38_controversy_keys(self, real_pdb_workspace: Path) -> None:
        from gpcr_tools.csv_generator.data_loader import load_pdb_data

        _, controversies, _ = load_pdb_data("9O38")
        assert set(controversies.keys()) == {
            "ligands[None].role.value",
            "ligands[None].type",
        }

    def test_9nor_ghost_ligand_warnings(self, real_pdb_workspace: Path) -> None:
        from gpcr_tools.csv_generator.data_loader import load_pdb_data

        _, _, validation_data = load_pdb_data("9NOR")
        warnings = validation_data.get("critical_warnings", [])
        warning_text = " ".join(warnings).lower()
        assert "ghost ligand" in warning_text
        assert "sul" in warning_text
        assert "apm" in warning_text

    def test_9ejz_uniprot_clash_warning(self, real_pdb_workspace: Path) -> None:
        from gpcr_tools.csv_generator.data_loader import load_pdb_data

        _, _, validation_data = load_pdb_data("9EJZ")
        warnings = validation_data.get("critical_warnings", [])
        warning_text = " ".join(warnings)
        assert "UNIPROT_CLASH" in warning_text
        assert "chrm5_human" in warning_text

    def test_9as1_ghost_chain_and_fake_uniprot_warnings(self, real_pdb_workspace: Path) -> None:
        from gpcr_tools.csv_generator.data_loader import load_pdb_data

        _, _, validation_data = load_pdb_data("9AS1")
        warnings = validation_data.get("critical_warnings", [])
        warning_text = " ".join(warnings)
        assert "Ghost Chain" in warning_text
        assert "Fake UniProt" in warning_text

    def test_9as1_algo_conflict_hallucination(self, real_pdb_workspace: Path) -> None:
        from gpcr_tools.csv_generator.data_loader import load_pdb_data

        _, _, validation_data = load_pdb_data("9AS1")
        conflicts = validation_data.get("algo_conflicts", [])
        assert len(conflicts) == 1
        assert "HALLUCINATION ALERT" in conflicts[0]
        assert "gnaq_human" in conflicts[0]

    def test_9m88_uniprot_clash_warning(self, real_pdb_workspace: Path) -> None:
        from gpcr_tools.csv_generator.data_loader import load_pdb_data

        _, _, validation_data = load_pdb_data("9M88")
        warnings = validation_data.get("critical_warnings", [])
        warning_text = " ".join(warnings)
        assert "UNIPROT_CLASH" in warning_text
        assert "gpr3_human" in warning_text

    def test_9o38_algo_conflict_tiebreaker(self, real_pdb_workspace: Path) -> None:
        from gpcr_tools.csv_generator.data_loader import load_pdb_data

        _, _, validation_data = load_pdb_data("9O38")
        conflicts = validation_data.get("algo_conflicts", [])
        assert len(conflicts) == 1
        assert "TIE-BREAKER OVERRIDE" in conflicts[0]


# ── RP-2.3: Focused Transform Tests ─────────────────────────────────────


class TestTransformPrecision:
    """RP-2.3: single-fixture regression checks for transform_for_csv."""

    def test_5g53_db_truncation(self, real_pdb_workspace: Path) -> None:
        from gpcr_tools.csv_generator.csv_writer import transform_for_csv

        main_data, _, _ = _load_and_inject("5G53")
        csv_data = transform_for_csv("5G53", main_data)
        row = csv_data["structures.csv"][0]
        assert row["ChainID"] == "A"
        assert row["Receptor_UniProt"] == "aa2ar_human"
        assert "DB TRUNCATION" in row["Note"]
        assert "HOMOMER" in row["Note"]
        assert "chains A, B" in row["Note"]

    def test_5g53_ligand_count(self, real_pdb_workspace: Path) -> None:
        from gpcr_tools.csv_generator.csv_writer import transform_for_csv

        main_data, _, _ = _load_and_inject("5G53")
        csv_data = transform_for_csv("5G53", main_data)
        assert len(csv_data["ligands.csv"]) == 3

    def test_9m88_chain_correction_note(self, real_pdb_workspace: Path) -> None:
        from gpcr_tools.csv_generator.csv_writer import transform_for_csv

        main_data, _, _ = _load_and_inject("9M88")
        csv_data = transform_for_csv("9M88", main_data)
        row = csv_data["structures.csv"][0]
        assert row["ChainID"] == "C"
        assert "CHAIN CORRECTED" in row["Note"]
        assert "A -> C" in row["Note"]
        assert "HALLUCINATION" in row["Note"]
        assert "HOMOMER" in row["Note"]

    def test_9m88_ligand_count(self, real_pdb_workspace: Path) -> None:
        from gpcr_tools.csv_generator.csv_writer import transform_for_csv

        main_data, _, _ = _load_and_inject("9M88")
        csv_data = transform_for_csv("9M88", main_data)
        assert len(csv_data["ligands.csv"]) == 6

    def test_9m88_fusion_protein(self, real_pdb_workspace: Path) -> None:
        from gpcr_tools.csv_generator.csv_writer import transform_for_csv

        main_data, _, _ = _load_and_inject("9M88")
        csv_data = transform_for_csv("9M88", main_data)
        assert len(csv_data["fusion_proteins.csv"]) == 1

    def test_8tii_antibodies(self, real_pdb_workspace: Path) -> None:
        from gpcr_tools.csv_generator.csv_writer import transform_for_csv

        main_data, _, _ = _load_and_inject("8TII")
        csv_data = transform_for_csv("8TII", main_data)
        assert len(csv_data["antibodies.csv"]) == 1
        assert csv_data["antibodies.csv"][0]["PDB"] == "8TII"

    def test_8tii_arrestins(self, real_pdb_workspace: Path) -> None:
        from gpcr_tools.csv_generator.csv_writer import transform_for_csv

        main_data, _, _ = _load_and_inject("8TII")
        csv_data = transform_for_csv("8TII", main_data)
        assert len(csv_data["arrestins.csv"]) == 1
        assert csv_data["arrestins.csv"][0]["PDB"] == "8TII"

    def test_9blw_ramp(self, real_pdb_workspace: Path) -> None:
        from gpcr_tools.csv_generator.csv_writer import transform_for_csv

        main_data, _, _ = _load_and_inject("9BLW")
        csv_data = transform_for_csv("9BLW", main_data)
        assert len(csv_data["ramp.csv"]) == 1
        assert csv_data["ramp.csv"][0]["PDB"] == "9BLW"

    def test_9iqs_other_aux_proteins(self, real_pdb_workspace: Path) -> None:
        from gpcr_tools.csv_generator.csv_writer import transform_for_csv

        main_data, _, _ = _load_and_inject("9IQS")
        csv_data = transform_for_csv("9IQS", main_data)
        assert len(csv_data["other_aux_proteins.csv"]) == 2
        assert all(r["PDB"] == "9IQS" for r in csv_data["other_aux_proteins.csv"])

    def test_9ejz_scfv(self, real_pdb_workspace: Path) -> None:
        from gpcr_tools.csv_generator.csv_writer import transform_for_csv

        main_data, _, _ = _load_and_inject("9EJZ")
        csv_data = transform_for_csv("9EJZ", main_data)
        assert len(csv_data["scfv.csv"]) == 1
        assert csv_data["scfv.csv"][0]["PDB"] == "9EJZ"

    def test_9o38_nanobodies(self, real_pdb_workspace: Path) -> None:
        from gpcr_tools.csv_generator.csv_writer import transform_for_csv

        main_data, _, _ = _load_and_inject("9O38")
        csv_data = transform_for_csv("9O38", main_data)
        assert len(csv_data["nanobodies.csv"]) == 1
        assert csv_data["nanobodies.csv"][0]["PDB"] == "9O38"

    def test_9iqs_heteromer_note(self, real_pdb_workspace: Path) -> None:
        from gpcr_tools.csv_generator.csv_writer import transform_for_csv

        main_data, _, _ = _load_and_inject("9IQS")
        csv_data = transform_for_csv("9IQS", main_data)
        note = csv_data["structures.csv"][0]["Note"]
        assert "HETEROMER" in note
        assert "MISSED_PROTOMER" in note

    def test_9nor_heteromer_note(self, real_pdb_workspace: Path) -> None:
        from gpcr_tools.csv_generator.csv_writer import transform_for_csv

        main_data, _, _ = _load_and_inject("9NOR")
        csv_data = transform_for_csv("9NOR", main_data)
        note = csv_data["structures.csv"][0]["Note"]
        assert "HETEROMER" in note
        assert "MISSED_PROTOMER" in note


# ── RP-2.4: Focused Injection & Warning Routing Tests ────────────────────


class TestInjectionAndWarningRouting:
    """RP-2.4: oligomer alert injection and warning routing precision."""

    def test_9m88_chain_corrected_injection(self, real_pdb_workspace: Path) -> None:
        _, _, validation_data = _load_and_inject("9M88")
        warnings = validation_data.get("critical_warnings", [])
        chain_corrected = [w for w in warnings if "CHAIN_ID CORRECTED" in w]
        assert len(chain_corrected) == 1
        assert "A -> C" in chain_corrected[0]

    def test_9m88_hallucination_injection(self, real_pdb_workspace: Path) -> None:
        _, _, validation_data = _load_and_inject("9M88")
        warnings = validation_data.get("critical_warnings", [])
        oligomer_hallucination = [
            w for w in warnings if "OLIGOMER ALERT" in w and "HALLUCINATION" in w
        ]
        assert len(oligomer_hallucination) == 1
        assert "receptor_info" in oligomer_hallucination[0]
        # The CHAIN_ID CORRECTED warning also mentions HALLUCINATION as its trigger,
        # so total HALLUCINATION-containing warnings is 2.
        all_hallucination = [w for w in warnings if "HALLUCINATION" in w]
        assert len(all_hallucination) == 2

    def test_9o38_incomplete_7tm_injection(self, real_pdb_workspace: Path) -> None:
        _, _, validation_data = _load_and_inject("9O38")
        warnings = validation_data.get("critical_warnings", [])
        structural = [w for w in warnings if "INCOMPLETE" in w or "7TM" in w]
        assert len(structural) == 1
        assert "INCOMPLETE 7TM" in structural[0]

    def test_9iqs_missed_protomer_injection(self, real_pdb_workspace: Path) -> None:
        _, _, validation_data = _load_and_inject("9IQS")
        warnings = validation_data.get("critical_warnings", [])
        missed = [w for w in warnings if "MISSED_PROTOMER" in w]
        assert len(missed) == 1
        assert "receptor_info" in missed[0]

    def test_8tii_incomplete_7tm_injection(self, real_pdb_workspace: Path) -> None:
        _, _, validation_data = _load_and_inject("8TII")
        warnings = validation_data.get("critical_warnings", [])
        structural = [w for w in warnings if "INCOMPLETE" in w or "7TM" in w]
        assert len(structural) == 1

    def test_9nor_ghost_ligand_in_warnings(self, real_pdb_workspace: Path) -> None:
        _, _, validation_data = _load_and_inject("9NOR")
        warnings = validation_data.get("critical_warnings", [])
        ghost_ligand_warnings = [
            w for w in warnings if "ghost ligand" in w.lower() or "ghost_ligand" in w.lower()
        ]
        assert len(ghost_ligand_warnings) >= 2

    def test_9as1_signaling_partners_warning_routing(self, real_pdb_workspace: Path) -> None:
        from gpcr_tools.csv_generator.validation_display import (
            get_relevant_validation_warnings,
        )

        _, _, validation_data = _load_and_inject("9AS1")
        relevant = get_relevant_validation_warnings("signaling_partners", validation_data)
        assert len(relevant) > 0
        combined = " ".join(relevant)
        assert "Ghost Chain" in combined or "HALLUCINATION" in combined

    def test_5g53_no_injection_warnings(self, real_pdb_workspace: Path) -> None:
        """5G53 is clean: no warnings should be injected."""
        _, _, validation_data = _load_and_inject("5G53")
        warnings = validation_data.get("critical_warnings", [])
        assert len(warnings) == 0


# ── RP-2.5: Batch CSV Integrity Test ────────────────────────────────────


class TestBatchCSVIntegrity:
    """RP-2.5: batch-process all fixtures and check CSV integrity."""

    def test_batch_csv_headers_and_no_duplicate_pdbs(self, real_pdb_workspace: Path) -> None:
        import csv as csv_mod

        from gpcr_tools.config import CSV_SCHEMA, get_config
        from gpcr_tools.csv_generator.csv_writer import append_to_csvs, transform_for_csv

        all_csv_data: dict[str, list[dict[str, str]]] = {}
        for pdb_id in REAL_PDB_IDS:
            main_data, _, _ = _load_and_inject(pdb_id)
            csv_data = transform_for_csv(pdb_id, main_data)
            for k, rows in csv_data.items():
                all_csv_data.setdefault(k, []).extend(rows)

        append_to_csvs(all_csv_data)

        cfg = get_config()
        csv_dir = cfg.csv_output_dir

        for filename, expected_fields in CSV_SCHEMA.items():
            filepath = csv_dir / filename
            if not filepath.exists():
                assert filename == "grk.csv", f"Unexpected missing CSV: {filename}"
                continue

            with open(filepath, encoding="utf-8") as f:
                reader = csv_mod.DictReader(f, delimiter="\t")
                assert reader.fieldnames is not None
                assert list(reader.fieldnames) == expected_fields, f"{filename} header mismatch"
                rows = list(reader)

            if filename == "structures.csv":
                pdb_ids = [r["PDB"] for r in rows]
                assert len(pdb_ids) == len(set(pdb_ids)), (
                    f"Duplicate PDB IDs in {filename}: "
                    f"{[p for p in pdb_ids if pdb_ids.count(p) > 1]}"
                )

    def test_structures_csv_row_count(self, real_pdb_workspace: Path) -> None:
        import csv as csv_mod

        from gpcr_tools.config import get_config
        from gpcr_tools.csv_generator.csv_writer import append_to_csvs, transform_for_csv

        all_csv_data: dict[str, list[dict[str, str]]] = {}
        for pdb_id in REAL_PDB_IDS:
            main_data, _, _ = _load_and_inject(pdb_id)
            csv_data = transform_for_csv(pdb_id, main_data)
            for k, rows in csv_data.items():
                all_csv_data.setdefault(k, []).extend(rows)

        append_to_csvs(all_csv_data)

        cfg = get_config()
        filepath = cfg.csv_output_dir / "structures.csv"
        with open(filepath, encoding="utf-8") as f:
            reader = csv_mod.reader(f, delimiter="\t")
            lines = list(reader)

        assert len(lines) == 10, f"Expected 1 header + 9 data = 10, got {len(lines)}"

    def test_grk_csv_not_created(self, real_pdb_workspace: Path) -> None:
        """Confirm grk.csv is never created by the current fixture set."""
        from gpcr_tools.config import get_config
        from gpcr_tools.csv_generator.csv_writer import append_to_csvs, transform_for_csv

        all_csv_data: dict[str, list[dict[str, str]]] = {}
        for pdb_id in REAL_PDB_IDS:
            main_data, _, _ = _load_and_inject(pdb_id)
            csv_data = transform_for_csv(pdb_id, main_data)
            for k, rows in csv_data.items():
                all_csv_data.setdefault(k, []).extend(rows)

        append_to_csvs(all_csv_data)

        cfg = get_config()
        assert not (cfg.csv_output_dir / "grk.csv").exists()
