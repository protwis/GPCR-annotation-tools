"""Tests for CSV transformation and writing logic.

These tests cover the pure data transformation layer — no UI, no user interaction.
"""

import csv
from dataclasses import replace
from unittest.mock import patch

from gpcr_tools.csv_generator.csv_writer import (
    append_to_csvs,
    sanitize_value,
    transform_for_csv,
)


class TestSanitizeValue:
    def test_none_returns_empty(self):
        assert sanitize_value(None) == ""

    def test_string_stripped(self):
        assert sanitize_value("  hello  ") == "hello"

    def test_numeric(self):
        assert sanitize_value(2.5) == "2.5"

    def test_zero(self):
        assert sanitize_value(0) == "0"

    def test_bool(self):
        assert sanitize_value(True) == "True"


class TestTransformForCSV:
    def test_produces_all_csv_keys(self, sample_pdb_data):
        result = transform_for_csv("TEST1", sample_pdb_data)
        from gpcr_tools.config import CSV_SCHEMA

        assert set(result.keys()) == set(CSV_SCHEMA.keys())

    def test_structures_csv_row(self, sample_pdb_data):
        result = transform_for_csv("TEST1", sample_pdb_data)
        rows = result["structures.csv"]
        assert len(rows) == 1
        row = rows[0]
        assert row["PDB"] == "TEST1"
        assert row["Receptor_UniProt"] == "aa2ar_human"
        assert row["Method"] == "ELECTRON MICROSCOPY"
        assert row["Resolution"] == "2.5"
        assert row["State"] == "Active"
        assert row["ChainID"] == "R"
        assert row["Date"] == "2025-01-15"

    def test_ligands_csv_row(self, sample_pdb_data):
        result = transform_for_csv("TEST1", sample_pdb_data)
        rows = result["ligands.csv"]
        assert len(rows) == 1
        row = rows[0]
        assert row["PDB"] == "TEST1"
        assert row["Name"] == "Adenosine"
        assert row["PubChemID"] == "2519"
        assert row["Role"] == "Agonist"
        assert row["ChainID"] == "A"
        assert row["InChIKey"] == "OIRDTQYFTABQOQ-KQYNXXCUSA-N"

    def test_smiles_stereo_priority(self, sample_pdb_data):
        """SMILES_stereo should take priority over SMILES."""
        result = transform_for_csv("TEST1", sample_pdb_data)
        row = result["ligands.csv"][0]
        expected_smiles = sample_pdb_data["ligands"][0]["SMILES_stereo"]
        assert row["SMILES"] == expected_smiles

    def test_g_protein_mapping(self, sample_pdb_data):
        result = transform_for_csv("TEST1", sample_pdb_data)
        rows = result["g_proteins.csv"]
        assert len(rows) == 1
        row = rows[0]
        assert row["Alpha_UniProt"] == "gnas2_human"
        assert row["Alpha_ChainID"] == "G"
        assert row["Beta_UniProt"] == "gbb1_human"
        assert row["Gamma_UniProt"] == "gbg2_human"

    def test_nanobody_dispatch(self, sample_pdb_data):
        result = transform_for_csv("TEST1", sample_pdb_data)
        rows = result["nanobodies.csv"]
        assert len(rows) == 1
        assert rows[0]["Name"] == "Nb35"

    def test_no_arrestin_when_absent(self, sample_pdb_data):
        result = transform_for_csv("TEST1", sample_pdb_data)
        assert result["arrestins.csv"] == []

    def test_empty_data_produces_structure_row(self):
        """Even minimal data should produce a structures.csv entry."""
        result = transform_for_csv("EMPTY", {})
        assert len(result["structures.csv"]) == 1
        assert result["structures.csv"][0]["PDB"] == "EMPTY"

    def test_controversy_data_transform(self, sample_controversy_data):
        """Test that controversy fixture also transforms correctly."""
        result = transform_for_csv("TEST2", sample_controversy_data)
        assert len(result["structures.csv"]) == 1
        assert result["structures.csv"][0]["Method"] == "X-RAY DIFFRACTION"
        assert result["g_proteins.csv"] == []  # no g-protein in this fixture

    def test_label_asym_id_with_oligomer(self, sample_oligomer_data):
        """Oligomer fixture with label_asym_id_map → mapped values in CSV rows."""
        result = transform_for_csv("OLIGO1", sample_oligomer_data)
        struct_row = result["structures.csv"][0]
        # Oligomer fixture has chain "A, B" → truncated to "A" (primary)
        # label_map: {"A": "A"} → label_asym_id = "A"
        assert "label_asym_id" in struct_row
        assert struct_row["label_asym_id"] == "A"

    def test_label_asym_id_without_oligo(self, sample_pdb_data):
        """Without oligomer_analysis, label_asym_id falls back to chain_id."""
        result = transform_for_csv("TEST1", sample_pdb_data)
        struct_row = result["structures.csv"][0]
        # No label_map → fallback: chain_id "R" mapped to itself
        assert struct_row["label_asym_id"] == "R"

    def test_truncation_in_structures(self, sample_oligomer_data):
        """Multi-chain oligomer fixture → structures.csv has single primary chain."""
        result = transform_for_csv("OLIGO1", sample_oligomer_data)
        struct_row = result["structures.csv"][0]
        # receptor_info.chain_id = "A, B" → truncated to "A"
        assert struct_row["ChainID"] == "A"
        assert "DB TRUNCATION" in struct_row["Note"]

    def test_note_enriched_with_oligo(self, sample_oligomer_data):
        """Oligomer fixture → Note contains classification + alerts."""
        result = transform_for_csv("OLIGO1", sample_oligomer_data)
        note = result["structures.csv"][0]["Note"]
        assert "HOMOMER" in note
        assert "MISSED_PROTOMER" in note

    def test_g_protein_label_asym_id(self, sample_oligomer_data):
        """G-protein subunit chain IDs are mapped via label_asym_id_map."""
        result = transform_for_csv("OLIGO1", sample_oligomer_data)
        gp_row = result["g_proteins.csv"][0]
        # label_map: D→A, C→D, E→B
        assert gp_row["Alpha_label_asym_id"] == "A"  # chain D → A
        assert gp_row["Beta_label_asym_id"] == "D"  # chain C → D
        assert gp_row["Gamma_label_asym_id"] == "B"  # chain E → B

    def test_ligand_label_asym_id(self, sample_oligomer_data):
        """Ligand chain IDs are mapped via label_asym_id_map."""
        result = transform_for_csv("OLIGO1", sample_oligomer_data)
        lig_rows = result["ligands.csv"]
        assert len(lig_rows) == 2
        # First ligand chain A → A (identity)
        assert lig_rows[0]["label_asym_id"] == "A"
        # Second ligand chain B → B (identity)
        assert lig_rows[1]["label_asym_id"] == "B"


def _mock_config_with_csv_dir(csv_dir):
    """Return a patched get_config that redirects csv_output_dir to *csv_dir*."""
    from gpcr_tools.config import get_config

    real_cfg = get_config()
    fake_cfg = replace(real_cfg, csv_output_dir=csv_dir)
    return patch("gpcr_tools.csv_generator.csv_writer.get_config", return_value=fake_cfg)


class TestAppendToCSVs:
    def test_creates_file_with_header(self, tmp_path, monkeypatch, sample_pdb_data):
        """Test that a new CSV file gets a header row."""
        monkeypatch.setenv("GPCR_WORKSPACE", str(tmp_path))
        from gpcr_tools.config import reset_config

        reset_config()

        csv_dir = tmp_path / "csv_out"
        with _mock_config_with_csv_dir(csv_dir):
            csv_data = transform_for_csv("TEST1", sample_pdb_data)
            append_to_csvs(csv_data)

        structures_file = csv_dir / "structures.csv"
        assert structures_file.exists()

        with open(structures_file) as f:
            reader = csv.reader(f, delimiter="\t")
            rows = list(reader)

        assert len(rows) == 2  # header + 1 data row
        assert rows[0][0] == "PDB"  # header
        assert rows[1][0] == "TEST1"  # data

    def test_append_no_duplicate_header(self, tmp_path, monkeypatch, sample_pdb_data):
        """Test that appending to an existing file does NOT duplicate the header."""
        monkeypatch.setenv("GPCR_WORKSPACE", str(tmp_path))
        from gpcr_tools.config import reset_config

        reset_config()

        csv_dir = tmp_path / "csv_out"
        with _mock_config_with_csv_dir(csv_dir):
            csv_data_1 = transform_for_csv("TEST1", sample_pdb_data)
            csv_data_2 = transform_for_csv("TEST2", sample_pdb_data)

            append_to_csvs(csv_data_1)
            append_to_csvs(csv_data_2)

        structures_file = csv_dir / "structures.csv"
        with open(structures_file) as f:
            reader = csv.reader(f, delimiter="\t")
            rows = list(reader)

        assert len(rows) == 3
        assert rows[0][0] == "PDB"  # header
        assert rows[1][0] == "TEST1"
        assert rows[2][0] == "TEST2"

    def test_empty_csv_data_no_file_created(self, tmp_path, monkeypatch):
        """If all CSV data is empty, no files should be created."""
        monkeypatch.setenv("GPCR_WORKSPACE", str(tmp_path))
        from gpcr_tools.config import reset_config

        reset_config()

        csv_dir = tmp_path / "csv_out"
        with _mock_config_with_csv_dir(csv_dir):
            from gpcr_tools.config import CSV_SCHEMA

            empty_data = {fname: [] for fname in CSV_SCHEMA}
            append_to_csvs(empty_data)

        csv_files = list(csv_dir.glob("*.csv")) if csv_dir.exists() else []
        assert len(csv_files) == 0

    def test_mismatched_headers_raises_error(self, tmp_path, monkeypatch, sample_pdb_data):
        """Existing CSV with outdated headers → CsvSchemaMismatchError raised."""
        import pytest

        monkeypatch.setenv("GPCR_WORKSPACE", str(tmp_path))
        from gpcr_tools.config import reset_config
        from gpcr_tools.csv_generator.exceptions import CsvSchemaMismatchError

        reset_config()

        csv_dir = tmp_path / "csv_out"
        csv_dir.mkdir(parents=True)

        # Write a file with old headers (missing label_asym_id)
        old_headers = "PDB\tReceptor_UniProt\tMethod\tResolution\tState\tChainID\tNote\tDate\n"
        structures_file = csv_dir / "structures.csv"
        structures_file.write_text(old_headers)

        with _mock_config_with_csv_dir(csv_dir):
            csv_data = transform_for_csv("TEST1", sample_pdb_data)
            with pytest.raises(CsvSchemaMismatchError):
                append_to_csvs(csv_data)

        # File should still have only the old header — no data appended
        content = structures_file.read_text()
        assert "TEST1" not in content
        assert content.strip() == old_headers.strip()

    def test_matching_headers_appended(self, tmp_path, monkeypatch, sample_pdb_data):
        """Existing CSV with correct headers → rows appended normally."""
        monkeypatch.setenv("GPCR_WORKSPACE", str(tmp_path))
        from gpcr_tools.config import CSV_SCHEMA, reset_config

        reset_config()

        csv_dir = tmp_path / "csv_out"
        csv_dir.mkdir(parents=True)

        # Write a file with current correct headers
        correct_headers = "\t".join(CSV_SCHEMA["structures.csv"]) + "\n"
        structures_file = csv_dir / "structures.csv"
        structures_file.write_text(correct_headers)

        with _mock_config_with_csv_dir(csv_dir):
            csv_data = transform_for_csv("TEST1", sample_pdb_data)
            append_to_csvs(csv_data)

        with open(structures_file) as f:
            reader = csv.reader(f, delimiter="\t")
            rows = list(reader)

        assert len(rows) == 2  # header + 1 data row
        assert rows[1][0] == "TEST1"
