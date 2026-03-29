"""Tests for CSV transformation and writing logic.

These tests cover the pure data transformation layer — no UI, no user interaction.
"""

import csv

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
        # Both are present in fixture; SMILES_stereo should be used
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


class TestAppendToCSVs:
    def test_creates_file_with_header(self, tmp_path, monkeypatch, sample_pdb_data):
        """Test that a new CSV file gets a header row."""
        monkeypatch.setattr("gpcr_tools.csv_generator.csv_writer.OUTPUT_DIR", tmp_path)

        csv_data = transform_for_csv("TEST1", sample_pdb_data)
        append_to_csvs(csv_data)

        structures_file = tmp_path / "structures.csv"
        assert structures_file.exists()

        with open(structures_file) as f:
            reader = csv.reader(f, delimiter="\t")
            rows = list(reader)

        assert len(rows) == 2  # header + 1 data row
        assert rows[0][0] == "PDB"  # header
        assert rows[1][0] == "TEST1"  # data

    def test_append_no_duplicate_header(self, tmp_path, monkeypatch, sample_pdb_data):
        """Test that appending to an existing file does NOT duplicate the header."""
        monkeypatch.setattr("gpcr_tools.csv_generator.csv_writer.OUTPUT_DIR", tmp_path)

        csv_data_1 = transform_for_csv("TEST1", sample_pdb_data)
        csv_data_2 = transform_for_csv("TEST2", sample_pdb_data)

        append_to_csvs(csv_data_1)
        append_to_csvs(csv_data_2)

        structures_file = tmp_path / "structures.csv"
        with open(structures_file) as f:
            reader = csv.reader(f, delimiter="\t")
            rows = list(reader)

        # Should have: 1 header + 2 data rows = 3 total
        assert len(rows) == 3
        assert rows[0][0] == "PDB"  # header
        assert rows[1][0] == "TEST1"
        assert rows[2][0] == "TEST2"

    def test_empty_csv_data_no_file_created(self, tmp_path, monkeypatch):
        """If all CSV data is empty, no files should be created."""
        monkeypatch.setattr("gpcr_tools.csv_generator.csv_writer.OUTPUT_DIR", tmp_path)

        from gpcr_tools.config import CSV_SCHEMA

        empty_data = {fname: [] for fname in CSV_SCHEMA}
        append_to_csvs(empty_data)

        csv_files = list(tmp_path.glob("*.csv"))
        assert len(csv_files) == 0
