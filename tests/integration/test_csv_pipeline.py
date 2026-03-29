"""Integration test: full CSV pipeline from JSON loading to CSV output.

Tests the complete happy path without interactive prompts.
"""

import csv
import json


class TestCSVPipeline:
    """End-to-end test: load fixture → transform → write CSV → verify content."""

    def test_full_pipeline(self, configure_paths, sample_pdb_data):
        """Test the complete pipeline from data load to CSV output."""
        _data_dir, output_dir = configure_paths
        csv_dir = output_dir / "csv"
        state_dir = output_dir.parent / "state"

        from gpcr_tools.csv_generator.csv_writer import append_to_csvs, transform_for_csv
        from gpcr_tools.csv_generator.data_loader import (
            get_pending_pdbs,
            load_pdb_data,
            update_processed_log,
        )

        pending, skipped, total = get_pending_pdbs()
        assert total == 1
        assert pending == ["TEST1"]
        assert skipped == []

        main_data, _controversies, _validation = load_pdb_data("TEST1")
        assert main_data is not None
        assert main_data["receptor_info"]["uniprot_entry_name"] == "aa2ar_human"

        csv_data = transform_for_csv("TEST1", main_data)
        assert len(csv_data["structures.csv"]) == 1
        assert len(csv_data["ligands.csv"]) == 1
        assert len(csv_data["g_proteins.csv"]) == 1
        assert len(csv_data["nanobodies.csv"]) == 1

        append_to_csvs(csv_data)

        structures_path = csv_dir / "structures.csv"
        assert structures_path.exists()
        with open(structures_path) as f:
            reader = csv.DictReader(f, delimiter="\t")
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["PDB"] == "TEST1"
        assert rows[0]["Receptor_UniProt"] == "aa2ar_human"
        assert rows[0]["Resolution"] == "2.5"

        ligands_path = csv_dir / "ligands.csv"
        assert ligands_path.exists()
        with open(ligands_path) as f:
            reader = csv.DictReader(f, delimiter="\t")
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["Name"] == "Adenosine"
        assert rows[0]["SMILES"] != ""

        gp_path = csv_dir / "g_proteins.csv"
        assert gp_path.exists()
        with open(gp_path) as f:
            reader = csv.DictReader(f, delimiter="\t")
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["Alpha_UniProt"] == "gnas2_human"

        update_processed_log("TEST1", "completed")

        pending, skipped, total = get_pending_pdbs()
        assert total == 1
        assert pending == []
        assert skipped == []

        assert (state_dir / "processed_log.json").exists()

    def test_multiple_pdbs(self, configure_paths, sample_pdb_data, sample_controversy_data):
        """Test processing multiple PDBs sequentially."""
        data_dir, output_dir = configure_paths
        csv_dir = output_dir / "csv"

        with open(data_dir / "TEST2.json", "w") as f:
            json.dump(sample_controversy_data, f)

        from gpcr_tools.csv_generator.csv_writer import append_to_csvs, transform_for_csv
        from gpcr_tools.csv_generator.data_loader import get_pending_pdbs, load_pdb_data

        pending, skipped, total = get_pending_pdbs()
        assert total == 2
        assert len(pending) == 2
        assert skipped == []

        for pdb_id in pending:
            main_data, _, _ = load_pdb_data(pdb_id)
            assert main_data is not None
            csv_data = transform_for_csv(pdb_id, main_data)
            append_to_csvs(csv_data)

        structures_path = csv_dir / "structures.csv"
        with open(structures_path) as f:
            reader = csv.DictReader(f, delimiter="\t")
            rows = list(reader)
        assert len(rows) == 2
        pdb_ids = {row["PDB"] for row in rows}
        assert pdb_ids == {"TEST1", "TEST2"}
