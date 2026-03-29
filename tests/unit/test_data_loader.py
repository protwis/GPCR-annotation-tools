"""Tests for data loading and PDB discovery."""

import json


class TestGetPendingPdbs:
    def test_all_new(self, configure_paths):
        """All PDBs in data dir should be pending when no log exists."""
        _data_dir, _output_dir = configure_paths

        from gpcr_tools.csv_generator.data_loader import get_pending_pdbs

        pending, skipped, total = get_pending_pdbs()
        assert total == 1
        assert pending == ["TEST1"]
        assert skipped == []

    def test_completed_excluded(self, configure_paths):
        """PDBs with status == 'completed' should be excluded from all queues."""
        _data_dir, output_dir = configure_paths
        state_dir = output_dir.parent / "state"
        state_dir.mkdir(exist_ok=True)

        log = {"TEST1": {"status": "completed", "timestamp": "2025-01-01T00:00:00Z"}}
        with open(state_dir / "processed_log.json", "w") as f:
            json.dump(log, f)

        from gpcr_tools.csv_generator.data_loader import get_pending_pdbs

        pending, skipped, total = get_pending_pdbs()
        assert total == 1
        assert pending == []
        assert skipped == []

    def test_skipped_returned_separately(self, configure_paths):
        """PDBs with status == 'skipped' should appear in the skipped list, not pending."""
        _data_dir, output_dir = configure_paths
        state_dir = output_dir.parent / "state"
        state_dir.mkdir(exist_ok=True)

        log = {"TEST1": {"status": "skipped", "timestamp": "2025-01-01T00:00:00Z"}}
        with open(state_dir / "processed_log.json", "w") as f:
            json.dump(log, f)

        from gpcr_tools.csv_generator.data_loader import get_pending_pdbs

        pending, skipped, total = get_pending_pdbs()
        assert total == 1
        assert pending == []
        assert skipped == ["TEST1"]

    def test_failed_returns_to_pending(self, configure_paths):
        """PDBs with status == 'failed' should auto-retry (appear in pending)."""
        _data_dir, output_dir = configure_paths
        state_dir = output_dir.parent / "state"
        state_dir.mkdir(exist_ok=True)

        log = {"TEST1": {"status": "failed", "timestamp": "2025-01-01T00:00:00Z"}}
        with open(state_dir / "processed_log.json", "w") as f:
            json.dump(log, f)

        from gpcr_tools.csv_generator.data_loader import get_pending_pdbs

        pending, skipped, total = get_pending_pdbs()
        assert total == 1
        assert pending == ["TEST1"]
        assert skipped == []

    def test_mixed_statuses(self, configure_paths, sample_pdb_data):
        """Test correct triage with a mix of completed, skipped, failed, and new PDBs."""
        data_dir, output_dir = configure_paths
        state_dir = output_dir.parent / "state"
        state_dir.mkdir(exist_ok=True)

        for name in ["TEST2", "TEST3", "TEST4"]:
            with open(data_dir / f"{name}.json", "w") as f:
                json.dump(sample_pdb_data, f)

        log = {
            "TEST1": {"status": "completed", "timestamp": "2025-01-01T00:00:00Z"},
            "TEST2": {"status": "skipped", "timestamp": "2025-01-01T00:00:00Z"},
            "TEST3": {"status": "failed", "timestamp": "2025-01-01T00:00:00Z"},
        }
        with open(state_dir / "processed_log.json", "w") as f:
            json.dump(log, f)

        from gpcr_tools.csv_generator.data_loader import get_pending_pdbs

        pending, skipped, total = get_pending_pdbs()
        assert total == 4
        assert pending == ["TEST3", "TEST4"]  # failed + new
        assert skipped == ["TEST2"]  # skipped only

    def test_empty_data_dir(self, tmp_path, monkeypatch):
        """Empty data directory should return nothing."""
        workspace = tmp_path / "ws"
        workspace.mkdir()
        (workspace / "aggregated").mkdir()
        monkeypatch.setenv("GPCR_WORKSPACE", str(workspace))

        from gpcr_tools.config import reset_config

        reset_config()

        from gpcr_tools.csv_generator.data_loader import get_pending_pdbs

        pending, skipped, total = get_pending_pdbs()
        assert total == 0
        assert pending == []
        assert skipped == []


class TestLoadPdbData:
    def test_load_existing(self, configure_paths):
        """Loading an existing PDB should return data dicts."""
        from gpcr_tools.csv_generator.data_loader import load_pdb_data

        main_data, controversies, validation = load_pdb_data("TEST1")
        assert main_data is not None
        assert "structure_info" in main_data
        assert isinstance(controversies, dict)
        assert isinstance(validation, dict)

    def test_load_nonexistent(self, configure_paths):
        """Loading a non-existent PDB should return None for main_data."""
        from gpcr_tools.csv_generator.data_loader import load_pdb_data

        main_data, _controversies, _validation = load_pdb_data("NONEXISTENT")
        assert main_data is None

    def test_load_with_voting_log(self, configure_paths, sample_voting_log):
        """Loading a PDB with a voting log should populate controversy_map."""
        data_dir, _ = configure_paths

        log_dir = data_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        with open(log_dir / "TEST1_voting_log.json", "w") as f:
            json.dump(sample_voting_log, f)

        from gpcr_tools.csv_generator.data_loader import load_pdb_data

        main_data, controversies, _validation = load_pdb_data("TEST1")
        assert main_data is not None
        assert len(controversies) == 2
        assert "receptor_info.uniprot_entry_name" in controversies


class TestUpdateProcessedLog:
    def test_creates_log(self, configure_paths):
        """Updating the log should create the file under state/."""
        _, output_dir = configure_paths
        state_dir = output_dir.parent / "state"

        from gpcr_tools.csv_generator.data_loader import update_processed_log

        update_processed_log("TEST1", "completed")

        log_file = state_dir / "processed_log.json"
        assert log_file.exists()

        with open(log_file) as f:
            log = json.load(f)
        assert "TEST1" in log
        assert log["TEST1"]["status"] == "completed"
