import json
from pathlib import Path
from unittest.mock import MagicMock

from gpcr_tools.annotator import runner
from gpcr_tools.config import get_config, reset_config


def test_run_single_pdb_skips_if_done(tmp_path, monkeypatch):
    """Test that runner fast-exits if all run files exist."""

    # Mock config via env vars
    monkeypatch.setenv("GPCR_AI_RESULTS_PATH", str(tmp_path / "ai_results"))
    reset_config()
    config = get_config()

    pdb_id = "7W55"
    out_dir = config.ai_results_dir / pdb_id
    out_dir.mkdir(parents=True)

    # Create 2 runs
    for n in range(1, 3):
        (out_dir / f"run_{n}.json").write_text("{}")

    # Attempting to run 2 times should be skipped
    # If it wasn't skipped, it would crash calling the unset Client.
    runner.run_single_pdb(pdb_id, {}, "Prompt", Path("dummy.pdf"), num_runs=2)

    # Output runs still the exact 2
    assert len(list(out_dir.glob("run_*.json"))) == 2


def test_build_and_submit_batch(tmp_path, monkeypatch):
    """Test building a JSONL and submitting it to the Gemini batch API."""
    monkeypatch.setenv("GPCR_ENRICHED_PATH", str(tmp_path / "enriched"))
    monkeypatch.setenv("GPCR_PAPERS_PATH", str(tmp_path / "papers"))
    monkeypatch.setenv("GPCR_AI_RESULTS_PATH", str(tmp_path / "ai_results"))
    monkeypatch.setenv("GPCR_STATE_PATH", str(tmp_path / "state"))
    reset_config()
    config = get_config()

    (tmp_path / "state").mkdir()
    config.enriched_dir.mkdir()
    config.papers_dir.mkdir()

    # Setup dummy target files
    pdb_id = "7W55"
    (config.enriched_dir / f"{pdb_id}.json").write_text("{}")
    (config.papers_dir / f"{pdb_id}.pdf").write_text("%PDF")

    # Mock get_client
    mock_client = MagicMock()
    mock_files = MagicMock()
    mock_uploaded = MagicMock()
    mock_uploaded.uri = "http://mock.uri"
    mock_uploaded.name = "mock/uploaded_file"
    mock_files.upload.return_value = mock_uploaded
    mock_client.files = mock_files

    mock_batches = MagicMock()
    mock_batch_job = MagicMock()
    mock_batch_job.name = "batchJobs/mock_job_name"
    mock_batches.create.return_value = mock_batch_job
    mock_client.batches = mock_batches

    monkeypatch.setattr("gpcr_tools.annotator.runner.get_client", lambda: mock_client)
    monkeypatch.setattr("gpcr_tools.annotator.runner.compress_pdf_if_needed", lambda a, b: a)

    runner.build_and_submit_batch([pdb_id], "Prompt", num_runs=1)

    # Verification
    assert mock_files.upload.call_count == 2  # Once for PDF, once for JSONL
    assert mock_batches.create.call_count == 1

    # Check registry updated
    registry = json.loads(config.uploaded_files_registry_file.read_text())
    assert registry[pdb_id] == "http://mock.uri"

    # Check job updated
    assert config.current_batch_job_file.read_text() == "batchJobs/mock_job_name"


def test_recover_batch(tmp_path, monkeypatch):
    """Test recovering JSONL responses into run_n.json files."""
    monkeypatch.setenv("GPCR_STATE_PATH", str(tmp_path / "state"))
    monkeypatch.setenv("GPCR_AI_RESULTS_PATH", str(tmp_path / "ai_results"))
    reset_config()
    config = get_config()
    config.pipeline_runs_dir.mkdir(parents=True)

    raw_output = config.pipeline_runs_dir / "raw_output_testjob.jsonl"

    # Construct a mock batch response lines
    mock_resp = {
        "id": "7W55__run_01",
        "response": {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "annotate_gpcr_db_structure",
                                    "args": {
                                        # To post processor
                                        "receptor_info": {"uniprot_entry_name": "OPSD_BOVIN"}
                                    },
                                }
                            }
                        ]
                    }
                }
            ]
        },
    }
    raw_output.write_text(json.dumps(mock_resp) + "\n")

    # Mock post_processor so it doesn't try to sanitize empty structure
    def mock_post_process(args):
        return {"sanitized": True, "receptor_info": args.get("receptor_info")}

    monkeypatch.setattr("gpcr_tools.annotator.runner.post_process_annotation", mock_post_process)

    runner.recover_batch()

    out_file = config.ai_results_dir / "7W55" / "run_1.json"
    assert out_file.exists()

    data = json.loads(out_file.read_text())
    assert data["sanitized"] is True
    assert data["receptor_info"]["uniprot_entry_name"] == "OPSD_BOVIN"
