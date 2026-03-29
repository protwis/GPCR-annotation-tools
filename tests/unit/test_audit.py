"""Tests for the audit trail logger."""

import json


class TestLogAuditTrail:
    def test_creates_entry(self, configure_paths):
        """An audit trail entry should be appended to the JSONL file."""
        _, output_dir = configure_paths

        from gpcr_tools.csv_generator.audit import log_audit_trail

        log_audit_trail("TEST1", "receptor_info.chain_id", "accept", "A", "A")

        audit_file = output_dir / "audit_trail.jsonl"
        assert audit_file.exists()

        with open(audit_file) as f:
            lines = f.readlines()
        assert len(lines) == 1

        entry = json.loads(lines[0])
        assert entry["pdb_id"] == "TEST1"
        assert entry["field_path"] == "receptor_info.chain_id"
        assert entry["action"] == "accept"

    def test_multiple_entries(self, configure_paths):
        """Multiple audit entries should be appended sequentially."""
        _, output_dir = configure_paths

        from gpcr_tools.csv_generator.audit import log_audit_trail

        log_audit_trail("TEST1", "field_a", "accept", "x", "x")
        log_audit_trail("TEST1", "field_b", "edit", "old", "new")
        log_audit_trail("TEST2", "field_c", "skip", None, None)

        audit_file = output_dir / "audit_trail.jsonl"
        with open(audit_file) as f:
            lines = f.readlines()
        assert len(lines) == 3

        # Verify each line is valid JSON
        for line in lines:
            entry = json.loads(line)
            assert "pdb_id" in entry
            assert "timestamp" in entry
