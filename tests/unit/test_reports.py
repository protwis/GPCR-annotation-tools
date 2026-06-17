"""Tests for operational reports."""

from __future__ import annotations

import json
from typing import Any

import pytest

from gpcr_tools import reports
from gpcr_tools.config import get_config, reset_config


@pytest.fixture()
def cfg(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("GPCR_WORKSPACE", str(tmp_path))
    reset_config()
    return get_config()


def _write_download_log(cfg: Any, mapping: dict) -> None:
    cfg.download_log_file.parent.mkdir(parents=True, exist_ok=True)
    cfg.download_log_file.write_text(json.dumps(mapping), encoding="utf-8")


def _write_validation(cfg: Any, pdb: str, **fields: Any) -> None:
    vdir = cfg.aggregated_dir / "validation_logs"
    vdir.mkdir(parents=True, exist_ok=True)
    (vdir / f"{pdb}_validation.json").write_text(json.dumps(fields), encoding="utf-8")


class TestPdfCoverage:
    def test_empty(self, cfg: Any) -> None:
        assert "no download log" in reports.report_pdf_coverage()

    def test_groups_by_status(self, cfg: Any) -> None:
        _write_download_log(
            cfg,
            {
                "A": {"status": "success_pdf_downloaded"},
                "B": {"status": "success_pdf_downloaded"},
                "C": {"status": "fallback_paywalled"},
            },
        )
        out = reports.report_pdf_coverage()
        assert "3 PDB" in out
        assert "success_pdf_downloaded" in out
        assert "fallback_paywalled" in out


class TestFullAudit:
    def test_empty(self, cfg: Any) -> None:
        assert "no validation logs" in reports.report_full_audit()

    def test_summarises_warnings_and_conflicts(self, cfg: Any) -> None:
        _write_validation(
            cfg, "AAA", critical_warnings=["w1"], algo_conflicts=[], chimera_status="success"
        )
        _write_validation(
            cfg, "BBB", critical_warnings=[], algo_conflicts=["c1"], chimera_status="skipped"
        )
        out = reports.report_full_audit()
        assert "2 PDB" in out
        assert "AAA" in out  # has a critical warning
        assert "BBB" in out  # has an algo conflict
        assert "success" in out
        assert "skipped" in out


class TestTailAnalysis:
    def test_empty(self, cfg: Any) -> None:
        assert "no validation logs" in reports.report_tail_analysis()

    def test_score_distribution_and_flagging(self, cfg: Any) -> None:
        _write_validation(cfg, "AAA", chimera_score=11, chimera_status="success")  # clean
        _write_validation(cfg, "BBB", chimera_score=5, chimera_status="success")  # sub-anchor
        _write_validation(cfg, "CCC", chimera_score=11, chimera_status="conflict")  # non-ok
        out = reports.report_tail_analysis()
        assert "3 PDB" in out
        assert "score 11" in out
        assert "score 5" in out
        assert "Flagged for review (non-success or score < 8): 2" in out
        assert "BBB" in out
        assert "CCC" in out


def test_pdf_coverage_handles_non_dict_log(cfg: Any) -> None:
    """A corrupt-but-parseable (non-dict) download log must not crash the report."""
    _write_download_log(cfg, ["not", "a", "dict"])
    assert "no download log" in reports.report_pdf_coverage()  # degrades, no crash


def test_pdf_coverage_none_status_renders_unknown(cfg: Any) -> None:
    """A null status renders as 'unknown', not the literal string 'None'."""
    _write_download_log(cfg, {"A": {"status": None}})
    assert "unknown" in reports.report_pdf_coverage()


def test_tail_analysis_accepts_float_score(cfg: Any) -> None:
    """A float chimera score below 4 is shown and flagged, not silently ignored."""
    _write_validation(cfg, "FLT", chimera_score=3.5, chimera_status="partial")
    out = reports.report_tail_analysis()
    assert "3.5" in out
    assert "FLT" in out


# ---------------------------------------------------------------------------
# Run manifest
# ---------------------------------------------------------------------------


def _write_targets(cfg: Any, ids: list[str]) -> None:
    cfg.targets_file.parent.mkdir(parents=True, exist_ok=True)
    cfg.targets_file.write_text("\n".join(ids) + "\n", encoding="utf-8")


def _write_runs(cfg: Any, pdb: str, model: str, n: int) -> None:
    from gpcr_tools.config import model_run_subdir

    d = cfg.ai_results_dir / pdb / model_run_subdir(model)
    d.mkdir(parents=True, exist_ok=True)
    for i in range(1, n + 1):
        (d / f"run_{i}.json").write_text("{}", encoding="utf-8")


class TestRunManifest:
    def test_empty_workspace_does_not_crash(self, cfg: Any) -> None:
        """A manifest over an empty (mid-pipeline) workspace degrades to zero
        counts rather than raising."""
        manifest = reports.build_run_manifest(model_name="m", num_runs=10)
        assert manifest["targets"]["count"] == 0
        assert manifest["no_pdf"]["count"] == 0
        assert manifest["run_counts"]["incomplete_count"] == 0
        assert manifest["quality"]["validated_count"] == 0

    def test_targets_from_targets_file(self, cfg: Any) -> None:
        _write_targets(cfg, ["7W55", "8ABC", "# a comment", "9XYZ"])
        manifest = reports.build_run_manifest(model_name="m", num_runs=10)
        assert manifest["targets"]["count"] == 3
        assert manifest["targets"]["pdb_ids"] == ["7W55", "8ABC", "9XYZ"]

    def test_no_pdf_grouped_by_reason(self, cfg: Any) -> None:
        _write_download_log(
            cfg,
            {
                "AAA": {"status": "fallback_paywalled", "file_path": None},
                "BBB": {"status": "failed_no_doi", "file_path": None},
                "CCC": {"status": "fallback_paywalled", "file_path": None},
                # DDD has a real PDF on disk -> excluded from no-PDF.
                "DDD": {"status": "success_pdf_downloaded", "file_path": None},
            },
        )
        cfg.papers_dir.mkdir(parents=True, exist_ok=True)
        (cfg.papers_dir / "DDD.pdf").write_text("%PDF", encoding="utf-8")
        manifest = reports.build_run_manifest(model_name="m", num_runs=10)
        no_pdf = manifest["no_pdf"]
        assert no_pdf["count"] == 3
        assert no_pdf["by_reason"]["fallback_paywalled"] == ["AAA", "CCC"]
        assert no_pdf["by_reason"]["failed_no_doi"] == ["BBB"]

    def test_target_absent_from_log_goes_to_not_in_download_log(self, cfg: Any) -> None:
        """A target that never entered the download log (never fetched) is
        accounted under 'not_in_download_log', not silently dropped."""
        _write_targets(cfg, ["AAA", "ZZZ"])  # ZZZ never fetched
        _write_download_log(cfg, {"AAA": {"status": "fallback_paywalled", "file_path": None}})
        cfg.papers_dir.mkdir(parents=True, exist_ok=True)
        no_pdf = reports.build_run_manifest(model_name="m", num_runs=10)["no_pdf"]
        assert no_pdf["by_reason"]["not_in_download_log"] == ["ZZZ"]
        assert no_pdf["count"] == 2  # AAA (paywalled) + ZZZ (never in the log)
        assert "success_pdf_downloaded" not in no_pdf["by_reason"]

    def test_incomplete_runs_detected(self, cfg: Any) -> None:
        _write_runs(cfg, "FULL", "m", 10)
        _write_runs(cfg, "PART", "m", 3)
        manifest = reports.build_run_manifest(model_name="m", num_runs=10)
        rc = manifest["run_counts"]
        assert rc["per_pdb"]["FULL"] == 10
        assert rc["per_pdb"]["PART"] == 3
        assert rc["incomplete_count"] == 1
        assert rc["incomplete"][0]["pdb_id"] == "PART"
        assert rc["incomplete"][0]["runs"] == 3

    def test_incomplete_reason_attributed_per_pdb_from_failed_job(self, cfg: Any) -> None:
        """A PDB that was in a FAILED job gets that reason; an incomplete PDB NOT
        in any failed job degrades to 'unknown' (no blanket corpus-wide blame)."""
        _write_runs(cfg, "PART", "m", 2)  # member of the failed job
        _write_runs(cfg, "OTHER", "m", 2)  # incomplete, but not in any failed job
        cfg.batch_jobs_registry_file.parent.mkdir(parents=True, exist_ok=True)
        cfg.batch_jobs_registry_file.write_text(
            json.dumps(
                {
                    "version": 1,
                    "jobs": {
                        "batchJobs/x": {
                            "job_name": "batchJobs/x",
                            "status": "failed",
                            "detect_advisory": {"PART": []},  # PART was in this job
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        manifest = reports.build_run_manifest(model_name="m", num_runs=10)
        reasons = {i["pdb_id"]: i["reason"] for i in manifest["run_counts"]["incomplete"]}
        assert reasons["PART"] == "batch_job_failed_or_expired"
        assert reasons["OTHER"] == "unknown"

    def test_quality_acceptable_vs_gated_and_type_breakdown(self, cfg: Any) -> None:
        _write_validation(cfg, "CLEAN", critical_warnings=[], algo_conflicts=[])
        _write_validation(
            cfg,
            "GATED",
            critical_warnings=["[HALLUCINATION ALERT] at x: bad", "[UNANNOTATED CHAIN] at y"],
            algo_conflicts=["[ALGO WARNING] at chimera_analysis: foo"],
        )
        manifest = reports.build_run_manifest(model_name="m", num_runs=10)
        q = manifest["quality"]
        assert q["validated_count"] == 2
        assert q["acceptable"] == ["CLEAN"]
        assert q["gated"] == ["GATED"]
        assert q["warning_types"]["HALLUCINATION ALERT"] == 1
        assert q["warning_types"]["UNANNOTATED CHAIN"] == 1
        assert q["conflict_types"]["ALGO WARNING"] == 1

    def test_provenance_records_version_model_prompt(self, cfg: Any, monkeypatch: Any) -> None:
        monkeypatch.setenv("GPCR_CODE_VERSION", "deadbee")
        # A default prompt file makes the prompt id resolvable.
        cfg.default_prompt_file.parent.mkdir(parents=True, exist_ok=True)
        cfg.default_prompt_file.write_text("prompt", encoding="utf-8")
        manifest = reports.build_run_manifest(model_name="my-model", num_runs=7)
        prov = manifest["provenance"]
        assert prov["code_version"] == "deadbee"
        assert prov["model"] == "my-model"
        assert prov["prompt"] == cfg.default_prompt_file.stem
        assert prov["expected_runs"] == 7

    def test_write_run_manifest_emits_both_files(self, cfg: Any) -> None:
        _write_targets(cfg, ["7W55"])
        json_path, md_path = reports.write_run_manifest(model_name="m", num_runs=10)
        assert json_path.is_file()
        assert md_path.is_file()
        data = json.loads(json_path.read_text())
        assert data["targets"]["count"] == 1
        md = md_path.read_text()
        assert "# GPCR annotation run manifest" in md
        assert "## Targets" in md

    def test_report_run_manifest_summary(self, cfg: Any) -> None:
        _write_targets(cfg, ["7W55", "8ABC"])
        out = reports.report_run_manifest()
        assert "Run manifest written" in out
        assert "Targets: 2" in out

    def test_render_id_list_truncates(self) -> None:
        ids = [f"P{i:03d}" for i in range(60)]
        rendered = reports._render_id_list(ids, limit=10)
        assert "+50 more" in rendered
        assert reports._render_id_list([]) == "(none)"
