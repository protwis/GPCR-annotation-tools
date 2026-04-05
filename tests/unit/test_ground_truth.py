"""Tests for ground truth injection (Epic 2)."""

from __future__ import annotations

import copy
from typing import Any

from gpcr_tools.aggregator.ground_truth import inject_ground_truth


def _make_enriched(
    *,
    method: str | None = None,
    em_resolution: float | None = None,
    xray_resolution: float | None = None,
    release_date: str | None = None,
) -> dict[str, Any]:
    """Build a minimal enriched_entry dict for testing."""
    entry: dict[str, Any] = {}
    if method is not None:
        entry["exptl"] = [{"method": method}]
    if em_resolution is not None:
        entry["em_3d_reconstruction"] = [{"resolution": em_resolution}]
    if xray_resolution is not None:
        entry["refine"] = [{"ls_d_res_high": xray_resolution}]
    if release_date is not None:
        entry["rcsb_accession_info"] = {"initial_release_date": release_date}
    return entry


class TestInjectGroundTruth:
    def test_em_resolution(self) -> None:
        data: dict[str, Any] = {"structure_info": {}}
        enriched = _make_enriched(method="ELECTRON MICROSCOPY", em_resolution=3.2)
        inject_ground_truth("TEST", data, enriched)
        assert data["structure_info"]["method"] == "ELECTRON MICROSCOPY"
        assert data["structure_info"]["resolution"] == 3.2

    def test_xray_resolution(self) -> None:
        data: dict[str, Any] = {"structure_info": {}}
        enriched = _make_enriched(method="X-RAY DIFFRACTION", xray_resolution=1.8)
        inject_ground_truth("TEST", data, enriched)
        assert data["structure_info"]["method"] == "X-RAY DIFFRACTION"
        assert data["structure_info"]["resolution"] == 1.8

    def test_em_preferred_over_xray(self) -> None:
        data: dict[str, Any] = {"structure_info": {}}
        enriched = _make_enriched(em_resolution=3.5, xray_resolution=2.0)
        inject_ground_truth("TEST", data, enriched)
        assert data["structure_info"]["resolution"] == 3.5

    def test_release_date_parsing(self) -> None:
        data: dict[str, Any] = {"structure_info": {}}
        enriched = _make_enriched(release_date="2023-06-15T00:00:00+00:00")
        inject_ground_truth("TEST", data, enriched)
        assert data["structure_info"]["release_date"] == "2023-06-15"

    def test_release_date_no_time(self) -> None:
        data: dict[str, Any] = {"structure_info": {}}
        enriched = _make_enriched(release_date="2023-06-15")
        inject_ground_truth("TEST", data, enriched)
        assert data["structure_info"]["release_date"] == "2023-06-15"

    def test_missing_fields_not_injected(self) -> None:
        data: dict[str, Any] = {"structure_info": {"method": "original"}}
        enriched: dict[str, Any] = {}
        inject_ground_truth("TEST", data, enriched)
        assert data["structure_info"]["method"] == "original"
        assert "resolution" not in data["structure_info"]

    def test_creates_structure_info_if_missing(self) -> None:
        data: dict[str, Any] = {}
        enriched = _make_enriched(method="EM")
        inject_ground_truth("TEST", data, enriched)
        assert data["structure_info"]["method"] == "EM"

    def test_deepcopy_check(self) -> None:
        """Verify that original data is NOT mutated when caller uses deepcopy."""
        original: dict[str, Any] = {"structure_info": {"method": "OLD"}}
        working_copy = copy.deepcopy(original)
        enriched = _make_enriched(method="NEW")
        inject_ground_truth("TEST", working_copy, enriched)
        assert working_copy["structure_info"]["method"] == "NEW"
        assert original["structure_info"]["method"] == "OLD"

    def test_non_dict_best_run(self) -> None:
        """Should log error and return without crashing."""
        inject_ground_truth("TEST", "not a dict", {})  # type: ignore[arg-type]

    def test_null_method_in_enriched(self) -> None:
        """Blood Lesson 1: explicit null method should not inject 'None'."""
        data: dict[str, Any] = {"structure_info": {}}
        enriched: dict[str, Any] = {"exptl": [{"method": None}]}
        inject_ground_truth("TEST", data, enriched)
        assert "method" not in data["structure_info"]

    def test_null_resolution_in_enriched(self) -> None:
        data: dict[str, Any] = {"structure_info": {}}
        enriched: dict[str, Any] = {"em_3d_reconstruction": [{"resolution": None}]}
        inject_ground_truth("TEST", data, enriched)
        assert "resolution" not in data["structure_info"]

    def test_null_rcsb_accession_info(self) -> None:
        """Blood Lesson 1: explicit null for rcsb_accession_info."""
        data: dict[str, Any] = {"structure_info": {}}
        enriched: dict[str, Any] = {"rcsb_accession_info": None}
        inject_ground_truth("TEST", data, enriched)
        assert "release_date" not in data["structure_info"]


class TestEnrichedLoader:
    """Tests for enriched_loader.py (placed here for convenience)."""

    def test_valid_load(self, tmp_path: Any, monkeypatch: Any) -> None:
        import json

        from gpcr_tools.aggregator.enriched_loader import load_enriched_data
        from gpcr_tools.config import reset_config

        monkeypatch.setenv("GPCR_WORKSPACE", str(tmp_path))
        reset_config()
        enriched_dir = tmp_path / "enriched"
        enriched_dir.mkdir()
        entry_data = {"polymer_entities": []}
        (enriched_dir / "5G53.json").write_text(
            json.dumps({"data": {"entry": entry_data}}), encoding="utf-8"
        )
        result = load_enriched_data("5G53")
        assert result == entry_data

    def test_missing_file(self, tmp_path: Any, monkeypatch: Any) -> None:
        from gpcr_tools.aggregator.enriched_loader import load_enriched_data
        from gpcr_tools.config import reset_config

        monkeypatch.setenv("GPCR_WORKSPACE", str(tmp_path))
        reset_config()
        (tmp_path / "enriched").mkdir()
        result = load_enriched_data("MISSING")
        assert result is None

    def test_malformed_json(self, tmp_path: Any, monkeypatch: Any) -> None:
        from gpcr_tools.aggregator.enriched_loader import load_enriched_data
        from gpcr_tools.config import reset_config

        monkeypatch.setenv("GPCR_WORKSPACE", str(tmp_path))
        reset_config()
        enriched_dir = tmp_path / "enriched"
        enriched_dir.mkdir()
        (enriched_dir / "BAD.json").write_text("{bad", encoding="utf-8")
        result = load_enriched_data("BAD")
        assert result is None

    def test_missing_data_entry(self, tmp_path: Any, monkeypatch: Any) -> None:
        import json

        from gpcr_tools.aggregator.enriched_loader import load_enriched_data
        from gpcr_tools.config import reset_config

        monkeypatch.setenv("GPCR_WORKSPACE", str(tmp_path))
        reset_config()
        enriched_dir = tmp_path / "enriched"
        enriched_dir.mkdir()
        (enriched_dir / "NOENT.json").write_text(
            json.dumps({"data": {"other": {}}}), encoding="utf-8"
        )
        result = load_enriched_data("NOENT")
        assert result is None

    def test_null_data_key(self, tmp_path: Any, monkeypatch: Any) -> None:
        """Blood Lesson 5: explicit null 'data' key."""
        import json

        from gpcr_tools.aggregator.enriched_loader import load_enriched_data
        from gpcr_tools.config import reset_config

        monkeypatch.setenv("GPCR_WORKSPACE", str(tmp_path))
        reset_config()
        enriched_dir = tmp_path / "enriched"
        enriched_dir.mkdir()
        (enriched_dir / "NULL.json").write_text(json.dumps({"data": None}), encoding="utf-8")
        result = load_enriched_data("NULL")
        assert result is None

    def test_empty_entry_is_valid(self, tmp_path: Any, monkeypatch: Any) -> None:
        """Blood Lesson 5: empty dict {} is valid enriched data."""
        import json

        from gpcr_tools.aggregator.enriched_loader import load_enriched_data
        from gpcr_tools.config import reset_config

        monkeypatch.setenv("GPCR_WORKSPACE", str(tmp_path))
        reset_config()
        enriched_dir = tmp_path / "enriched"
        enriched_dir.mkdir()
        (enriched_dir / "EMPTY.json").write_text(
            json.dumps({"data": {"entry": {}}}), encoding="utf-8"
        )
        result = load_enriched_data("EMPTY")
        assert result == {}
        assert result is not None  # NOT None — empty dict is valid
