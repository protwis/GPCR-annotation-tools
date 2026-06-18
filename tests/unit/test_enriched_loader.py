import json

from gpcr_tools.aggregator.enriched_loader import enriched_is_incomplete, load_enriched_data
from gpcr_tools.config import get_config, reset_config


def _write(cfg, pdb, obj):
    (cfg.enriched_dir / f"{pdb}.json").write_text(json.dumps(obj))


def test_enriched_is_incomplete(tmp_path, monkeypatch):
    """The top-level transient-outage marker is detected; a complete record and a
    missing file are not flagged. The marker lives OUTSIDE data.entry, so
    load_enriched_data still returns the entry (the consumer is what refuses)."""
    monkeypatch.setenv("GPCR_ENRICHED_PATH", str(tmp_path / "enriched"))
    reset_config()
    cfg = get_config()
    cfg.enriched_dir.mkdir(parents=True)

    _write(cfg, "AAAA", {"data": {"entry": {"rcsb_id": "AAAA"}}})
    _write(cfg, "BBBB", {"_enrich_incomplete": True, "data": {"entry": {"rcsb_id": "BBBB"}}})

    assert enriched_is_incomplete("AAAA") is False
    assert enriched_is_incomplete("BBBB") is True
    assert enriched_is_incomplete("MISSING") is False
    # The loader itself does not gate — it still returns the entry of a marked record.
    assert load_enriched_data("BBBB") == {"rcsb_id": "BBBB"}


def test_run_detect_skips_incomplete_enrichment(tmp_path, monkeypatch):
    """detect refuses an enrichment marked incomplete: returns no signals and
    writes no detect file (so a re-run after re-enrich recomputes it)."""
    from gpcr_tools.detector.stage import run_detect

    monkeypatch.setenv("GPCR_WORKSPACE", str(tmp_path))
    reset_config()
    cfg = get_config()
    cfg.enriched_dir.mkdir(parents=True, exist_ok=True)
    cfg.detect_dir.mkdir(parents=True, exist_ok=True)
    (cfg.enriched_dir / "7W55.json").write_text(
        json.dumps({"_enrich_incomplete": True, "data": {"entry": {"rcsb_id": "7W55"}}})
    )

    out = run_detect("7W55", skip_api_checks=True)

    assert out == []
    assert not (cfg.detect_dir / "7W55.json").exists()
