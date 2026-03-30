"""Tests for the v3.1 workspace configuration model."""

from pathlib import Path

import pytest

from gpcr_tools.config import WorkspaceConfig, get_config, reset_config


@pytest.fixture(autouse=True)
def _clean_config(monkeypatch):
    """Ensure each test starts with a fresh config cache and no stale env."""
    env_vars = [
        "GPCR_WORKSPACE",
        "GPCR_RAW_PATH",
        "GPCR_ENRICHED_PATH",
        "GPCR_PAPERS_PATH",
        "GPCR_AI_RESULTS_PATH",
        "GPCR_AGGREGATED_PATH",
        "GPCR_OUTPUT_PATH",
        "GPCR_CACHE_PATH",
        "GPCR_STATE_PATH",
        "GPCR_TMP_PATH",
    ]
    for var in env_vars:
        monkeypatch.delenv(var, raising=False)
    reset_config()
    yield
    reset_config()


class TestWorkspaceDefaults:
    def test_default_workspace(self):
        cfg = get_config()
        assert cfg.workspace == Path("/workspace").resolve()

    def test_all_dirs_under_workspace(self):
        cfg = get_config()
        for name in (
            "raw_dir",
            "enriched_dir",
            "papers_dir",
            "ai_results_dir",
            "aggregated_dir",
            "output_dir",
            "cache_dir",
            "state_dir",
            "tmp_dir",
        ):
            assert getattr(cfg, name).is_relative_to(cfg.workspace)

    def test_derived_paths(self):
        cfg = get_config()
        assert cfg.contract_file == cfg.workspace / "contract" / "storage_contract.json"
        assert cfg.csv_output_dir == cfg.output_dir / "csv"
        assert cfg.audit_output_dir == cfg.output_dir / "audit"
        assert cfg.processed_log_file == cfg.state_dir / "processed_log.json"
        assert cfg.pipeline_runs_dir == cfg.state_dir / "pipeline_runs"

    def test_all_paths_absolute(self):
        cfg = get_config()
        for field in WorkspaceConfig.__dataclass_fields__:
            val = getattr(cfg, field)
            if isinstance(val, Path):
                assert val.is_absolute(), f"{field} is not absolute: {val}"


class TestWorkspaceOverride:
    def test_custom_workspace(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GPCR_WORKSPACE", str(tmp_path))
        reset_config()
        cfg = get_config()
        assert cfg.workspace == tmp_path.resolve()
        assert cfg.raw_dir == (tmp_path / "raw").resolve()

    def test_single_override(self, tmp_path, monkeypatch):
        ws = tmp_path / "ws"
        ws.mkdir()
        external = tmp_path / "external_cache"
        external.mkdir()
        monkeypatch.setenv("GPCR_WORKSPACE", str(ws))
        monkeypatch.setenv("GPCR_CACHE_PATH", str(external))
        reset_config()
        cfg = get_config()
        assert cfg.cache_dir == external.resolve()
        assert cfg.raw_dir == (ws / "raw").resolve()

    def test_multiple_overrides(self, tmp_path, monkeypatch):
        ws = tmp_path / "ws"
        ext_cache = tmp_path / "ext_cache"
        ext_state = tmp_path / "ext_state"
        for d in (ws, ext_cache, ext_state):
            d.mkdir()
        monkeypatch.setenv("GPCR_WORKSPACE", str(ws))
        monkeypatch.setenv("GPCR_CACHE_PATH", str(ext_cache))
        monkeypatch.setenv("GPCR_STATE_PATH", str(ext_state))
        reset_config()
        cfg = get_config()
        assert cfg.cache_dir == ext_cache.resolve()
        assert cfg.state_dir == ext_state.resolve()
        assert cfg.processed_log_file == ext_state.resolve() / "processed_log.json"


class TestConfigReset:
    def test_reset_gives_new_config(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GPCR_WORKSPACE", str(tmp_path / "a"))
        reset_config()
        cfg_a = get_config()

        monkeypatch.setenv("GPCR_WORKSPACE", str(tmp_path / "b"))
        reset_config()
        cfg_b = get_config()

        assert cfg_a.workspace != cfg_b.workspace

    def test_cached_without_reset(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GPCR_WORKSPACE", str(tmp_path / "a"))
        reset_config()
        cfg_a = get_config()

        monkeypatch.setenv("GPCR_WORKSPACE", str(tmp_path / "b"))
        cfg_b = get_config()  # no reset — should be cached

        assert cfg_a is cfg_b


class TestNoLegacyAccessors:
    """Legacy module-level names (DATA_DIR etc.) have been removed."""

    def test_unknown_attr_raises(self):
        from gpcr_tools import config

        with pytest.raises(AttributeError):
            _ = config.DATA_DIR

    def test_output_dir_attr_raises(self):
        from gpcr_tools import config

        with pytest.raises(AttributeError):
            _ = config.OUTPUT_DIR
