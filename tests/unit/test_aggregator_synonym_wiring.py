"""Wiring guard: the PubChem synonym check must stay connected to aggregation.

The synonym check is opt-in -- ``validate_and_enrich_ligands`` only runs it when
given a ``synonym_cache``. These tests assert ``aggregate_pdb`` threads the cache
through when API checks are on, and suppresses it (passes ``None``) under
``--skip-api-checks``. Without them, a future edit that drops the ``synonym_cache=``
keyword or flips the gate would silently return the check to dormant and every
other test would still pass.
"""

from __future__ import annotations

from types import SimpleNamespace

import gpcr_tools.aggregator.runner as runner


class _StopAfterLigandStepError(Exception):
    """Raised by the spy to short-circuit the steps after ligand validation."""


def _stub_to_ligand_step(monkeypatch, recorder, cache_dir):
    """Stub everything up to and including step 5 so ``aggregate_pdb`` reaches the
    ligand-validation step, and replace ``validate_and_enrich_ligands`` with a spy
    that records the ``synonym_cache`` it received, then short-circuits the rest of
    the pipeline (the spy raises; ``aggregate_pdb`` catches it) so the test makes no
    network or disk calls. ``cache_dir`` backs the lazily-built polymer-features
    cache the single-PDB path now constructs before the ligand step.
    """
    monkeypatch.setattr(runner, "get_config", lambda: SimpleNamespace(cache_dir=cache_dir))
    monkeypatch.setattr("gpcr_tools.workspace.validate_contract", lambda cfg: None)
    monkeypatch.setattr(runner, "load_ai_runs", lambda pdb_id: [{"ligands": []}])
    monkeypatch.setattr(runner, "get_majority_votes", lambda runs: ({}, {}))
    monkeypatch.setattr(runner, "select_best_run", lambda runs, votes: (0, {"ligands": []}))
    monkeypatch.setattr(runner, "load_enriched_data", lambda pdb_id: {})
    monkeypatch.setattr(runner, "inject_ground_truth", lambda *a, **k: None)

    def _spy(pdb_id, best_run_data, enriched, *, synonym_cache=None):
        recorder["called"] = True
        recorder["synonym_cache"] = synonym_cache
        raise _StopAfterLigandStepError

    monkeypatch.setattr(runner, "validate_and_enrich_ligands", _spy)


def test_synonym_cache_threaded_when_api_checks_on(monkeypatch, tmp_path):
    recorder: dict = {}
    _stub_to_ligand_step(monkeypatch, recorder, tmp_path)
    sentinel_cache = object()

    runner.aggregate_pdb("XXXX", skip_api_checks=False, synonym_cache=sentinel_cache)

    assert recorder.get("called") is True
    assert recorder["synonym_cache"] is sentinel_cache


def test_synonym_cache_suppressed_under_skip_api_checks(monkeypatch, tmp_path):
    recorder: dict = {}
    _stub_to_ligand_step(monkeypatch, recorder, tmp_path)
    sentinel_cache = object()

    runner.aggregate_pdb("XXXX", skip_api_checks=True, synonym_cache=sentinel_cache)

    assert recorder.get("called") is True
    assert recorder["synonym_cache"] is None


class _StopAfterOligomerStepError(Exception):
    """Raised by the oligomer spy to short-circuit the remaining pipeline steps."""


def test_single_pdb_path_lazily_builds_polymer_features_cache(monkeypatch, tmp_path):
    """The single-PDB path (caller passes no polymer cache) must still hand a real
    cache to ``analyze_oligomer``, so a genuine TM-fetch failure routes to a curator
    instead of failing open. Without this, the single-PDB and batch paths diverge
    (the original two-code-path foot-gun).
    """
    monkeypatch.setattr(runner, "get_config", lambda: SimpleNamespace(cache_dir=tmp_path))
    monkeypatch.setattr("gpcr_tools.workspace.validate_contract", lambda cfg: None)
    monkeypatch.setattr(runner, "load_ai_runs", lambda pdb_id: [{"ligands": []}])
    monkeypatch.setattr(runner, "get_majority_votes", lambda runs: ({}, {}))
    monkeypatch.setattr(runner, "select_best_run", lambda runs, votes: (0, {"ligands": []}))
    monkeypatch.setattr(runner, "load_enriched_data", lambda pdb_id: {})
    monkeypatch.setattr(runner, "enriched_is_incomplete", lambda pdb_id: False)
    monkeypatch.setattr(runner, "inject_ground_truth", lambda *a, **k: None)
    monkeypatch.setattr(runner, "validate_and_enrich_ligands", lambda *a, **k: [])
    monkeypatch.setattr(runner, "validate_receptor_identity", lambda *a, **k: [])
    monkeypatch.setattr(runner, "_coupling_protomer", lambda pdb_id: None)

    recorder: dict = {}

    def _oligomer_spy(
        pdb_id, best_run_data, enriched, *, coupling_chain=None, polymer_features_cache=None
    ):
        recorder["polymer_features_cache"] = polymer_features_cache
        raise _StopAfterOligomerStepError

    monkeypatch.setattr(runner, "analyze_oligomer", _oligomer_spy)

    runner.aggregate_pdb("XXXX", skip_api_checks=True)

    assert recorder["polymer_features_cache"] is not None
    assert isinstance(recorder["polymer_features_cache"], runner.PolymerFeaturesCache)
