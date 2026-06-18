"""Tests for the v3.1 workspace configuration model."""

from pathlib import Path

import pytest

from gpcr_tools.config import (
    INCIDENTAL_CANDIDATES,
    LIGAND_EXCLUDE_LIST,
    SOFT_FIELD_KEYS,
    WorkspaceConfig,
    ensure_alert_prefix,
    get_config,
    list_item_identity,
    reset_config,
    safe_name_normalize,
)


class TestEnsureAlertPrefix:
    """The "[TYPE]" prefix must appear exactly once regardless of whether the
    incoming message already carries it (current validator output) or not
    (older recorded data)."""

    def test_prefixed_message_kept_once(self) -> None:
        # Current validator format already carries the prefix.
        msg = "[MISSED_PROTOMER] at 'oligomer_analysis': Missed: ['A']."
        out = ensure_alert_prefix("MISSED_PROTOMER", msg)
        assert out == msg
        assert out.count("[MISSED_PROTOMER]") == 1

    def test_bare_message_gets_prefix(self) -> None:
        # Older recorded data stored the bare description, no prefix.
        out = ensure_alert_prefix("MISSED_PROTOMER", "GPCR roster has chains ['A', 'B'].")
        assert out == "[MISSED_PROTOMER] GPCR roster has chains ['A', 'B']."
        assert out.count("[MISSED_PROTOMER]") == 1

    def test_empty_message_returns_prefix(self) -> None:
        assert ensure_alert_prefix("HALLUCINATION", "") == "[HALLUCINATION]"
        assert ensure_alert_prefix("HALLUCINATION", None) == "[HALLUCINATION]"

    def test_leading_whitespace_prefixed_message(self) -> None:
        out = ensure_alert_prefix("HALLUCINATION", "  [HALLUCINATION] at 'x': y")
        assert out.count("[HALLUCINATION]") == 1
        assert out.startswith("[HALLUCINATION]")


class TestListItemIdentity:
    """site_ref keeps two same-component ligand entries distinct in voting."""

    def test_plain_component_key(self) -> None:
        assert list_item_identity({"chem_comp_id": "CLR"}, "chem_comp_id", 0) == "CLR"

    def test_distinct_sites_do_not_collapse(self) -> None:
        a = list_item_identity(
            {"chem_comp_id": "A1AEI", "site_ref": "orthosteric"}, "chem_comp_id", 0
        )
        b = list_item_identity(
            {"chem_comp_id": "A1AEI", "site_ref": "extracellular_vestibule"}, "chem_comp_id", 1
        )
        assert a != b

    def test_unknown_site_ref_ignored(self) -> None:
        # An 'unknown' (or absent) site_ref must not change the identity.
        assert (
            list_item_identity({"chem_comp_id": "CLR", "site_ref": "unknown"}, "chem_comp_id", 0)
            == "CLR"
        )

    def test_keyless_ligand_name_variants_merge(self) -> None:
        # A keyless ligand (no component id) keyed by name: a trailing
        # parenthetical annotation on one entity must not split it into a
        # second group.
        plain = list_item_identity(
            {"chem_comp_id": "None", "name": "Stalk peptide"}, "chem_comp_id", 0
        )
        annotated = list_item_identity(
            {"chem_comp_id": "None", "name": "Stalk peptide (tethered agonist)"},
            "chem_comp_id",
            1,
        )
        assert plain == annotated

    def test_keyless_ligand_trailing_digits_stay_distinct(self) -> None:
        # Trailing digits are load-bearing: two different numbered compounds
        # must key apart even after normalization.
        a = list_item_identity({"chem_comp_id": "None", "name": "Compound 28"}, "chem_comp_id", 0)
        b = list_item_identity({"chem_comp_id": "None", "name": "Compound 29"}, "chem_comp_id", 1)
        assert a != b


class TestSafeNameNormalize:
    """The SAFE (rule-based, no fuzzy) name normalizer for grouping keys."""

    def test_case_insensitive(self) -> None:
        assert safe_name_normalize("BRIL") == safe_name_normalize("bril")

    def test_strips_one_trailing_parenthetical(self) -> None:
        assert safe_name_normalize("Stalk peptide (tethered agonist)") == "stalk peptide"

    def test_collapses_hyphen_underscore_asterisk_whitespace(self) -> None:
        assert (
            safe_name_normalize("anti-Fab")
            == safe_name_normalize("anti Fab")
            == safe_name_normalize("anti_fab")
            == safe_name_normalize("anti**Fab")
            == "anti fab"
        )

    def test_greek_letters_folded_to_words(self) -> None:
        assert safe_name_normalize("Gα") == "galpha"
        assert safe_name_normalize("Gβ") == "gbeta"
        assert safe_name_normalize("Gγ") == "ggamma"

    def test_trailing_digits_preserved(self) -> None:
        # The receptor subtype digit is load-bearing and must survive.
        assert safe_name_normalize("mGlu2") != safe_name_normalize("mGlu7")
        assert safe_name_normalize("mGlu2") == "mglu2"


class TestAuxiliaryProteinIdentity:
    """Auxiliary proteins key on normalized name PLUS a chain-set suffix, so a
    reused label never over-merges entities on disjoint chains."""

    def test_name_variants_same_chain_merge(self) -> None:
        # Case + asterisk + trailing parenthetical noise on one entity, same
        # chain -> one identity. (Separators — hyphen, underscore, whitespace,
        # asterisk — all collapse to a single space, so punctuation/spacing
        # noise around the same words folds to one stem.)
        a = list_item_identity({"name": "anti-Fab Nanobody", "chain_id": "K"}, "name", 0)
        b = list_item_identity({"name": "Anti-Fab  nanobody", "chain_id": "K"}, "name", 1)
        c = list_item_identity({"name": "anti-Fab nanobody (Nb)", "chain_id": "K"}, "name", 2)
        d = list_item_identity({"name": "anti_Fab*nanobody", "chain_id": "K"}, "name", 3)
        assert a == b == c == d

    def test_disjoint_chains_do_not_merge(self) -> None:
        # Same normalized name on disjoint chains -> distinct identities (a
        # model reusing one label, e.g. "BRIL", for a fusion / Fab / nanobody).
        fusion = list_item_identity({"name": "BRIL", "chain_id": "A"}, "name", 0)
        fab = list_item_identity({"name": "BRIL", "chain_id": "H, L"}, "name", 1)
        nanobody = list_item_identity({"name": "BRIL", "chain_id": "K"}, "name", 2)
        assert len({fusion, fab, nanobody}) == 3

    def test_no_chain_id_falls_back_to_name_only(self) -> None:
        # With no chain recorded, the identity is the normalized name alone (no
        # "|ch:" suffix), so two same-named entries that both omit a chain do
        # merge into one vote group.
        a = list_item_identity({"name": "BRIL"}, "name", 0)
        b = list_item_identity({"name": "bril"}, "name", 1)
        assert a == b == "bril"

    def test_chain_id_and_no_chain_id_do_not_merge(self) -> None:
        # A run that records a chain and a run that omits it produce different
        # identities — an accepted under-merge (separate vote groups), never an
        # over-merge.
        with_chain = list_item_identity({"name": "BRIL", "chain_id": "A"}, "name", 0)
        without_chain = list_item_identity({"name": "BRIL"}, "name", 1)
        assert with_chain != without_chain

    def test_chain_set_order_and_spacing_invariant(self) -> None:
        # The chain string varies in order/spacing across runs; the same chain
        # SET must yield the same identity.
        assert (
            list_item_identity({"name": "BRIL", "chain_id": "H, L"}, "name", 0)
            == list_item_identity({"name": "BRIL", "chain_id": "L,H"}, "name", 1)
            == list_item_identity({"name": "BRIL", "chain_id": " L , H "}, "name", 2)
        )

    def test_distinct_names_same_chain_stay_distinct(self) -> None:
        # Trailing-digit receptors and unrelated soluble partners must never
        # collapse even when modelled on the same chain.
        assert list_item_identity(
            {"name": "mGlu2", "chain_id": "A"}, "name", 0
        ) != list_item_identity({"name": "mGlu7", "chain_id": "A"}, "name", 1)
        assert list_item_identity(
            {"name": "FKBP", "chain_id": "A"}, "name", 0
        ) != list_item_identity({"name": "FRB", "chain_id": "A"}, "name", 1)


class TestSoftFieldKeys:
    """Free-text justification prose is excluded from cross-run voting."""

    def test_justification_and_evidence_are_soft(self) -> None:
        assert "site_ref_justification" in SOFT_FIELD_KEYS
        assert "evidence" in SOFT_FIELD_KEYS


class TestLigandLists:
    """Invariants on the curated exclude / incidental lists."""

    def test_exclude_list_keys_on_pdb_codes_not_chemical_names(self) -> None:
        # The lists key on the PDB three-letter chem_comp id, so a chemical's
        # human name never matches the enriched data. DMSO and HEPES must be
        # carried by their actual PDB codes (DMS, EPE), not their names.
        assert "DMSO" not in LIGAND_EXCLUDE_LIST
        assert "HEPES" not in LIGAND_EXCLUDE_LIST
        assert "DMS" in LIGAND_EXCLUDE_LIST
        assert "EPE" in LIGAND_EXCLUDE_LIST

    def test_detergents_and_matrix_lipids_excluded(self) -> None:
        # Detergents, LCP matrix lipids and PEG oligomers are never functional
        # ligands and must be stripped before the model sees them.
        for code in ("OLC", "OLB", "Y01", "BOG", "SOG", "1PE", "P6G"):
            assert code in LIGAND_EXCLUDE_LIST

    def test_functional_lipids_are_incidental_never_excluded(self) -> None:
        # Free fatty acids and other biological lipids can be the endogenous
        # agonist, so they are model-judged (incidental), never hard-excluded.
        for code in ("OLA", "S1P", "HXA", "NKP", "ACT", "SIN"):
            assert code in INCIDENTAL_CANDIDATES
            assert code not in LIGAND_EXCLUDE_LIST

    def test_incidental_override_member_present(self) -> None:
        # The incidental fork un-strips a member that is also on the exclude
        # list (PLM); that override must keep at least one such member so the
        # bypass path stays exercised.
        assert "PLM" in (INCIDENTAL_CANDIDATES & LIGAND_EXCLUDE_LIST)


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
