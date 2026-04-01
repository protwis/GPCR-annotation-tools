"""Real-data gating tests for app.py mode availability logic.

Verifies that the accept-all / fix-mode / review gating behaves correctly
with audited real fixtures, without running the full interactive UI loop.
The gating logic under test (from app.py) is:

    oligo = main_data.get("oligomer_analysis") or {}
    inject_oligomer_alerts(oligo, validation_data)
    has_crit_issues = validation_data.get("critical_warnings") or validation_data.get("algo_conflicts")
    accept_all_available = not has_crit_issues and not controversies
"""

from pathlib import Path

import pytest


def _compute_gating(
    pdb_id: str,
) -> tuple[bool, bool, dict, dict]:
    """Replicate app.py gating logic for a real fixture.

    Returns (accept_all_available, has_crit_issues, controversies, validation_data).
    """
    from gpcr_tools.csv_generator.data_loader import load_pdb_data
    from gpcr_tools.csv_generator.validation_display import inject_oligomer_alerts

    main_data, controversies, validation_data = load_pdb_data(pdb_id)
    assert main_data is not None

    oligo = main_data.get("oligomer_analysis") or {}
    inject_oligomer_alerts(oligo, validation_data)

    has_crit_issues = bool(
        validation_data.get("critical_warnings") or validation_data.get("algo_conflicts")
    )
    accept_all_available = not has_crit_issues and not controversies
    return accept_all_available, has_crit_issues, controversies, validation_data


# ── RP-3.2: Blocked By Critical Warnings ────────────────────────────────


class TestBlockedByCriticalWarnings:
    """Fixtures that should block accept-all due to critical issues."""

    @pytest.mark.parametrize(
        "pdb_id",
        ["9M88", "9AS1", "9EJZ"],
    )
    def test_accept_all_disabled(self, pdb_id: str, real_pdb_workspace: Path) -> None:
        accept_all, has_crit, _, _ = _compute_gating(pdb_id)
        assert has_crit is True, f"{pdb_id} should have critical issues"
        assert accept_all is False, f"{pdb_id} should not have accept-all available"

    def test_9m88_mode_choices_exclude_a(self, real_pdb_workspace: Path) -> None:
        accept_all, _, _controversies, _ = _compute_gating("9M88")
        choices = ["r", "s", "f"]
        if accept_all:
            choices.insert(0, "a")
        assert "a" not in choices

    def test_9as1_has_both_criticals_and_controversies(self, real_pdb_workspace: Path) -> None:
        accept_all, has_crit, controversies, _ = _compute_gating("9AS1")
        assert has_crit is True
        assert len(controversies) > 0
        assert accept_all is False

    def test_9ejz_blocked_by_uniprot_clash(self, real_pdb_workspace: Path) -> None:
        _, has_crit, controversies, validation_data = _compute_gating("9EJZ")
        assert has_crit is True
        assert len(controversies) == 0
        warnings_text = " ".join(validation_data.get("critical_warnings", []))
        assert "UNIPROT_CLASH" in warnings_text

    def test_9nor_blocked_by_ghost_ligand(self, real_pdb_workspace: Path) -> None:
        accept_all, has_crit, _, validation_data = _compute_gating("9NOR")
        assert has_crit is True
        assert accept_all is False
        warnings_text = " ".join(validation_data.get("critical_warnings", [])).lower()
        assert "ghost ligand" in warnings_text or "ghost_ligand" in warnings_text

    def test_8tii_blocked_by_incomplete_7tm_injection(self, real_pdb_workspace: Path) -> None:
        accept_all, has_crit, _, validation_data = _compute_gating("8TII")
        assert has_crit is True
        assert accept_all is False
        warnings_text = " ".join(validation_data.get("critical_warnings", []))
        assert "INCOMPLETE" in warnings_text


# ── RP-3.3: Blocked By Controversy Only ─────────────────────────────────


class TestBlockedByControversyOnly:
    """Fixtures that block accept-all via controversies, not criticals."""

    def test_9blw_controversy_blocks_accept_all(self, real_pdb_workspace: Path) -> None:
        accept_all, has_crit, controversies, _ = _compute_gating("9BLW")
        assert has_crit is False, "9BLW should have no critical issues"
        assert len(controversies) > 0, "9BLW should have controversies"
        assert accept_all is False

    def test_9blw_mode_choices_exclude_a(self, real_pdb_workspace: Path) -> None:
        accept_all, _, _, _ = _compute_gating("9BLW")
        choices = ["r", "s", "f"]
        if accept_all:
            choices.insert(0, "a")
        assert "a" not in choices

    def test_9blw_blocker_is_controversy_not_critical(self, real_pdb_workspace: Path) -> None:
        _, has_crit, controversies, _ = _compute_gating("9BLW")
        assert has_crit is False
        assert len(controversies) == 2
        assert set(controversies.keys()) == {
            "auxiliary_proteins[Nanobody-35].type.evidence.source",
            "ligands[None].name",
        }


# ── RP-3.4: Clean Accept-All Available ──────────────────────────────────


class TestCleanAcceptAll:
    """Fixtures where accept-all should be available (5G53)."""

    def test_5g53_accept_all_available(self, real_pdb_workspace: Path) -> None:
        accept_all, has_crit, controversies, _ = _compute_gating("5G53")
        assert has_crit is False, "5G53 should have no critical issues"
        assert len(controversies) == 0, "5G53 should have no controversies"
        assert accept_all is True

    def test_5g53_mode_choices_include_a(self, real_pdb_workspace: Path) -> None:
        _accept_all, has_crit, controversies, _ = _compute_gating("5G53")
        choices = ["r", "s", "f"]
        if not has_crit and not controversies:
            choices.insert(0, "a")
        assert "a" in choices
        assert choices[0] == "a"

    def test_5g53_validation_data_clean(self, real_pdb_workspace: Path) -> None:
        _, _, _, validation_data = _compute_gating("5G53")
        assert validation_data.get("critical_warnings") == []
        assert validation_data.get("algo_conflicts") == []
