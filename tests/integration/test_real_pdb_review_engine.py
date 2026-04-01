"""Real-data review-engine interactive tests with prompt monkeypatching.

Tests prompt flows of review_toplevel_blocks using audited real fixtures
and deterministic scripted responses.  Monkeypatches target the module
import site: gpcr_tools.csv_generator.review_engine.{Prompt,Confirm}.
"""

import copy
from pathlib import Path
from typing import Any

import pytest

# ── RP-4.1: PromptScript Helper ─────────────────────────────────────────


class PromptScript:
    """Deterministic response queue for monkeypatched Prompt.ask / Confirm.ask.

    Each call pops the next response from the queue.  Raises AssertionError
    if the queue is exhausted unexpectedly or has leftover responses.
    """

    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self._call_log: list[dict[str, Any]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        assert self._responses, (
            f"PromptScript exhausted. Unexpected prompt call: args={args}, kwargs={kwargs}"
        )
        resp = self._responses.pop(0)
        self._call_log.append({"args": args, "kwargs": kwargs, "response": resp})
        return resp

    @property
    def call_count(self) -> int:
        return len(self._call_log)

    @property
    def remaining(self) -> int:
        return len(self._responses)

    def assert_exhausted(self) -> None:
        assert self.remaining == 0, (
            f"PromptScript has {self.remaining} unused response(s): {self._responses}"
        )


def _load_and_inject(pdb_id: str) -> tuple[dict, dict, dict]:
    """Load a real PDB and inject oligomer alerts."""
    from gpcr_tools.csv_generator.data_loader import load_pdb_data
    from gpcr_tools.csv_generator.validation_display import inject_oligomer_alerts

    main_data, controversies, validation_data = load_pdb_data(pdb_id)
    assert main_data is not None
    oligo = main_data.get("oligomer_analysis") or {}
    inject_oligomer_alerts(oligo, validation_data)
    return main_data, controversies, validation_data


# ── RP-4.2: Clean-Block Review Flow ─────────────────────────────────────


class TestCleanBlockReviewFlow:
    """Test that clean blocks route through Confirm.ask and preserve structure."""

    def test_5g53_accept_all_blocks_via_confirm(
        self,
        real_pdb_workspace: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """5G53 is clean: all 6 top-level blocks go through Confirm.ask."""
        from gpcr_tools.csv_generator.review_engine import review_toplevel_blocks

        main_data, controversies, validation_data = _load_and_inject("5G53")

        # 5G53 has 6 top-level blocks, all clean → each prompts Confirm.ask once
        confirm_script = PromptScript([True] * 6)
        monkeypatch.setattr(
            "gpcr_tools.csv_generator.review_engine.Confirm.ask",
            confirm_script,
        )

        result = review_toplevel_blocks(
            "5G53", copy.deepcopy(main_data), controversies, validation_data
        )

        assert result is not None
        confirm_script.assert_exhausted()

        expected_keys = {
            "structure_info",
            "receptor_info",
            "ligands",
            "signaling_partners",
            "auxiliary_proteins",
            "key_findings",
        }
        assert expected_keys.issubset(set(result.keys()))
        assert len(result.get("ligands", [])) == len(main_data.get("ligands", []))

    def test_5g53_decline_triggers_deep_review(
        self,
        real_pdb_workspace: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Declining a clean block enters deep review via review_node."""
        from gpcr_tools.csv_generator.review_engine import review_toplevel_blocks

        main_data, controversies, validation_data = _load_and_inject("5G53")

        # Accept first 5 blocks, decline key_findings → triggers review_node.
        # review_node on key_findings (a dict) will recurse; all leaves prompt
        # via Prompt.ask with choices ["y","e","s","q"].
        # We'll accept the first block normally, then quit on the review_node call.
        confirm_responses: list[bool] = [True, True, True, True, True, False]
        confirm_script = PromptScript(confirm_responses)

        # When review_node fires for key_findings, first child will prompt.
        # We respond "q" to quit, which propagates None → returns None.
        prompt_script = PromptScript(["q"])

        monkeypatch.setattr(
            "gpcr_tools.csv_generator.review_engine.Confirm.ask",
            confirm_script,
        )
        monkeypatch.setattr(
            "gpcr_tools.csv_generator.review_engine.Prompt.ask",
            prompt_script,
        )

        result = review_toplevel_blocks(
            "5G53", copy.deepcopy(main_data), controversies, validation_data
        )

        assert result is None
        confirm_script.assert_exhausted()


# ── RP-4.3: Fix-Mode Trivial Auto-Resolve ────────────────────────────────


class TestFixModeTrivialAutoResolve:
    """Test that fix_mode auto-resolves trivial controversies and auto-accepts clean blocks."""

    def test_9blw_fix_mode_auto_accepts_clean_and_trivial_blocks(
        self,
        real_pdb_workspace: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """9BLW in fix_mode: only ligands should prompt (name controversy is significant).

        Block behavior in fix_mode:
        - structure_info: no controversy, no warnings → auto-accept (no prompt)
        - receptor_info: no controversy, no warnings → auto-accept (no prompt)
        - ligands: controversy on ligands[None].name (significant) → prompts
        - signaling_partners: no controversy, no warnings → auto-accept (no prompt)
        - auxiliary_proteins: controversy on source (trivial) → auto-resolve + accept (no Confirm prompt)
        - key_findings: no controversy, no warnings → auto-accept (no prompt)
        """
        from gpcr_tools.csv_generator.review_engine import review_toplevel_blocks

        main_data, controversies, validation_data = _load_and_inject("9BLW")

        # The ligands block enters the dirty (has_contra=True) while-loop.
        # Prompt.ask for action: respond "a" (accept).
        prompt_script = PromptScript(["a"])
        confirm_script = PromptScript([])

        monkeypatch.setattr(
            "gpcr_tools.csv_generator.review_engine.Prompt.ask",
            prompt_script,
        )
        monkeypatch.setattr(
            "gpcr_tools.csv_generator.review_engine.Confirm.ask",
            confirm_script,
        )

        result = review_toplevel_blocks(
            "9BLW",
            copy.deepcopy(main_data),
            controversies,
            validation_data,
            fix_mode=True,
        )

        assert result is not None
        prompt_script.assert_exhausted()
        confirm_script.assert_exhausted()

        assert "structure_info" in result
        assert "receptor_info" in result
        assert "ligands" in result
        assert "signaling_partners" in result
        assert "auxiliary_proteins" in result


# ── RP-4.4: Fix-Mode Stops On Significant ────────────────────────────────


class TestFixModeStopsOnSignificant:
    """Test that fix_mode stops for blocks with significant issues."""

    def test_9as1_fix_mode_prompts_on_dirty_blocks(
        self,
        real_pdb_workspace: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """9AS1 in fix_mode: structure_info auto-resolves, ligands and signaling_partners prompt.

        Block behavior:
        - structure_info: controversy on source (trivial) → auto-resolve + accept
        - receptor_info: no controversy, no direct warnings → auto-accept
        - ligands: controversy on name (significant) → prompts
        - signaling_partners: critical warnings (Ghost Chain, Fake UniProt) → prompts
        - auxiliary_proteins: no controversy, no warnings → auto-accept
        - key_findings: no controversy, no warnings → auto-accept
        """
        from gpcr_tools.csv_generator.review_engine import review_toplevel_blocks

        main_data, controversies, validation_data = _load_and_inject("9AS1")

        # Two dirty blocks prompt via Prompt.ask:
        # 1. ligands (controversy) → accept
        # 2. signaling_partners (critical warnings) → accept
        prompt_script = PromptScript(["a", "a"])
        confirm_script = PromptScript([])

        monkeypatch.setattr(
            "gpcr_tools.csv_generator.review_engine.Prompt.ask",
            prompt_script,
        )
        monkeypatch.setattr(
            "gpcr_tools.csv_generator.review_engine.Confirm.ask",
            confirm_script,
        )

        result = review_toplevel_blocks(
            "9AS1",
            copy.deepcopy(main_data),
            controversies,
            validation_data,
            fix_mode=True,
        )

        assert result is not None
        prompt_script.assert_exhausted()

        assert "structure_info" in result
        assert "receptor_info" in result
        assert "ligands" in result
        assert "signaling_partners" in result


# ── RP-4.5: Delete-Block Suggestion ──────────────────────────────────────


class TestDeleteBlockSuggestion:
    """Test that DELETE_BLOCK suggestions are offered and functional."""

    def test_9m88_receptor_info_delete_block_available(
        self,
        real_pdb_workspace: Path,
    ) -> None:
        """Verify analyze_validation_impact yields DELETE_BLOCK for 9M88 receptor_info."""
        from gpcr_tools.csv_generator.validation_display import analyze_validation_impact

        main_data, _, validation_data = _load_and_inject("9M88")
        result = analyze_validation_impact(
            "receptor_info", main_data["receptor_info"], validation_data
        )
        assert result is not None
        assert result["action"] == "DELETE_BLOCK"
        assert "hallucination" in result["reason"].lower()

    def test_9m88_delete_block_removes_receptor_info(
        self,
        real_pdb_workspace: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Selecting 'd' on receptor_info removes it from returned data."""
        from gpcr_tools.csv_generator.review_engine import review_toplevel_blocks

        main_data, controversies, validation_data = _load_and_inject("9M88")

        # In fix_mode, 9M88 has:
        # - structure_info: no controversy at structure_info level directly,
        #   but has no relevant warnings → auto-accept
        # - receptor_info: has controversy + critical warnings → prompts → respond "d" (delete)
        # - ligands: has controversies (name, type) → prompts → respond "a" (accept)
        # - signaling_partners: has critical warnings (HALLUCINATION in algo_conflicts) → prompts → respond "a"
        # - auxiliary_proteins: no controversy → auto-accept
        # - key_findings: no controversy → auto-accept
        #
        # But NOT in fix_mode: all 6 blocks are presented.
        # Clean blocks go through Confirm.ask.
        # Dirty blocks (receptor_info, ligands, signaling_partners) go through Prompt.ask.

        # Confirm.ask for clean blocks: structure_info, auxiliary_proteins, key_findings
        confirm_script = PromptScript([True, True, True])

        # Prompt.ask for dirty blocks:
        # 1. receptor_info → "d" (delete)
        # 2. ligands → "a" (accept)
        # 3. signaling_partners → "a" (accept)
        prompt_script = PromptScript(["d", "a", "a"])

        monkeypatch.setattr(
            "gpcr_tools.csv_generator.review_engine.Confirm.ask",
            confirm_script,
        )
        monkeypatch.setattr(
            "gpcr_tools.csv_generator.review_engine.Prompt.ask",
            prompt_script,
        )

        result = review_toplevel_blocks(
            "9M88",
            copy.deepcopy(main_data),
            controversies,
            validation_data,
        )

        assert result is not None
        assert "receptor_info" not in result, "receptor_info should have been deleted by 'd' action"
        assert "ligands" in result
        assert "signaling_partners" in result
        assert "structure_info" in result
