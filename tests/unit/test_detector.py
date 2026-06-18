"""Tests for the pre-annotation detect stage: the signal contract, the
G-protein identity detector, and the stage runner (persist + reload)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from gpcr_tools.config import (
    API_MAX_RETRIES,
    DETECT_INCOMPLETE_MARKER_KEY,
    FULL_G_ALPHA_CANDIDATES,
    reset_config,
)
from gpcr_tools.detector.gprotein import G_PROTEIN_LOCUS, detect_g_protein_identity
from gpcr_tools.detector.ligands import detect_incidental_candidates
from gpcr_tools.detector.signals import (
    SEVERITY_ADVISORY,
    SEVERITY_REVIEW,
    SIGNAL_CHIMERIC_GPROTEIN,
    SIGNAL_INCIDENTAL_CANDIDATE,
    DetectSignal,
    to_critical_warnings,
)
from gpcr_tools.detector.stage import load_detect_signals, run_detect, run_detect_stage
from gpcr_tools.validator.cache import SequenceCache

TRANSDUCIN_A5 = "IKENLKDCGLF"
DISTINCT_A5 = "WWWWWWWWWWW"


def _mock_refs(tail_by_slug: dict[str, str | None], default_tail: str = DISTINCT_A5) -> Any:
    def _fetch(accession: str, cache: Any) -> str | None:
        slug = FULL_G_ALPHA_CANDIDATES.get(accession)
        tail = tail_by_slug.get(slug, default_tail) if slug else default_tail
        # An explicit None entry simulates a fetch abstain (transient outage /
        # absent accession): the reference silently drops out of scoring.
        return None if tail is None else "GGGGG" + tail

    return _fetch


def _galpha_entry(sequence: str) -> dict[str, Any]:
    return {
        "polymer_entities": [
            {
                "rcsb_polymer_entity": {"pdbx_description": "G alpha subunit"},
                "entity_poly": {"pdbx_seq_one_letter_code_can": sequence},
            }
        ]
    }


def _nonpoly_entry(comp_ids: list[str]) -> dict[str, Any]:
    return {
        "nonpolymer_entities": [{"nonpolymer_comp": {"chem_comp": {"id": c}}} for c in comp_ids]
    }


_TRANSDUCIN_TAILS = dict.fromkeys(("gnat1_human", "gnat2_human", "gnat3_human"), TRANSDUCIN_A5)


def _outage_fetch(accession: str, cache: Any) -> None:
    """A get_sequence_from_uniprot stand-in that transiently abstains for every
    accession, marking it unavailable this run exactly as the real transient
    (timeout/5xx) branch does."""
    cache.mark_unavailable(accession)
    return None


def _resp(status_code: int) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = ""
    return resp


class TestIncidentalCandidateLigandDetector:
    def test_incidental_candidate_molecules_emit_advisory(self) -> None:
        sigs = detect_incidental_candidates("X", _nonpoly_entry(["CLR", "PLM", "HOH"]))
        assert {s.payload["comp_id"] for s in sigs} == {"CLR", "PLM"}
        assert all(s.kind == SIGNAL_INCIDENTAL_CANDIDATE for s in sigs)
        assert all(s.severity == SEVERITY_ADVISORY for s in sigs)  # prompt routing, not review

    def test_non_incidental_candidate_ligand_not_flagged(self) -> None:
        assert detect_incidental_candidates("X", _nonpoly_entry(["RET", "HOH"])) == []

    def test_no_nonpolymer_no_signal(self) -> None:
        assert detect_incidental_candidates("X", {"polymer_entities": []}) == []


class TestDetectSignal:
    def test_dict_roundtrip(self) -> None:
        s = DetectSignal(
            kind="k", target_ref="a.b", summary="hi", payload={"x": 1}, severity=SEVERITY_REVIEW
        )
        assert DetectSignal.from_dict(s.to_dict()) == s

    def test_to_critical_warnings_only_review(self) -> None:
        sigs = [
            DetectSignal("k1", "loc1", "advisory one", severity=SEVERITY_ADVISORY),
            DetectSignal("k2", "loc2", "review two", severity=SEVERITY_REVIEW),
        ]
        assert to_critical_warnings(sigs) == ["k2 at 'loc2': review two"]


class TestSeverityFailSafe:
    """Anything that is not an explicit, recognised advisory becomes review, so
    an unclassified signal is surfaced to a human, never fed to the model."""

    def test_default_severity_is_review(self) -> None:
        # A detector that forgets to classify its signal must NOT silently get
        # advisory routing (which would feed the model prompt).
        assert DetectSignal(kind="k", target_ref="a", summary="b").severity == SEVERITY_REVIEW

    def test_unrecognised_severity_coerced_to_review(self) -> None:
        sig = DetectSignal("k", "a", "b", severity="totally-bogus")
        assert sig.severity == SEVERITY_REVIEW

    def test_empty_severity_coerced_to_review(self) -> None:
        assert DetectSignal("k", "a", "b", severity="").severity == SEVERITY_REVIEW

    def test_explicit_advisory_preserved(self) -> None:
        sig = DetectSignal("k", "a", "b", severity=SEVERITY_ADVISORY)
        assert sig.severity == SEVERITY_ADVISORY

    def test_from_dict_missing_severity_is_review(self) -> None:
        # A serialised signal from a malformed / future version with no severity
        # key must fail safe to review rather than default to advisory.
        sig = DetectSignal.from_dict({"kind": "k", "target_ref": "a", "summary": "b"})
        assert sig.severity == SEVERITY_REVIEW

    def test_from_dict_unrecognised_severity_is_review(self) -> None:
        sig = DetectSignal.from_dict(
            {"kind": "k", "target_ref": "a", "summary": "b", "severity": "weird"}
        )
        assert sig.severity == SEVERITY_REVIEW

    def test_from_dict_roundtrip_preserves_advisory(self) -> None:
        original = DetectSignal("k", "a", "b", payload={"x": 1}, severity=SEVERITY_ADVISORY)
        assert DetectSignal.from_dict(original.to_dict()) == original


class TestGProteinDetector:
    def test_transducin_emits_review_signal(self, tmp_path: Path) -> None:
        cache = SequenceCache(tmp_path / "seq.json")
        entry = _galpha_entry("MMMMMMMMMM" + TRANSDUCIN_A5)
        with patch(
            "gpcr_tools.validator.chimera.get_sequence_from_uniprot",
            side_effect=_mock_refs(_TRANSDUCIN_TAILS),
        ):
            sigs, _ = detect_g_protein_identity("9IIX", entry, cache)
        assert len(sigs) == 1
        s = sigs[0]
        assert s.kind == SIGNAL_CHIMERIC_GPROTEIN
        assert s.target_ref == G_PROTEIN_LOCUS
        assert s.severity == SEVERITY_REVIEW
        assert s.payload["family"] == "Gi/o"
        assert "Gi/o" in s.summary

    def test_resolved_subtype_is_advisory(self, tmp_path: Path) -> None:
        cache = SequenceCache(tmp_path / "seq.json")
        target = "ACDEFGHIKLM"
        entry = _galpha_entry("MMMMMMMMMM" + target)
        with patch(
            "gpcr_tools.validator.chimera.get_sequence_from_uniprot",
            side_effect=_mock_refs({"gnas2_human": target}),
        ):
            sigs, _ = detect_g_protein_identity("X", entry, cache)
        assert len(sigs) == 1
        assert sigs[0].severity == SEVERITY_ADVISORY
        assert sigs[0].payload["subtype"] == "gnas2_human"

    def test_tie_partner_abstain_routes_subtype_to_review(self, tmp_path: Path) -> None:
        """A partial outage that drops a tie-partner must surface as REVIEW, not a
        confidently-wrong advisory. The transducin alpha5 ties gnat1/2/3; gnat2 and
        gnat3 abstain while gnat1 is fetched. The lone gnat1 survivor must NOT
        become an advisory subtype call -- the consumer emits a Gi/o family review
        signal instead, with a clean family message."""
        cache = SequenceCache(tmp_path / "seq.json")
        entry = _galpha_entry("MMMMMMMMMM" + TRANSDUCIN_A5)
        tails: dict[str, str | None] = {
            "gnat1_human": TRANSDUCIN_A5,
            "gnat2_human": None,  # fetch abstained this run
            "gnat3_human": None,  # fetch abstained this run
        }
        with patch(
            "gpcr_tools.validator.chimera.get_sequence_from_uniprot",
            side_effect=_mock_refs(tails),
        ):
            sigs, _ = detect_g_protein_identity("X", entry, cache)
        assert len(sigs) == 1
        s = sigs[0]
        assert s.severity == SEVERITY_REVIEW
        assert s.payload["subtype"] is None
        assert s.payload["family"] == "Gi/o"
        assert "Gi/o" in s.summary

    def test_unrelated_abstain_keeps_unique_subtype_advisory(self, tmp_path: Path) -> None:
        """A partial outage that drops an UNRELATED reference must not force a
        unique subtype to review. gnas2 is uniquely matched; gnaz (no shared
        inseparable set) abstains and could never have tied it, so the subtype
        stays an advisory call -- no false-review storm during a partial outage."""
        cache = SequenceCache(tmp_path / "seq.json")
        target = "ACDEFGHIKLM"
        entry = _galpha_entry("MMMMMMMMMM" + target)
        tails: dict[str, str | None] = {"gnas2_human": target, "gnaz_human": None}
        with patch(
            "gpcr_tools.validator.chimera.get_sequence_from_uniprot",
            side_effect=_mock_refs(tails),
        ):
            sigs, _ = detect_g_protein_identity("X", entry, cache)
        assert len(sigs) == 1
        assert sigs[0].severity == SEVERITY_ADVISORY
        assert sigs[0].payload["subtype"] == "gnas2_human"

    def test_no_g_protein_no_signal(self, tmp_path: Path) -> None:
        cache = SequenceCache(tmp_path / "seq.json")
        entry = {
            "polymer_entities": [
                {
                    "rcsb_polymer_entity": {"pdbx_description": "Dopamine receptor D2"},
                    "entity_poly": {"pdbx_seq_one_letter_code_can": "MMMMMMMMMMMM"},
                }
            ]
        }
        assert detect_g_protein_identity("X", entry, cache) == ([], False)

    def test_low_confidence_emits_weak_review(self, tmp_path: Path) -> None:
        cache = SequenceCache(tmp_path / "seq.json")
        entry = _galpha_entry("A" * 25)  # matches no real alpha5
        with patch(
            "gpcr_tools.validator.chimera.get_sequence_from_uniprot",
            side_effect=_mock_refs({}),  # every ref gets DISTINCT_A5 (no 'A')
        ):
            sigs, _ = detect_g_protein_identity("X", entry, cache)
        assert len(sigs) == 1
        assert sigs[0].severity == SEVERITY_REVIEW
        assert "too weak" in sigs[0].summary

    def test_transient_reference_outage_reports_degraded(self, tmp_path: Path) -> None:
        # Every reference fetch transiently abstains: no signal can be emitted and
        # the result is flagged degraded so the detect stage marks it for re-run.
        cache = SequenceCache(tmp_path / "seq.json")
        entry = _galpha_entry("MMMMMMMMMM" + TRANSDUCIN_A5)
        with patch(
            "gpcr_tools.validator.chimera.get_sequence_from_uniprot",
            side_effect=_outage_fetch,
        ):
            sigs, degraded = detect_g_protein_identity("X", entry, cache)
        assert sigs == []
        assert degraded is True

    def test_clean_resolution_not_degraded(self, tmp_path: Path) -> None:
        cache = SequenceCache(tmp_path / "seq.json")
        target = "ACDEFGHIKLM"
        entry = _galpha_entry("MMMMMMMMMM" + target)
        with patch(
            "gpcr_tools.validator.chimera.get_sequence_from_uniprot",
            side_effect=_mock_refs({"gnas2_human": target}),
        ):
            sigs, degraded = detect_g_protein_identity("X", entry, cache)
        assert len(sigs) == 1
        assert degraded is False


class TestDetectStage:
    @pytest.fixture
    def ws(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        from gpcr_tools.config import SUPPORTED_CONTRACT_VERSION

        workspace = tmp_path / "ws"
        for sub in ("enriched", "detect", "cache", "contract"):
            (workspace / sub).mkdir(parents=True)
        # run_detect_stage now validates the storage contract before working.
        (workspace / "contract" / "storage_contract.json").write_text(
            json.dumps({"storage_contract_version": SUPPORTED_CONTRACT_VERSION})
        )
        monkeypatch.setenv("GPCR_WORKSPACE", str(workspace))
        reset_config()
        yield workspace
        reset_config()

    def test_run_detect_persists_and_reloads(self, ws: Path) -> None:
        (ws / "enriched" / "9IIX.json").write_text(
            json.dumps(_galpha_entry("MMMMMMMMMM" + TRANSDUCIN_A5))
        )
        with patch(
            "gpcr_tools.validator.chimera.get_sequence_from_uniprot",
            side_effect=_mock_refs(_TRANSDUCIN_TAILS),
        ):
            sigs = run_detect("9IIX")
        assert len(sigs) == 1
        assert sigs[0].severity == SEVERITY_REVIEW
        assert (ws / "detect" / "9IIX.json").is_file()
        assert load_detect_signals("9IIX") == sigs

    def test_run_detect_unwraps_data_entry_envelope(self, ws: Path) -> None:
        # enriched files may be wrapped as {"data": {"entry": {...}}}.
        enveloped = {"data": {"entry": _galpha_entry("MMMMMMMMMM" + TRANSDUCIN_A5)}}
        (ws / "enriched" / "9IIX.json").write_text(json.dumps(enveloped))
        with patch(
            "gpcr_tools.validator.chimera.get_sequence_from_uniprot",
            side_effect=_mock_refs(_TRANSDUCIN_TAILS),
        ):
            sigs = run_detect("9IIX")
        assert len(sigs) == 1
        assert sigs[0].severity == SEVERITY_REVIEW

    def test_run_detect_unwraps_envelope_for_ligand_detector(self, ws: Path) -> None:
        # The envelope unwrap must reach the ligand detector too, not just gprotein.
        enveloped = {"data": {"entry": _nonpoly_entry(["PLM"])}}
        (ws / "enriched" / "9XYZ.json").write_text(json.dumps(enveloped))
        sigs = run_detect("9XYZ", skip_api_checks=True)
        # PLM is an incidental_candidate molecule; the unwrap must reach that ligand detector.
        assert SIGNAL_INCIDENTAL_CANDIDATE in {s.kind for s in sigs}

    def test_run_detect_missing_enriched(self, ws: Path) -> None:
        assert run_detect("NOPE") == []
        assert not (ws / "detect" / "NOPE.json").is_file()  # no file when no input

    def test_skip_api_checks_writes_empty_signal_file(self, ws: Path) -> None:
        (ws / "enriched" / "9IIX.json").write_text(
            json.dumps(_galpha_entry("MMMMMMMMMM" + TRANSDUCIN_A5))
        )
        sigs = run_detect("9IIX", skip_api_checks=True)
        assert sigs == []
        assert (ws / "detect" / "9IIX.json").is_file()  # stage output always present

    def test_metadata_detector_runs_under_skip_api(self, ws: Path) -> None:
        # The excluded-ligand detector is metadata-only, so it runs even when
        # sequence-based detectors are skipped.
        (ws / "enriched" / "9XYZ.json").write_text(json.dumps(_nonpoly_entry(["PLM"])))
        sigs = run_detect("9XYZ", skip_api_checks=True)
        # PLM fires the metadata-only incidental_candidate detector even under skip_api.
        assert SIGNAL_INCIDENTAL_CANDIDATE in {s.kind for s in sigs}

    def test_provided_cache_is_not_saved_by_run_detect(
        self, ws: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A caller running many PDBs owns the shared cache; run_detect must not
        # save a cache it did not create.
        from gpcr_tools.config import get_config
        from gpcr_tools.validator.cache import SequenceCache

        (ws / "enriched" / "9IIX.json").write_text(
            json.dumps(_galpha_entry("MMMMMMMMMM" + TRANSDUCIN_A5))
        )
        cache = SequenceCache(get_config().cache_dir / "uniprot_sequence_cache.json")
        saves: list[int] = []
        monkeypatch.setattr(cache, "save", lambda: saves.append(1))
        with patch(
            "gpcr_tools.validator.chimera.get_sequence_from_uniprot",
            side_effect=_mock_refs(_TRANSDUCIN_TAILS),
        ):
            run_detect("9IIX", cache=cache)
        assert saves == []

    def test_run_detect_stage_persists_cache(self, ws: Path) -> None:
        (ws / "enriched" / "9IIX.json").write_text(
            json.dumps(_galpha_entry("MMMMMMMMMM" + TRANSDUCIN_A5))
        )
        with patch(
            "gpcr_tools.validator.chimera.get_sequence_from_uniprot",
            side_effect=_mock_refs(_TRANSDUCIN_TAILS),
        ):
            run_detect_stage("9IIX")
        assert (ws / "cache" / "uniprot_sequence_cache.json").is_file()

    def test_complete_output_is_skipped_on_rerun(self, ws: Path) -> None:
        (ws / "enriched" / "9IIX.json").write_text(
            json.dumps(_galpha_entry("MMMMMMMMMM" + TRANSDUCIN_A5))
        )
        with patch(
            "gpcr_tools.validator.chimera.get_sequence_from_uniprot",
            side_effect=_mock_refs(_TRANSDUCIN_TAILS),
        ):
            first = run_detect("9IIX")  # writes a complete (un-marked) output
        # A second run must not recompute: the detector would raise if reached.
        with patch(
            "gpcr_tools.detector.stage.detect_g_protein_identity",
            side_effect=AssertionError("complete output should be skipped"),
        ):
            again = run_detect("9IIX")
        assert again == first  # served from the persisted file

    def test_force_recomputes_complete_output(self, ws: Path) -> None:
        (ws / "enriched" / "9IIX.json").write_text(
            json.dumps(_galpha_entry("MMMMMMMMMM" + TRANSDUCIN_A5))
        )
        with patch(
            "gpcr_tools.validator.chimera.get_sequence_from_uniprot",
            side_effect=_mock_refs(_TRANSDUCIN_TAILS),
        ):
            run_detect("9IIX")
            recomputed: list[str] = []

            def _spy(pdb_id: str, entry: Any, cache: Any) -> tuple[list, bool]:
                recomputed.append(pdb_id)
                return [], False

            with patch("gpcr_tools.detector.stage.detect_g_protein_identity", side_effect=_spy):
                run_detect("9IIX", force=True)
        assert recomputed == ["9IIX"]  # recomputed despite a complete output

    def test_incomplete_marked_output_is_recomputed_and_marker_cleared(self, ws: Path) -> None:
        (ws / "enriched" / "9IIX.json").write_text(
            json.dumps(_galpha_entry("MMMMMMMMMM" + TRANSDUCIN_A5))
        )
        (ws / "detect" / "9IIX.json").write_text(
            json.dumps({"pdb_id": "9IIX", "signals": [], DETECT_INCOMPLETE_MARKER_KEY: True})
        )
        with patch(
            "gpcr_tools.validator.chimera.get_sequence_from_uniprot",
            side_effect=_mock_refs(_TRANSDUCIN_TAILS),
        ):
            sigs = run_detect("9IIX")  # recompute now that the fetch succeeds
        assert len(sigs) == 1
        data = json.loads((ws / "detect" / "9IIX.json").read_text())
        assert DETECT_INCOMPLETE_MARKER_KEY not in data  # marker cleared on clean recompute

    def test_transient_outage_marks_detect_incomplete(self, ws: Path) -> None:
        (ws / "enriched" / "9IIX.json").write_text(
            json.dumps(_galpha_entry("MMMMMMMMMM" + TRANSDUCIN_A5))
        )
        with patch(
            "gpcr_tools.validator.chimera.get_sequence_from_uniprot",
            side_effect=_outage_fetch,
        ):
            run_detect("9IIX")
        data = json.loads((ws / "detect" / "9IIX.json").read_text())
        assert data.get(DETECT_INCOMPLETE_MARKER_KEY) is True

    def test_skip_api_checks_preserves_existing_incomplete_marker(self, ws: Path) -> None:
        # A metadata-only pass over a file marked incomplete by an earlier outage
        # cannot resolve the sequence-fetch gap, so it must NOT clear the marker
        # (else a later full run would skip it and freeze the missing signal).
        (ws / "enriched" / "9IIX.json").write_text(
            json.dumps(_galpha_entry("MMMMMMMMMM" + TRANSDUCIN_A5))
        )
        (ws / "detect" / "9IIX.json").write_text(
            json.dumps({"pdb_id": "9IIX", "signals": [], DETECT_INCOMPLETE_MARKER_KEY: True})
        )
        run_detect("9IIX", skip_api_checks=True)
        data = json.loads((ws / "detect" / "9IIX.json").read_text())
        assert data.get(DETECT_INCOMPLETE_MARKER_KEY) is True

    def test_signal_free_structure_not_marked_and_skipped(self, ws: Path) -> None:
        # A structure with no G-alpha legitimately yields no chimera signal: it
        # must carry no incomplete marker and be skipped on a later run (never
        # re-run forever).
        entry: dict[str, Any] = {
            "polymer_entities": [
                {
                    "rcsb_polymer_entity": {"pdbx_description": "Dopamine receptor D2"},
                    "entity_poly": {"pdbx_seq_one_letter_code_can": "MMMMMMMMMMMM"},
                }
            ]
        }
        (ws / "enriched" / "9XYZ.json").write_text(json.dumps(entry))
        run_detect("9XYZ")
        data = json.loads((ws / "detect" / "9XYZ.json").read_text())
        assert DETECT_INCOMPLETE_MARKER_KEY not in data
        with patch(
            "gpcr_tools.detector.stage.detect_g_protein_identity",
            side_effect=AssertionError("signal-free output should be skipped"),
        ):
            run_detect("9XYZ")

    def test_outage_reference_fetched_once_across_run(self, ws: Path) -> None:
        # The shared per-run cache memoizes a transient failure, so during a total
        # outage each reference is attempted once (with its retries) on the first
        # PDB, not re-requested for every subsequent PDB in the batch.
        for pid in ("AAAA", "BBBB", "CCCC"):
            (ws / "enriched" / f"{pid}.json").write_text(
                json.dumps(_galpha_entry("MMMMMMMMMM" + TRANSDUCIN_A5))
            )
        with (
            patch("gpcr_tools.validator.chimera.requests.get", return_value=_resp(503)) as mock_get,
            patch("gpcr_tools.validator.chimera.time.sleep"),
        ):
            run_detect_stage()
        # Bounded by the roster size x retries, regardless of the number of PDBs.
        assert mock_get.call_count <= len(FULL_G_ALPHA_CANDIDATES) * API_MAX_RETRIES
