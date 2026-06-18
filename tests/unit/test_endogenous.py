"""Tests for the endogenous-ligand classifier (GtoPdb-derived, ligand-intrinsic)."""

from __future__ import annotations

import pytest

from gpcr_tools.validator import endogenous as endo
from gpcr_tools.validator.endogenous import classify_endogenous, load_endogenous_table

# A real endogenous ligand present in the shipped GtoPdb table: (-)-adrenaline.
ADRENALINE_IK = "UCTWMZQNUQWSLP-VIFPVBQESA-N"


def test_shipped_table_loads_and_has_adrenaline() -> None:
    inchikeys, _cids = load_endogenous_table()
    assert ADRENALINE_IK in inchikeys
    assert len(inchikeys) > 100  # the artifact is present and parsed


class TestClassifyAgainstShippedTable:
    def test_endogenous_inchikey_is_true(self) -> None:
        assert classify_endogenous(ADRENALINE_IK, None) == "true"

    def test_real_non_endogenous_small_molecule_is_false(self) -> None:
        # Synthetic garbage key (valid character count, guaranteed absent from GtoPdb):
        # a real small molecule whose identifier is not in the endogenous set -> false.
        assert classify_endogenous("AAAAAAAAAAAAAA-AAAAAAAAAA-A", None) == "false"

    def test_no_identifier_is_unknown(self) -> None:
        # Peptide / ion / unmatched ligand: nothing to look up -> unknown.
        assert classify_endogenous(None, None) == "unknown"
        assert classify_endogenous("", "") == "unknown"

    def test_pubchem_cid_zero_is_not_an_identifier(self) -> None:
        # PubChem CID "0" means "no CID" -> not a usable identifier -> unknown, not false.
        assert classify_endogenous(None, "0") == "unknown"


class TestClassifyWithStubTable:
    def test_missing_table_degrades_to_unknown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(endo, "load_endogenous_table", lambda: (frozenset(), frozenset()))
        assert classify_endogenous(ADRENALINE_IK, None) == "unknown"

    def test_pubchem_cid_is_a_match_axis(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            endo, "load_endogenous_table", lambda: (frozenset(), frozenset({"5816"}))
        )
        assert classify_endogenous(None, "5816") == "true"
        assert classify_endogenous(None, "424242") == "false"
