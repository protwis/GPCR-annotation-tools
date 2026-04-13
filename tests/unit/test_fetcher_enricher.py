"""Tests for fetcher/enricher.py — enrichment logic with mocked HTTP."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from gpcr_tools.fetcher.enricher import (
    _determine_ligand_type,
    _enrich_siblings,
    _enrich_uniprot,
    enrich_single_pdb,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_MINIMAL_PDB_DATA: dict[str, Any] = {
    "data": {
        "entry": {
            "rcsb_id": "7W55",
            "polymer_entities": [
                {
                    "uniprots": [
                        {"rcsb_id": "P29274"},
                        {"rcsb_id": "P63092"},
                    ]
                }
            ],
            "nonpolymer_entities": [
                {
                    "nonpolymer_comp": {
                        "chem_comp": {
                            "id": "ZMA",
                            "formula_weight": 385.4,
                        },
                        "rcsb_chem_comp_descriptor": {
                            "InChIKey": "OIPILFWXSMYKGL-UHFFFAOYSA-N",
                        },
                    }
                }
            ],
            "rcsb_primary_citation": {
                "pdbx_database_id_DOI": "10.1038/s41586-022-04958-8",
            },
        }
    }
}


@pytest.fixture()
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Set up workspace with raw JSON and dirs."""
    monkeypatch.setenv("GPCR_WORKSPACE", str(tmp_path))

    from gpcr_tools.config import reset_config

    reset_config()

    raw_dir = tmp_path / "raw" / "pdb_json"
    raw_dir.mkdir(parents=True)
    enriched_dir = tmp_path / "enriched"
    enriched_dir.mkdir()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    # Write raw JSON
    raw_file = raw_dir / "7W55.json"
    raw_file.write_text(json.dumps(_MINIMAL_PDB_DATA))

    yield tmp_path

    reset_config()


# ---------------------------------------------------------------------------
# determine_ligand_type
# ---------------------------------------------------------------------------


class TestDetermineLigandType:
    def test_small_molecule(self) -> None:
        assert _determine_ligand_type(385.4) == "small-molecule"

    def test_peptide(self) -> None:
        assert _determine_ligand_type(1200.0) == "peptide"

    def test_boundary(self) -> None:
        assert _determine_ligand_type(900.0) == "peptide"

    def test_none(self) -> None:
        assert _determine_ligand_type(None) == "unknown"

    def test_invalid_string(self) -> None:
        assert _determine_ligand_type("not_a_number") == "unknown"


# ---------------------------------------------------------------------------
# UniProt enrichment
# ---------------------------------------------------------------------------


class TestEnrichUniprot:
    def test_adds_slug_from_api(self) -> None:
        pdb_data: dict[str, Any] = {
            "data": {"entry": {"polymer_entities": [{"uniprots": [{"rcsb_id": "P29274"}]}]}}
        }
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [{"primaryAccession": "P29274", "uniProtkbId": "AA2AR_HUMAN"}]
        }
        mock_session.post.return_value = mock_response

        _enrich_uniprot(pdb_data, mock_session, cache=None)

        uni = pdb_data["data"]["entry"]["polymer_entities"][0]["uniprots"][0]
        assert uni["gpcrdb_entry_name_slug"] == "aa2ar_human"

    def test_uses_cache_hit(self) -> None:
        pdb_data: dict[str, Any] = {
            "data": {"entry": {"polymer_entities": [{"uniprots": [{"rcsb_id": "P29274"}]}]}}
        }
        mock_session = MagicMock()
        cache = MagicMock()
        cache.has.return_value = True
        cache.get.return_value = "aa2ar_human"

        _enrich_uniprot(pdb_data, mock_session, cache=cache)

        # Session should not be called (cache hit)
        mock_session.post.assert_not_called()
        uni = pdb_data["data"]["entry"]["polymer_entities"][0]["uniprots"][0]
        assert uni["gpcrdb_entry_name_slug"] == "aa2ar_human"


# ---------------------------------------------------------------------------
# Sibling enrichment
# ---------------------------------------------------------------------------


class TestEnrichSiblings:
    def test_adds_siblings_excluding_self(self) -> None:
        pdb_data: dict[str, Any] = {
            "data": {"entry": {"rcsb_primary_citation": {"pdbx_database_id_DOI": "10.1038/test"}}}
        }
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "result_set": [
                {"identifier": "7W55"},
                {"identifier": "8ABC"},
                {"identifier": "9XYZ"},
            ]
        }
        mock_session.post.return_value = mock_response

        _enrich_siblings(pdb_data, "7W55", mock_session, cache=None)

        siblings = pdb_data["data"]["entry"]["sibling_pdbs"]
        assert siblings == ["8ABC", "9XYZ"]
        assert "7W55" not in siblings

    def test_no_doi_gives_empty_list(self) -> None:
        pdb_data: dict[str, Any] = {"data": {"entry": {"rcsb_primary_citation": None}}}
        _enrich_siblings(pdb_data, "7W55", MagicMock(), cache=None)
        assert pdb_data["data"]["entry"]["sibling_pdbs"] == []


# ---------------------------------------------------------------------------
# Full enrich_single_pdb
# ---------------------------------------------------------------------------


class TestEnrichSinglePdb:
    def test_skips_if_enriched_exists(self, workspace: Path) -> None:
        enriched_path = workspace / "enriched" / "7W55.json"
        enriched_path.write_text("{}")

        result = enrich_single_pdb("7W55")
        assert result is True  # skipped successfully

    def test_fails_if_raw_missing(self, workspace: Path) -> None:
        result = enrich_single_pdb("XXXX")
        assert result is False

    @patch("gpcr_tools.fetcher.enricher._enrich_siblings")
    @patch("gpcr_tools.fetcher.enricher._enrich_ligands")
    @patch("gpcr_tools.fetcher.enricher._enrich_uniprot")
    def test_writes_enriched_output(
        self,
        mock_uniprot: MagicMock,
        mock_ligands: MagicMock,
        mock_siblings: MagicMock,
        workspace: Path,
    ) -> None:
        result = enrich_single_pdb("7W55")
        assert result is True

        enriched_path = workspace / "enriched" / "7W55.json"
        assert enriched_path.exists()

        data = json.loads(enriched_path.read_text())
        assert data["data"]["entry"]["rcsb_id"] == "7W55"
