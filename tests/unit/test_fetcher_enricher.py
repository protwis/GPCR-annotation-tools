"""Tests for fetcher/enricher.py — enrichment logic with mocked HTTP."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import requests

from gpcr_tools.fetcher.enricher import (
    INCOMPLETE_MARKER_KEY,
    _determine_ligand_type,
    _enrich_siblings,
    _enrich_uniprot,
    _fetch_chem_comp_descriptors,
    _get_pubchem_cid,
    _get_pubchem_synonyms,
    _resolve_secondary_accession,
    _tag_polymer_ligand_types,
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
    def test_lipid_from_whitelist(self) -> None:
        # Cholesterol, palmitic acid and myristic acid are genuine lipids on the
        # curated whitelist regardless of CCD type.
        for comp_id in ("CLR", "PLM", "MYR"):
            assert _determine_ligand_type(comp_id, {"type": "non-polymer"}) == "lipid"

    def test_non_lipid_acids_are_small_molecule(self) -> None:
        # Short-chain acids and retinal are deliberately excluded from the lipid
        # whitelist and stay small molecules.
        for comp_id in ("ACT", "SIN", "RET"):
            assert _determine_ligand_type(comp_id, {"type": "non-polymer"}) == "small-molecule"

    def test_heavy_nonpolymer_is_not_peptide(self) -> None:
        # Heavy detergents (e.g. maltose-neopentyl-glycol surrogates) are small
        # molecules; the old weight proxy wrongly called them peptides.
        for comp_id in ("AV0", "BYI", "WB2"):
            assert (
                _determine_ligand_type(comp_id, {"type": "non-polymer", "formula_weight": 1050.0})
                == "small-molecule"
            )

    def test_saccharide_routes_to_small_molecule(self) -> None:
        assert _determine_ligand_type("NAG", {"type": "D-saccharide"}) == "small-molecule"

    def test_free_amino_acid_is_small_molecule(self) -> None:
        assert _determine_ligand_type("GLU", {"type": "L-peptide linking"}) == "small-molecule"

    def test_free_mononucleotide_is_small_molecule(self) -> None:
        # A single nucleotide-linking component is a free mononucleotide (e.g. a
        # bound GDP/GTP cofactor) -- a small molecule. The 'na' value is reserved
        # for polymer nucleic-acid entities, classified on the polymer path.
        assert _determine_ligand_type("A", {"type": "RNA linking"}) == "small-molecule"
        assert _determine_ligand_type("DA", {"type": "DNA linking"}) == "small-molecule"

    def test_unknown_type_defaults_small_molecule(self) -> None:
        assert _determine_ligand_type("ZMA", {}) == "small-molecule"

    def test_missing_comp_id_defaults_small_molecule(self) -> None:
        # A nonpolymer entity with no comp_id cannot hit the lipid whitelist and
        # has no CCD type to route on; it falls through to small-molecule.
        assert _determine_ligand_type(None, {}) == "small-molecule"

    def test_lipid_whitelist_honored_without_ccd_type(self) -> None:
        # Older fetches lack chem_comp.type; the lipid whitelist must still win.
        assert _determine_ligand_type("CLR", {}) == "lipid"


class TestTagPolymerLigandTypes:
    @staticmethod
    def _poly(
        poly_type: str,
        length: int | None,
        uniprot_ids: list[str] | None = None,
        uniprots: list[dict[str, Any]] | None = None,
        reference_sequence_identifiers: list[Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "entity_poly": {"type": poly_type, "rcsb_sample_sequence_length": length},
            "rcsb_polymer_entity_container_identifiers": {
                "uniprot_ids": uniprot_ids,
                "reference_sequence_identifiers": reference_sequence_identifiers,
            },
            "uniprots": uniprots or [],
        }

    def _run(self, polymers: list[dict[str, Any]]) -> list[dict[str, Any]]:
        pdb_data = {"data": {"entry": {"polymer_entities": polymers}}}
        _tag_polymer_ligand_types(pdb_data)
        return polymers

    def test_short_peptide_without_uniprot_is_peptide(self) -> None:
        poly = self._poly("polypeptide(L)", 12)
        self._run([poly])
        assert poly["gpcrdb_determined_type"] == "peptide"

    def test_polypeptide_with_uniprot_is_skipped(self) -> None:
        poly = self._poly("polypeptide(L)", 12, uniprot_ids=["P12345"])
        self._run([poly])
        assert "gpcrdb_determined_type" not in poly

    def test_polypeptide_with_uniprot_object_is_skipped(self) -> None:
        poly = self._poly("polypeptide(L)", 12, uniprots=[{"rcsb_id": "P12345"}])
        self._run([poly])
        assert "gpcrdb_determined_type" not in poly

    def test_polypeptide_with_reference_sequence_is_skipped(self) -> None:
        # A non-UniProt cross-reference (PDB / EMBL / etc.) still marks a resolved
        # entity, not a peptide ligand, so the hint must not be set.
        poly = self._poly(
            "polypeptide(L)", 12, reference_sequence_identifiers=[{"database_accession": "1ABC"}]
        )
        self._run([poly])
        assert "gpcrdb_determined_type" not in poly

    def test_long_polypeptide_is_skipped(self) -> None:
        poly = self._poly("polypeptide(L)", 350)
        self._run([poly])
        assert "gpcrdb_determined_type" not in poly

    def test_nucleotide_polymer_is_na(self) -> None:
        poly = self._poly("polyribonucleotide", 20)
        self._run([poly])
        assert poly["gpcrdb_determined_type"] == "na"


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

    def test_secondary_accession_resolved_via_search(self) -> None:
        # A secondary (merged) accession returns nothing from the batch endpoint;
        # the sec_acc search resolves it to the current entry, so the receptor
        # still gets its slug (instead of being silently lost + negative-cached).
        pdb_data: dict[str, Any] = {
            "data": {"entry": {"polymer_entities": [{"uniprots": [{"rcsb_id": "O14823"}]}]}}
        }
        mock_session = MagicMock()
        batch = MagicMock(status_code=200)
        batch.json.return_value = {"results": []}  # secondary -> no batch match
        mock_session.post.return_value = batch
        search = MagicMock(status_code=200)
        search.json.return_value = {
            "results": [{"primaryAccession": "P07550", "uniProtkbId": "ADRB2_HUMAN"}]
        }
        mock_session.get.return_value = search

        _enrich_uniprot(pdb_data, mock_session, cache=None)

        uni = pdb_data["data"]["entry"]["polymer_entities"][0]["uniprots"][0]
        assert uni["gpcrdb_entry_name_slug"] == "adrb2_human"

    def test_unknown_accession_is_negative_cached_not_recovered(self) -> None:
        # Neither the batch nor the sec_acc search resolves it -> genuine miss:
        # no slug, and it is negative-cached so it is not re-queried.
        pdb_data: dict[str, Any] = {
            "data": {"entry": {"polymer_entities": [{"uniprots": [{"rcsb_id": "X99999"}]}]}}
        }
        mock_session = MagicMock()
        empty = MagicMock(status_code=200)
        empty.json.return_value = {"results": []}
        mock_session.post.return_value = empty
        mock_session.get.return_value = empty
        cache = MagicMock()
        cache.has.return_value = False

        _enrich_uniprot(pdb_data, mock_session, cache=cache)

        uni = pdb_data["data"]["entry"]["polymer_entities"][0]["uniprots"][0]
        assert "gpcrdb_entry_name_slug" not in uni
        cache.set.assert_any_call("X99999", None, allow_none=True)

    def test_transient_secondary_failure_is_not_negative_cached(self) -> None:
        # A TRANSIENT sec_acc-search failure must NOT freeze the accession as a
        # permanent negative -- it abstains so a later run retries it. (Guards the
        # very anti-poisoning rule this whole fix series is about.)
        pdb_data: dict[str, Any] = {
            "data": {"entry": {"polymer_entities": [{"uniprots": [{"rcsb_id": "O14823"}]}]}}
        }
        mock_session = MagicMock()
        batch = MagicMock(status_code=200)
        batch.json.return_value = {"results": []}  # not matched by the batch
        mock_session.post.return_value = batch
        mock_session.get.side_effect = requests.exceptions.Timeout("search down")
        cache = MagicMock()
        cache.has.return_value = False

        _enrich_uniprot(pdb_data, mock_session, cache=cache)

        uni = pdb_data["data"]["entry"]["polymer_entities"][0]["uniprots"][0]
        assert "gpcrdb_entry_name_slug" not in uni  # unresolved this run
        # The accession was NOT negative-cached (so it will be retried next run).
        for call in cache.set.call_args_list:
            assert call.args[:2] != ("O14823", None)


class TestResolveSecondaryAccession:
    # Returns (slug, confirmed): confirmed=True is a definitive HTTP-200 answer;
    # confirmed=False is a transient failure the caller must abstain on.
    def test_single_result_returns_slug_confirmed(self) -> None:
        mock_session = MagicMock()
        resp = MagicMock(status_code=200)
        resp.json.return_value = {
            "results": [{"primaryAccession": "P07550", "uniProtkbId": "ADRB2_HUMAN"}]
        }
        mock_session.get.return_value = resp
        assert _resolve_secondary_accession("O14823", mock_session) == ("adrb2_human", True)

    def test_no_result_is_confirmed_miss(self) -> None:
        mock_session = MagicMock()
        resp = MagicMock(status_code=200)
        resp.json.return_value = {"results": []}
        mock_session.get.return_value = resp
        assert _resolve_secondary_accession("X99999", mock_session) == (None, True)

    def test_ambiguous_demerge_is_confirmed_miss(self) -> None:
        # More than one current entry = ambiguous demerge -> do not guess.
        mock_session = MagicMock()
        resp = MagicMock(status_code=200)
        resp.json.return_value = {
            "results": [{"uniProtkbId": "A_HUMAN"}, {"uniProtkbId": "B_HUMAN"}]
        }
        mock_session.get.return_value = resp
        assert _resolve_secondary_accession("O14823", mock_session) == (None, True)

    def test_non_200_is_transient_not_confirmed(self) -> None:
        mock_session = MagicMock()
        resp = MagicMock(status_code=503)
        mock_session.get.return_value = resp
        assert _resolve_secondary_accession("O14823", mock_session) == (None, False)

    def test_exception_is_transient_not_confirmed(self) -> None:
        mock_session = MagicMock()
        mock_session.get.side_effect = requests.exceptions.Timeout("boom")
        assert _resolve_secondary_accession("O14823", mock_session) == (None, False)


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

    def test_all_lookups_failed_returns_false_and_writes_nothing(self, workspace: Path) -> None:
        """If every external lookup hard-fails (transient outage), don't persist
        a hollow enriched file or report success — leave it for the next run."""
        enriched = workspace / "enriched" / "7W55.json"
        if enriched.exists():
            enriched.unlink()

        session = MagicMock()
        session.post.side_effect = requests.exceptions.ConnectionError("down")
        session.get.side_effect = requests.exceptions.ConnectionError("down")

        result = enrich_single_pdb("7W55", session=session, force=True)

        assert result is False
        assert not enriched.exists()

    def test_all_lookups_failed_via_non_200_returns_false_and_writes_nothing(
        self, workspace: Path
    ) -> None:
        """The outage guard must also fire when lookups fail via a non-forcelist
        non-200 status (e.g. 403). The PubChem/chem_comp helpers abstain in-band
        on such a status (the new three-way path) while the raise_for_status
        helpers surface it as an HTTPError; either way it counts as hard_failed,
        so a 403 on every lookup is a total outage just like a connection error."""
        enriched = workspace / "enriched" / "7W55.json"
        if enriched.exists():
            enriched.unlink()

        # A realistic 403: status_code is 403 and raise_for_status() raises (as
        # requests does for a 4xx), so the in-band three-way helpers see the 403
        # directly and the raise_for_status helpers fail via their except clause.
        resp = MagicMock()
        resp.status_code = 403
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError("403")
        resp.json.return_value = {}
        session = MagicMock()
        session.post.return_value = resp
        session.get.return_value = resp

        result = enrich_single_pdb("7W55", session=session, force=True)

        assert result is False
        assert not enriched.exists()


class TestEnrichmentCacheNotPoisoned:
    """A transient lookup failure must never be written to the shared cache —
    the cache is keyed by InChIKey/CID/comp_id (not PDB), persists across runs,
    and is not cleared by --force, so a cached negative from one network blip
    would suppress that ligand's enrichment for every future run.
    """

    @staticmethod
    def _resp(status, payload=None):
        r = MagicMock()
        r.status_code = status
        r.json.return_value = payload or {}
        return r

    def test_cid_transient_failure_not_cached(self):
        cache = MagicMock()
        cache.has.return_value = False
        session = MagicMock()
        session.get.side_effect = requests.exceptions.ConnectionError("boom")
        assert _get_pubchem_cid("KEY", session, cache) is None
        cache.set.assert_not_called()

    def test_cid_non_200_not_cached(self):
        cache = MagicMock()
        cache.has.return_value = False
        session = MagicMock()
        session.get.return_value = self._resp(503)
        assert _get_pubchem_cid("KEY", session, cache) is None
        cache.set.assert_not_called()

    def test_cid_200_with_no_cid_caches_confirmed_none(self):
        # A 200 with no CID is a confirmed negative for this InChIKey and is
        # cached as None (opt-in) so it is not re-queried; it is NOT an outage.
        cache = MagicMock()
        cache.has.return_value = False
        session = MagicMock()
        session.get.return_value = self._resp(200, {})
        stats = {"attempted": 0, "hard_failed": 0}
        assert _get_pubchem_cid("KEY", session, cache, stats=stats) is None
        cache.set.assert_called_once_with("KEY", None, allow_none=True)
        assert stats["hard_failed"] == 0

    def test_cid_200_is_cached(self):
        cache = MagicMock()
        cache.has.return_value = False
        session = MagicMock()
        session.get.return_value = self._resp(200, {"IdentifierList": {"CID": [271]}})
        assert _get_pubchem_cid("KEY", session, cache) == "271"
        # The 200 path opts into None-caching (so a 200-with-no-CID confirmed
        # negative is cached); a real CID stores identically with the flag set.
        cache.set.assert_called_once_with("KEY", "271", allow_none=True)

    def test_synonyms_transient_failure_not_cached(self):
        cache = MagicMock()
        cache.has.return_value = False
        session = MagicMock()
        session.get.side_effect = requests.exceptions.ReadTimeout("slow")
        assert _get_pubchem_synonyms("271", session, cache) is None
        cache.set.assert_not_called()

    def test_chem_comp_transient_failure_not_cached(self):
        cache = MagicMock()
        cache.has.return_value = False
        session = MagicMock()
        session.post.side_effect = requests.exceptions.ConnectionError("boom")
        assert _fetch_chem_comp_descriptors("ATP", session, cache) is None
        cache.set.assert_not_called()


class TestEnrichmentHelpersThreeWayOutcome:
    """Each enrichment helper must distinguish a definitive 404 (a real
    negative) from a non-forcelist non-200 (a transient/unexpected status).
    A 404 returns None without counting as an outage; any other non-200 counts
    as hard_failed so the outage guard can see it — and neither caches anything,
    so a blip can never freeze as a silent 'absent' in the shared cache.
    """

    @staticmethod
    def _resp(status, payload=None):
        r = MagicMock()
        r.status_code = status
        r.json.return_value = payload or {}
        return r

    # -- _get_pubchem_cid ----------------------------------------------------

    def test_cid_404_is_definitive_not_hard_failed(self):
        cache = MagicMock()
        cache.has.return_value = False
        session = MagicMock()
        session.get.return_value = self._resp(404)
        stats = {"attempted": 0, "hard_failed": 0}
        assert _get_pubchem_cid("KEY", session, cache, stats=stats) is None
        cache.set.assert_not_called()
        assert stats["hard_failed"] == 0

    def test_cid_non_forcelist_non_200_is_hard_failed(self):
        # 403 is not on the urllib3 status_forcelist, so it does NOT raise; it
        # reaches neither the 200 branch nor the except — it must be caught as a
        # transient abstention, counted in hard_failed, and cached nothing.
        cache = MagicMock()
        cache.has.return_value = False
        session = MagicMock()
        session.get.return_value = self._resp(403)
        stats = {"attempted": 0, "hard_failed": 0}
        assert _get_pubchem_cid("KEY", session, cache, stats=stats) is None
        cache.set.assert_not_called()
        assert stats["hard_failed"] == 1

    # -- _get_pubchem_synonyms ----------------------------------------------

    def test_synonyms_404_is_definitive_not_hard_failed(self):
        cache = MagicMock()
        cache.has.return_value = False
        session = MagicMock()
        session.get.return_value = self._resp(404)
        stats = {"attempted": 0, "hard_failed": 0}
        assert _get_pubchem_synonyms("271", session, cache, stats=stats) is None
        cache.set.assert_not_called()
        assert stats["hard_failed"] == 0

    def test_synonyms_non_forcelist_non_200_is_hard_failed(self):
        cache = MagicMock()
        cache.has.return_value = False
        session = MagicMock()
        session.get.return_value = self._resp(520)
        stats = {"attempted": 0, "hard_failed": 0}
        assert _get_pubchem_synonyms("271", session, cache, stats=stats) is None
        cache.set.assert_not_called()
        assert stats["hard_failed"] == 1

    # -- _fetch_chem_comp_descriptors ---------------------------------------

    def test_chem_comp_404_is_definitive_not_hard_failed(self):
        cache = MagicMock()
        cache.has.return_value = False
        session = MagicMock()
        session.post.return_value = self._resp(404)
        stats = {"attempted": 0, "hard_failed": 0}
        assert _fetch_chem_comp_descriptors("ATP", session, cache, stats=stats) is None
        cache.set.assert_not_called()
        assert stats["hard_failed"] == 0

    def test_chem_comp_non_forcelist_non_200_is_hard_failed(self):
        cache = MagicMock()
        cache.has.return_value = False
        session = MagicMock()
        session.post.return_value = self._resp(598)
        stats = {"attempted": 0, "hard_failed": 0}
        assert _fetch_chem_comp_descriptors("ATP", session, cache, stats=stats) is None
        cache.set.assert_not_called()
        assert stats["hard_failed"] == 1


class TestEnrichIncompleteMarker:
    """A partial transient failure (some lookups fail, not all) must persist the
    record stamped incomplete, and the resume skip must re-enrich a record
    carrying that marker — while a fully-successful record is unmarked and
    skipped normally.
    """

    def test_successful_enrich_is_unmarked_and_skipped_on_resume(self, workspace: Path) -> None:
        enriched = workspace / "enriched" / "7W55.json"
        if enriched.exists():
            enriched.unlink()

        with (
            patch("gpcr_tools.fetcher.enricher._enrich_uniprot"),
            patch("gpcr_tools.fetcher.enricher._enrich_ligands"),
            patch("gpcr_tools.fetcher.enricher._enrich_siblings"),
        ):
            assert enrich_single_pdb("7W55", force=True) is True

        data = json.loads(enriched.read_text())
        assert INCOMPLETE_MARKER_KEY not in data

        # Resume (no force): a clean record is skipped without re-enriching.
        with patch("gpcr_tools.fetcher.enricher._enrich_uniprot") as m_uniprot:
            assert enrich_single_pdb("7W55") is True
            m_uniprot.assert_not_called()

    def test_partial_failure_is_marked_incomplete_and_persisted(self, workspace: Path) -> None:
        enriched = workspace / "enriched" / "7W55.json"
        if enriched.exists():
            enriched.unlink()

        # One lookup hard-fails, the rest succeed -> partial outage.
        def _fail_one(pdb_data, *args, stats=None, **kwargs):
            if stats is not None:
                stats["attempted"] += 1
                stats["hard_failed"] += 1

        def _succeed_one(pdb_data, *args, stats=None, **kwargs):
            if stats is not None:
                stats["attempted"] += 1

        with (
            patch("gpcr_tools.fetcher.enricher._enrich_uniprot", side_effect=_succeed_one),
            patch("gpcr_tools.fetcher.enricher._enrich_ligands", side_effect=_fail_one),
            patch("gpcr_tools.fetcher.enricher._enrich_siblings", side_effect=_succeed_one),
        ):
            assert enrich_single_pdb("7W55", force=True) is True

        # Record IS written (so the pipeline proceeds) but marked incomplete.
        assert enriched.exists()
        data = json.loads(enriched.read_text())
        assert data[INCOMPLETE_MARKER_KEY] is True

    def test_incomplete_record_is_reenriched_on_resume(self, workspace: Path) -> None:
        enriched = workspace / "enriched" / "7W55.json"
        # Pre-seed an existing-but-incomplete record (as a prior partial run left it).
        enriched.write_text(json.dumps({INCOMPLETE_MARKER_KEY: True, "data": {}}))

        # Resume WITHOUT force: the marker must trigger a re-enrich, not a skip.
        with (
            patch("gpcr_tools.fetcher.enricher._enrich_uniprot") as m_uniprot,
            patch("gpcr_tools.fetcher.enricher._enrich_ligands"),
            patch("gpcr_tools.fetcher.enricher._enrich_siblings"),
        ):
            assert enrich_single_pdb("7W55") is True
            m_uniprot.assert_called_once()

        # The re-enrich succeeded cleanly, so the marker is cleared.
        data = json.loads(enriched.read_text())
        assert INCOMPLETE_MARKER_KEY not in data

    def test_corrupt_enriched_record_is_reenriched_on_resume(self, workspace: Path) -> None:
        # A truncated / unreadable enriched file (e.g. a prior run interrupted
        # mid-write) must not be trusted as a completed checkpoint: the resume
        # skip must re-enrich it rather than skip it.
        enriched = workspace / "enriched" / "7W55.json"
        enriched.write_text('{"data": {"entry"')  # truncated, invalid JSON

        with (
            patch("gpcr_tools.fetcher.enricher._enrich_uniprot") as m_uniprot,
            patch("gpcr_tools.fetcher.enricher._enrich_ligands"),
            patch("gpcr_tools.fetcher.enricher._enrich_siblings"),
        ):
            assert enrich_single_pdb("7W55") is True
            m_uniprot.assert_called_once()

        # The re-enrich rewrote a valid, clean record.
        data = json.loads(enriched.read_text())
        assert INCOMPLETE_MARKER_KEY not in data
