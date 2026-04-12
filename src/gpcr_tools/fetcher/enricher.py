"""Enrichment pipeline — add UniProt, PubChem, SMILES, and sibling data.

Read ``raw/pdb_json/{pdb_id}.json``, enrich with external API data,
write to ``enriched/{pdb_id}.json``.

Enrichment steps (in order):
  1. UniProt entry name lookup  (adds ``gpcrdb_entry_name_slug``)
  2. Ligand type + PubChem CID  (adds ``gpcrdb_determined_type``,
     ``gpcrdb_pubchem_cid``, ``gpcrdb_pubchem_synonyms``, SMILES keys)
  3. Sibling PDB discovery      (adds ``sibling_pdbs``)
"""

from __future__ import annotations

import json
import logging
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from gpcr_tools.config import (
    HTTP_RETRY_ALLOWED_METHODS,
    HTTP_RETRY_BACKOFF_FACTOR,
    HTTP_RETRY_CONNECT,
    HTTP_RETRY_READ,
    HTTP_RETRY_STATUS_FORCELIST,
    HTTP_RETRY_TOTAL,
    LIGAND_EXCLUDE_LIST,
    LIGAND_WEIGHT_THRESHOLD,
    PUBCHEM_REST_URL,
    RCSB_GRAPHQL_URL,
    RCSB_SEARCH_URL,
    TIMEOUT_PUBCHEM_CID,
    TIMEOUT_PUBCHEM_SYNONYMS,
    TIMEOUT_RCSB_CHEM_COMP,
    TIMEOUT_RCSB_SEARCH,
    TIMEOUT_UNIPROT_BATCH,
    UNIPROT_REST_URL,
    USER_AGENT_ENRICHER,
    get_config,
)
from gpcr_tools.fetcher.cache import JsonCache

logger = logging.getLogger(__name__)

_CHEM_COMP_QUERY = """\
query($id: String!) {
  chem_comp(comp_id: $id) {
    rcsb_chem_comp_descriptor {
      InChIKey
      InChI
      SMILES
      SMILES_stereo
    }
  }
}
"""

# ---------------------------------------------------------------------------
# Session (shared, with retry adapter)
# ---------------------------------------------------------------------------


def _build_session() -> requests.Session:
    """Create a requests Session with retry strategy and User-Agent."""
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT_ENRICHER})
    retry = Retry(
        total=HTTP_RETRY_TOTAL,
        read=HTTP_RETRY_READ,
        connect=HTTP_RETRY_CONNECT,
        backoff_factor=HTTP_RETRY_BACKOFF_FACTOR,
        status_forcelist=HTTP_RETRY_STATUS_FORCELIST,
        allowed_methods=list(HTTP_RETRY_ALLOWED_METHODS),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def enrich_single_pdb(
    pdb_id: str,
    *,
    force: bool = False,
    session: requests.Session | None = None,
    uniprot_cache: JsonCache | None = None,
    pubchem_cache: JsonCache | None = None,
    synonyms_cache: JsonCache | None = None,
    doi_cache: JsonCache | None = None,
    smiles_cache: JsonCache | None = None,
) -> bool:
    """Enrich a single PDB entry.

    Return True on success, False on failure or skip.
    """
    cfg = get_config()
    pdb_id = pdb_id.upper()
    raw_path = cfg.raw_pdb_json_dir / f"{pdb_id}.json"
    enriched_path = cfg.enriched_dir / f"{pdb_id}.json"

    if enriched_path.exists() and not force:
        logger.info("[%s] Enriched JSON already exists, skipping", pdb_id)
        return True

    if not raw_path.exists():
        logger.error("[%s] Raw JSON not found at %s", pdb_id, raw_path)
        return False

    try:
        with open(raw_path, encoding="utf-8") as f:
            pdb_data: dict[str, Any] = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("[%s] Failed to read raw JSON: %s", pdb_id, exc)
        return False

    sess = session or _build_session()

    # 1. UniProt enrichment
    _enrich_uniprot(pdb_data, sess, uniprot_cache)

    # 2. Ligand type + PubChem enrichment
    _enrich_ligands(pdb_data, sess, pubchem_cache, synonyms_cache, smiles_cache)

    # 3. Sibling PDB discovery
    _enrich_siblings(pdb_data, pdb_id, sess, doi_cache)

    # Write enriched output
    cfg.enriched_dir.mkdir(parents=True, exist_ok=True)
    with open(enriched_path, "w", encoding="utf-8") as f:
        json.dump(pdb_data, f, indent=2, ensure_ascii=False)
    logger.info("[%s] Enriched → %s", pdb_id, enriched_path)
    return True


# ---------------------------------------------------------------------------
# Step 1: UniProt entry name lookup
# ---------------------------------------------------------------------------


def _enrich_uniprot(
    pdb_data: dict[str, Any],
    session: requests.Session,
    cache: JsonCache | None,
) -> None:
    """Add ``gpcrdb_entry_name_slug`` to each UniProt on polymer entities."""
    polymers = ((pdb_data.get("data") or {}).get("entry") or {}).get("polymer_entities") or []
    if not polymers:
        return

    # Collect all accessions from all polymers
    all_accessions: list[str] = []
    for poly in polymers:
        for uni in poly.get("uniprots") or []:
            acc = uni.get("rcsb_id")
            if acc:
                all_accessions.append(acc)

    if not all_accessions:
        return

    # Resolve all at once
    slug_map = _resolve_uniprot_slugs(all_accessions, session, cache)

    # Inject back into polymers
    for poly in polymers:
        for uni in poly.get("uniprots") or []:
            acc = uni.get("rcsb_id")
            if acc and acc in slug_map:
                uni["gpcrdb_entry_name_slug"] = slug_map[acc]


def _resolve_uniprot_slugs(
    accessions: list[str],
    session: requests.Session,
    cache: JsonCache | None,
) -> dict[str, str | None]:
    """Resolve UniProt accessions to entry name slugs via API + cache."""
    result: dict[str, str | None] = {}
    to_fetch: set[str] = set()

    for acc in set(accessions):
        if cache and cache.has(acc):
            result[acc] = cache.get(acc)
        else:
            to_fetch.add(acc)

    if not to_fetch:
        return result

    logger.info("Querying UniProt API for %d new accession(s)", len(to_fetch))
    api_url = f"{UNIPROT_REST_URL}/accessions"
    params = {"accessions": ",".join(to_fetch), "fields": "accession,id"}

    try:
        response = session.post(api_url, params=params, timeout=TIMEOUT_UNIPROT_BATCH)
        response.raise_for_status()
        api_data = response.json()

        found: set[str] = set()
        for item in api_data.get("results") or []:
            accession = item.get("primaryAccession")
            entry_name = item.get("uniProtkbId")
            if accession and entry_name:
                slug = entry_name.lower()
                result[accession] = slug
                if cache:
                    cache.set(accession, slug)
                found.add(accession)

        # Cache misses as None so we don't re-query
        for acc in to_fetch - found:
            if cache:
                cache.set(acc, None)

    except requests.exceptions.RequestException as exc:
        logger.error("UniProt API request failed: %s", exc)
        for acc in to_fetch:
            result[acc] = None

    return result


# ---------------------------------------------------------------------------
# Step 2: Ligand type + PubChem + SMILES
# ---------------------------------------------------------------------------


def _enrich_ligands(
    pdb_data: dict[str, Any],
    session: requests.Session,
    pubchem_cache: JsonCache | None,
    synonyms_cache: JsonCache | None,
    smiles_cache: JsonCache | None,
) -> None:
    """Add type, PubChem CID, synonyms, and SMILES to nonpolymer entities."""
    non_polymers = ((pdb_data.get("data") or {}).get("entry") or {}).get(
        "nonpolymer_entities"
    ) or []

    for np_entity in non_polymers:
        comp = np_entity.get("nonpolymer_comp") or {}
        chem_comp = comp.get("chem_comp") or {}
        descriptor = comp.get("rcsb_chem_comp_descriptor") or {}

        # Determined type
        formula_weight = chem_comp.get("formula_weight")
        comp["gpcrdb_determined_type"] = _determine_ligand_type(formula_weight)

        # PubChem CID from InChIKey
        inchikey = descriptor.get("InChIKey")
        pubchem_id = _get_pubchem_cid(inchikey, session, pubchem_cache)
        comp["gpcrdb_pubchem_cid"] = pubchem_id

        # PubChem synonyms
        if pubchem_id:
            synonyms = _get_pubchem_synonyms(pubchem_id, session, synonyms_cache)
            comp["gpcrdb_pubchem_synonyms"] = synonyms if synonyms else []
        else:
            comp["gpcrdb_pubchem_synonyms"] = []

        # SMILES/InChIKey for non-excluded ligands
        comp_id = chem_comp.get("id")
        if comp_id and comp_id not in LIGAND_EXCLUDE_LIST:
            smiles_data = _fetch_chem_comp_descriptors(comp_id, session, smiles_cache)
            if smiles_data:
                descriptor["SMILES"] = smiles_data.get("SMILES")
                descriptor["SMILES_stereo"] = smiles_data.get("SMILES_stereo")
                if not descriptor.get("InChIKey"):
                    descriptor["InChIKey"] = smiles_data.get("InChIKey")


def _determine_ligand_type(formula_weight: Any) -> str:
    """Classify ligand as small-molecule, peptide, or unknown."""
    if formula_weight is None:
        return "unknown"
    try:
        return "small-molecule" if float(formula_weight) < LIGAND_WEIGHT_THRESHOLD else "peptide"
    except (ValueError, TypeError):
        return "unknown"


def _get_pubchem_cid(
    inchikey: str | None,
    session: requests.Session,
    cache: JsonCache | None,
) -> str | None:
    """Resolve an InChIKey to a PubChem CID."""
    if not inchikey:
        return None
    if cache and cache.has(inchikey):
        return cache.get(inchikey)  # type: ignore[return-value]

    logger.info("Querying PubChem for InChIKey: %s...", inchikey[:15])
    url = f"{PUBCHEM_REST_URL}/inchikey/{inchikey}/cids/JSON"
    pubchem_id: str | None = None
    try:
        response = session.get(url, timeout=TIMEOUT_PUBCHEM_CID)
        if response.status_code == 200:
            data = response.json()
            cids = (data.get("IdentifierList") or {}).get("CID")
            if cids and isinstance(cids, list) and len(cids) > 0:
                pubchem_id = str(cids[0])
    except requests.exceptions.RequestException as exc:
        logger.error("PubChem CID lookup failed for %s: %s", inchikey, exc)

    if cache:
        cache.set(inchikey, pubchem_id)
    return pubchem_id


def _get_pubchem_synonyms(
    cid: str,
    session: requests.Session,
    cache: JsonCache | None,
) -> list[str] | None:
    """Fetch PubChem synonyms for a CID."""
    if cache and cache.has(cid):
        return cache.get(cid)  # type: ignore[return-value]

    logger.info("Querying PubChem synonyms for CID: %s", cid)
    url = f"{PUBCHEM_REST_URL}/cid/{cid}/synonyms/JSON"
    synonyms: list[str] | None = None
    try:
        response = session.get(url, timeout=TIMEOUT_PUBCHEM_SYNONYMS)
        if response.status_code == 200:
            data = response.json()
            synonyms = (data.get("InformationList") or {}).get("Information", [{}])[0].get(
                "Synonym"
            ) or []
    except requests.exceptions.RequestException as exc:
        logger.error("PubChem synonyms lookup failed for CID %s: %s", cid, exc)

    if cache:
        cache.set(cid, synonyms)
    return synonyms


def _fetch_chem_comp_descriptors(
    comp_id: str,
    session: requests.Session,
    cache: JsonCache | None,
) -> dict[str, Any] | None:
    """Fetch SMILES/InChIKey via RCSB ``chem_comp`` GraphQL."""
    if cache and cache.has(comp_id):
        return cache.get(comp_id)  # type: ignore[return-value]

    result: dict[str, Any] = {}
    try:
        resp = session.post(
            RCSB_GRAPHQL_URL,
            json={"query": _CHEM_COMP_QUERY, "variables": {"id": comp_id}},
            headers={"Content-Type": "application/json"},
            timeout=TIMEOUT_RCSB_CHEM_COMP,
        )
        if resp.status_code == 200:
            data = resp.json().get("data") or {}
            result = (data.get("chem_comp") or {}).get("rcsb_chem_comp_descriptor") or {}
    except requests.exceptions.RequestException as exc:
        logger.error("RCSB chem_comp query failed for %s: %s", comp_id, exc)

    if cache:
        cache.set(comp_id, result)
    return result if result else None


# ---------------------------------------------------------------------------
# Step 3: Sibling PDB discovery
# ---------------------------------------------------------------------------


def _enrich_siblings(
    pdb_data: dict[str, Any],
    pdb_id: str,
    session: requests.Session,
    cache: JsonCache | None,
) -> None:
    """Add ``sibling_pdbs`` list to the entry."""
    entry = (pdb_data.get("data") or {}).get("entry") or {}
    doi = (entry.get("rcsb_primary_citation") or {}).get("pdbx_database_id_DOI")

    if not doi:
        entry["sibling_pdbs"] = []
        return

    siblings = _get_pdbs_from_doi(doi, session, cache)
    entry["sibling_pdbs"] = sorted(pid for pid in siblings if pid != pdb_id.upper())


def _get_pdbs_from_doi(
    doi: str,
    session: requests.Session,
    cache: JsonCache | None,
) -> list[str]:
    """Query RCSB Search API for PDB IDs sharing a DOI."""
    if cache and cache.has(doi):
        return cache.get(doi) or []  # type: ignore[return-value]

    api_url = RCSB_SEARCH_URL
    query = {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_primary_citation.pdbx_database_id_DOI",
                "operator": "exact_match",
                "value": doi,
            },
        },
        "return_type": "entry",
        "request_options": {"return_all_hits": True},
    }

    try:
        logger.info("Querying RCSB Search API for DOI: %s", doi)
        response = session.post(api_url, json=query, timeout=TIMEOUT_RCSB_SEARCH)
        response.raise_for_status()
        results = response.json()
        pdb_ids: list[str] = sorted(
            item["identifier"] for item in (results.get("result_set") or [])
        )
        if cache:
            cache.set(doi, pdb_ids)
        return pdb_ids
    except requests.exceptions.RequestException as exc:
        logger.error("RCSB Search API failed for DOI %s: %s", doi, exc)
        return []
