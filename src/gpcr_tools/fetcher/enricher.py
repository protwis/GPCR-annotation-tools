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
import os
import tempfile
from pathlib import Path
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
    LIPID_COMP_IDS,
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

# A resolved polypeptide entity with no reference sequence is treated as a
# peptide ligand only up to this length; longer chains are receptors, fusion
# partners, antibodies or crystallization scaffolds rather than ligands. Sized
# to clear typical peptide agonists while staying below short antibody fragments
# and structural domains.
PEPTIDE_LIGAND_MAX_LENGTH = 50

# Top-level marker stamped on an enriched record that was written while one or
# more external lookups transiently failed. The resume skip re-enriches a record
# carrying this marker (the cross-run caches mean only the failed lookup is
# re-hit), so a partial-outage record self-heals rather than freezing its gaps.
INCOMPLETE_MARKER_KEY = "_enrich_incomplete"

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
        # A previous run may have written this record while one or more lookups
        # were transiently failing; it carries the _enrich_incomplete marker.
        # Re-enrich those (the successful lookups are cached cross-run, so only
        # the previously-failed lookup is re-hit — cheap and self-healing once
        # the transient clears). A clean record has no marker and is skipped.
        if _enriched_is_incomplete(enriched_path):
            logger.info("[%s] Enriched JSON exists but is marked incomplete, re-enriching", pdb_id)
        else:
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

    # Tally external-lookup outcomes so a total outage is not silently written
    # as a successful enrichment.
    stats: dict[str, int] = {"attempted": 0, "hard_failed": 0}

    # 1. UniProt enrichment
    _enrich_uniprot(pdb_data, sess, uniprot_cache, stats=stats)

    # 1b. Polymer ligand-type hints (peptide / nucleic-acid ligands)
    _tag_polymer_ligand_types(pdb_data)

    # 2. Ligand type + PubChem enrichment
    _enrich_ligands(pdb_data, sess, pubchem_cache, synonyms_cache, smiles_cache, stats=stats)

    # 3. Sibling PDB discovery
    _enrich_siblings(pdb_data, pdb_id, sess, doi_cache, stats=stats)

    # Enrichment normally succeeds; if EVERY lookup we attempted hard-failed
    # (network down / all retries exhausted) it is a transient outage, not a
    # real "no data" result. Don't persist a hollow record or report success —
    # leave the PDB unwritten so the next run retries it (the raw JSON is kept).
    if stats["attempted"] > 0 and stats["hard_failed"] == stats["attempted"]:
        logger.error(
            "[%s] All %d enrichment lookup(s) failed (transient outage?) — "
            "not writing enriched JSON; rerun to retry",
            pdb_id,
            stats["attempted"],
        )
        return False

    # A PARTIAL transient failure (some lookups succeeded, some hard-failed)
    # still leaves the record with null/empty fields where the failed lookups
    # would sit. Persist it so the rest of the pipeline proceeds, but stamp it
    # incomplete so the resume skip re-enriches it on the next run instead of
    # freezing the gaps as truth. A fully-successful enrich carries no marker
    # (and any stale marker from a prior partial run is cleared) so it is
    # skipped normally.
    if stats["hard_failed"] > 0:
        pdb_data[INCOMPLETE_MARKER_KEY] = True
        logger.warning(
            "[%s] %d of %d enrichment lookup(s) failed — writing record marked "
            "incomplete; rerun to re-enrich the failed lookup(s)",
            pdb_id,
            stats["hard_failed"],
            stats["attempted"],
        )
    else:
        pdb_data.pop(INCOMPLETE_MARKER_KEY, None)

    # Write enriched output atomically. The existence-based resume skip treats
    # enriched/{id}.json as a completed checkpoint, so a half-written file from
    # an interrupted run must never be left behind — a later run would trust it
    # and the annotate stage's json.load would choke on the truncated JSON.
    try:
        cfg.enriched_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=cfg.enriched_dir, suffix=".tmp", delete=False
        ) as f:
            json.dump(pdb_data, f, indent=2, ensure_ascii=False)
            tmp_name = f.name
        os.replace(tmp_name, enriched_path)
    except OSError as exc:
        logger.error("[%s] Failed to write enriched JSON: %s", pdb_id, exc)
        return False
    logger.info("[%s] Enriched → %s", pdb_id, enriched_path)
    return True


def _enriched_is_incomplete(enriched_path: Path) -> bool:
    """Return True if an existing enriched record is marked incomplete.

    A record stamped with :data:`INCOMPLETE_MARKER_KEY` was written while a
    lookup transiently failed and must be re-enriched. A record that cannot be
    read (truncated / corrupt) is also treated as needing a redo. A clean,
    fully-successful record returns False and is skipped on resume.
    """
    try:
        with open(enriched_path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        # Unreadable: don't trust it as a completed checkpoint — re-enrich.
        return True
    return bool(isinstance(data, dict) and data.get(INCOMPLETE_MARKER_KEY))


# ---------------------------------------------------------------------------
# Step 1: UniProt entry name lookup
# ---------------------------------------------------------------------------


def _enrich_uniprot(
    pdb_data: dict[str, Any],
    session: requests.Session,
    cache: JsonCache | None,
    stats: dict[str, int] | None = None,
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
    slug_map = _resolve_uniprot_slugs(all_accessions, session, cache, stats=stats)

    # Inject back into polymers
    for poly in polymers:
        for uni in poly.get("uniprots") or []:
            acc = uni.get("rcsb_id")
            if acc and acc in slug_map:
                uni["gpcrdb_entry_name_slug"] = slug_map[acc]


def _resolve_secondary_accession(acc: str, session: requests.Session) -> tuple[str | None, bool]:
    """Resolve an accession the batch endpoint did not match (a secondary one).

    The ``/accessions`` batch endpoint matches PRIMARY accessions only -- a
    secondary (merged/retired) accession returns nothing. UniProt merges
    accessions over time, so an RCSB polymer entity can legitimately carry a
    secondary accession; without this fallback that receptor would silently lose
    its ``gpcrdb_entry_name_slug``. The search endpoint maps a secondary accession
    to its current entry via ``sec_acc:``.

    Returns ``(slug, confirmed)``:
      * ``(slug, True)``  -- resolved to a single current entry.
      * ``(None, True)``  -- a definitive HTTP-200 answer with no single match
        (genuinely absent, or an ambiguous demerge); the caller may negative-cache.
      * ``(None, False)`` -- a TRANSIENT failure (non-200 / timeout / parse error);
        the caller must abstain and NOT freeze it as a permanent negative.
    """
    try:
        response = session.get(
            f"{UNIPROT_REST_URL}/search",
            params={"query": f"sec_acc:{acc}", "fields": "accession,id", "format": "json"},
            timeout=TIMEOUT_UNIPROT_BATCH,
        )
        if response.status_code == 200:
            results = response.json().get("results") or []
            # Exactly one current entry = an unambiguous merge. A rare demerge
            # (one old accession -> several entries) is left unresolved rather
            # than guessing which one this PDB meant -- a confirmed non-match.
            if len(results) == 1:
                entry_name = results[0].get("uniProtkbId")
                if entry_name:
                    return str(entry_name).lower(), True
            return None, True  # confirmed: genuinely absent or ambiguous demerge
        return None, False  # non-200: transient/unexpected -> abstain, do not cache
    except (requests.exceptions.RequestException, ValueError) as exc:
        logger.warning("[UniProt sec_acc] Lookup failed for %s: %s", acc, exc)
        return None, False  # transient -> abstain, do not cache


def _resolve_uniprot_slugs(
    accessions: list[str],
    session: requests.Session,
    cache: JsonCache | None,
    stats: dict[str, int] | None = None,
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

    if stats is not None:
        stats["attempted"] += 1
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

        # Accessions the batch did not match are likely SECONDARY (merged/retired)
        # -- the batch endpoint matches primary accessions only. Resolve each via a
        # sec_acc search so a real GPCR carrying a secondary accession is not
        # silently left slug-less.
        confirmed_missing: set[str] = set()
        for acc in to_fetch - found:
            if stats is not None:
                stats["attempted"] += 1
            slug, confirmed = _resolve_secondary_accession(acc, session)
            if slug is not None:
                result[acc] = slug
                if cache:
                    cache.set(acc, slug)
            elif confirmed:
                confirmed_missing.add(acc)
            elif stats is not None:
                # Transient sec_acc-search failure: abstain -- do NOT negative-cache
                # (so it is retried), and count it so the record is marked
                # incomplete and re-enriched on resume. Never freeze a transient
                # outage as a permanent negative (mirrors the primary path).
                stats["hard_failed"] += 1

        # Negative-cache ONLY confirmed misses (a definitive 200 "not in UniProt"),
        # never a transient failure. (Opt into None-caching explicitly.)
        for acc in confirmed_missing:
            if cache:
                cache.set(acc, None, allow_none=True)

    except requests.exceptions.RequestException as exc:
        logger.error("UniProt API request failed: %s", exc)
        if stats is not None:
            stats["hard_failed"] += 1
        for acc in to_fetch:
            result[acc] = None

    return result


def _tag_polymer_ligand_types(pdb_data: dict[str, Any]) -> None:
    """Set ``gpcrdb_determined_type`` on polymer entities that are ligands.

    Peptide and nucleic-acid ligands are polymer entities, so they never reach
    the nonpolymer classifier and previously carried no type hint. A short
    polypeptide with no reference sequence (no UniProt / cross-reference) is a
    peptide ligand; a nucleotide polymer is a nucleic acid. Receptors, fusion
    partners and antibody fragments are excluded by the reference-sequence and
    length checks, so their hint is left unset and they fall through to the
    polymer ``type`` already shown to the model.

    The hint is persisted on the enriched record here. Surfacing it to the
    model prompt is a separate concern: the prompt's polymer block does not yet
    carry ``gpcrdb_determined_type`` (only the nonpolymer block does), so this
    write is currently persist-only and not consumed downstream. Wiring the
    polymer hint into the prompt belongs to the prompt-building layer and is
    intentionally left out of the enrichment step.
    """
    polymers = ((pdb_data.get("data") or {}).get("entry") or {}).get("polymer_entities") or []
    for poly in polymers:
        entity_poly = poly.get("entity_poly") or {}
        poly_type = (entity_poly.get("type") or "").lower()

        if "ribonucleotide" in poly_type or "deoxyribonucleotide" in poly_type:
            poly["gpcrdb_determined_type"] = "na"
            continue

        if "polypeptide" not in poly_type:
            continue

        # A reference sequence (UniProt accession) marks a receptor or fusion
        # partner, never a peptide ligand.
        identifiers = poly.get("rcsb_polymer_entity_container_identifiers") or {}
        if identifiers.get("uniprot_ids"):
            continue
        if identifiers.get("reference_sequence_identifiers"):
            continue
        if any(u.get("rcsb_id") for u in (poly.get("uniprots") or [])):
            continue

        length = entity_poly.get("rcsb_sample_sequence_length")
        if isinstance(length, int | float) and length <= PEPTIDE_LIGAND_MAX_LENGTH:
            poly["gpcrdb_determined_type"] = "peptide"


# ---------------------------------------------------------------------------
# Step 2: Ligand type + PubChem + SMILES
# ---------------------------------------------------------------------------


def _enrich_ligands(
    pdb_data: dict[str, Any],
    session: requests.Session,
    pubchem_cache: JsonCache | None,
    synonyms_cache: JsonCache | None,
    smiles_cache: JsonCache | None,
    stats: dict[str, int] | None = None,
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
        comp_id = chem_comp.get("id")
        comp["gpcrdb_determined_type"] = _determine_ligand_type(comp_id, chem_comp)

        # PubChem CID from InChIKey
        inchikey = descriptor.get("InChIKey")
        pubchem_id = _get_pubchem_cid(inchikey, session, pubchem_cache, stats=stats)
        comp["gpcrdb_pubchem_cid"] = pubchem_id

        # PubChem synonyms
        if pubchem_id:
            synonyms = _get_pubchem_synonyms(pubchem_id, session, synonyms_cache, stats=stats)
            comp["gpcrdb_pubchem_synonyms"] = synonyms if synonyms else []
        else:
            comp["gpcrdb_pubchem_synonyms"] = []

        # SMILES/InChIKey for non-excluded ligands
        if comp_id and comp_id not in LIGAND_EXCLUDE_LIST:
            smiles_data = _fetch_chem_comp_descriptors(comp_id, session, smiles_cache, stats=stats)
            if smiles_data:
                descriptor["SMILES"] = smiles_data.get("SMILES")
                descriptor["SMILES_stereo"] = smiles_data.get("SMILES_stereo")
                if not descriptor.get("InChIKey"):
                    descriptor["InChIKey"] = smiles_data.get("InChIKey")


def _determine_ligand_type(comp_id: str | None, chem_comp: dict[str, Any]) -> str:
    """Classify a nonpolymer ligand by chemical identity, not molecular weight.

    The hint is a deterministic cascade over the comp_id and the CCD
    ``_chem_comp.type``:

    1. A comp_id on the curated lipid whitelist resolves to ``lipid`` directly.
       This is checked first so the answer is stable even for older fetches that
       predate the ``type`` field.
    2. Otherwise the CCD type routes the classes it can name: saccharides,
       free amino acids (peptide-linking monomers), and free mononucleotides
       (nucleotide-linking monomers, e.g. a bound GDP) are all small molecules.
    3. Anything else is a small molecule.

    Peptide and protein ligands are polymer entities and never reach this
    function; their hint is set on the polymer path instead. The previous
    weight-proxy never produced ``lipid`` and mislabelled heavy small molecules
    as peptides, so it is gone.
    """
    if comp_id and comp_id in LIPID_COMP_IDS:
        return "lipid"

    ccd_type = (chem_comp.get("type") or "").lower()
    if "saccharide" in ccd_type:
        return "small-molecule"
    if "peptide linking" in ccd_type:
        # A single free amino acid; treated as a small molecule until the schema
        # gains a dedicated amino-acid value.
        return "small-molecule"
    # A nucleotide-linking component here is a single free mononucleotide (a
    # bound nucleotide cofactor, e.g. a GDP/GTP) -- a small molecule. The 'na'
    # value is reserved for polymer nucleic-acid entities, classified on the
    # polymer path.
    return "small-molecule"


def _get_pubchem_cid(
    inchikey: str | None,
    session: requests.Session,
    cache: JsonCache | None,
    stats: dict[str, int] | None = None,
) -> str | None:
    """Resolve an InChIKey to a PubChem CID."""
    if not inchikey:
        return None
    if cache and cache.has(inchikey):
        return cache.get(inchikey)  # type: ignore[return-value]

    logger.info("Querying PubChem for InChIKey: %s...", inchikey[:15])
    url = f"{PUBCHEM_REST_URL}/inchikey/{inchikey}/cids/JSON"
    pubchem_id: str | None = None
    if stats is not None:
        stats["attempted"] += 1
    try:
        response = session.get(url, timeout=TIMEOUT_PUBCHEM_CID)
        # Three-way outcome (mirrors the validator's existence checks):
        #   200 -> confirmed answer, parse + cache (a 200 with no CID is a real
        #          negative for this InChIKey, cached as a confirmed None).
        #   404 -> definitive "not found"; abstain from caching but it is a true
        #          negative, NOT an outage, so it does not count as hard_failed.
        #   other -> transient/unexpected (403/408/520/Cloudflare 5xx, etc.):
        #          count as hard_failed so the outage guard sees it, cache
        #          nothing, return None. Forcelist transients raise into except.
        if response.status_code == 200:
            data = response.json()
            cids = (data.get("IdentifierList") or {}).get("CID")
            if cids is not None:
                if isinstance(cids, list) and len(cids) > 0:
                    pubchem_id = str(cids[0])
                elif isinstance(cids, int | float):
                    pubchem_id = str(int(cids))
            # Cache only a confirmed (HTTP 200) answer; a 200-with-no-CID is a
            # confirmed negative, so opt into caching the None explicitly.
            if cache:
                cache.set(inchikey, pubchem_id, allow_none=True)
        elif response.status_code == 404:
            logger.info("PubChem has no CID for InChIKey %s (HTTP 404)", inchikey)
        else:
            logger.warning(
                "PubChem CID lookup for %s returned unexpected HTTP %s — abstaining",
                inchikey,
                response.status_code,
            )
            if stats is not None:
                stats["hard_failed"] += 1
    except requests.exceptions.RequestException as exc:
        logger.error("PubChem CID lookup failed for %s: %s", inchikey, exc)
        if stats is not None:
            stats["hard_failed"] += 1

    return pubchem_id


def _get_pubchem_synonyms(
    cid: str,
    session: requests.Session,
    cache: JsonCache | None,
    stats: dict[str, int] | None = None,
) -> list[str] | None:
    """Fetch PubChem synonyms for a CID."""
    if cache and cache.has(cid):
        return cache.get(cid)  # type: ignore[return-value]

    logger.info("Querying PubChem synonyms for CID: %s", cid)
    url = f"{PUBCHEM_REST_URL}/cid/{cid}/synonyms/JSON"
    synonyms: list[str] | None = None
    if stats is not None:
        stats["attempted"] += 1
    try:
        response = session.get(url, timeout=TIMEOUT_PUBCHEM_SYNONYMS)
        # Three-way outcome — see _get_pubchem_cid for the full rationale.
        #   200 -> confirmed list (possibly empty), parse + cache.
        #   404 -> definitive "no synonyms on record": leave None, no cache, NOT
        #          counted as hard_failed (a real negative, not an outage).
        #   other -> transient/unexpected: count as hard_failed, cache nothing.
        if response.status_code == 200:
            data = response.json()
            info_list = (data.get("InformationList") or {}).get("Information") or []
            synonyms = info_list[0].get("Synonym") or [] if info_list else []
            # Cache only a confirmed (HTTP 200) answer — see _get_pubchem_cid.
            if cache:
                cache.set(cid, synonyms)
        elif response.status_code == 404:
            logger.info("PubChem has no synonyms for CID %s (HTTP 404)", cid)
        else:
            logger.warning(
                "PubChem synonyms lookup for CID %s returned unexpected HTTP %s — abstaining",
                cid,
                response.status_code,
            )
            if stats is not None:
                stats["hard_failed"] += 1
    except requests.exceptions.RequestException as exc:
        logger.error("PubChem synonyms lookup failed for CID %s: %s", cid, exc)
        if stats is not None:
            stats["hard_failed"] += 1

    return synonyms


def _fetch_chem_comp_descriptors(
    comp_id: str,
    session: requests.Session,
    cache: JsonCache | None,
    stats: dict[str, int] | None = None,
) -> dict[str, Any] | None:
    """Fetch SMILES/InChIKey via RCSB ``chem_comp`` GraphQL."""
    if cache and cache.has(comp_id):
        return cache.get(comp_id)  # type: ignore[return-value]

    result: dict[str, Any] = {}
    if stats is not None:
        stats["attempted"] += 1
    try:
        resp = session.post(
            RCSB_GRAPHQL_URL,
            json={"query": _CHEM_COMP_QUERY, "variables": {"id": comp_id}},
            headers={"Content-Type": "application/json"},
            timeout=TIMEOUT_RCSB_CHEM_COMP,
        )
        # Three-way outcome — see _get_pubchem_cid for the full rationale.
        #   200 -> confirmed answer (descriptor dict, possibly empty), cache it.
        #   404 -> definitive "not found": no cache, NOT counted as hard_failed.
        #   other -> transient/unexpected: count as hard_failed, cache nothing.
        if resp.status_code == 200:
            data = resp.json().get("data") or {}
            result = (data.get("chem_comp") or {}).get("rcsb_chem_comp_descriptor") or {}
            # Cache only a confirmed (HTTP 200) answer — see _get_pubchem_cid.
            if cache:
                cache.set(comp_id, result)
        elif resp.status_code == 404:
            logger.info("RCSB chem_comp has no descriptor for %s (HTTP 404)", comp_id)
        else:
            logger.warning(
                "RCSB chem_comp query for %s returned unexpected HTTP %s — abstaining",
                comp_id,
                resp.status_code,
            )
            if stats is not None:
                stats["hard_failed"] += 1
    except requests.exceptions.RequestException as exc:
        logger.error("RCSB chem_comp query failed for %s: %s", comp_id, exc)
        if stats is not None:
            stats["hard_failed"] += 1

    return result if result else None


# ---------------------------------------------------------------------------
# Step 3: Sibling PDB discovery
# ---------------------------------------------------------------------------


def _enrich_siblings(
    pdb_data: dict[str, Any],
    pdb_id: str,
    session: requests.Session,
    cache: JsonCache | None,
    stats: dict[str, int] | None = None,
) -> None:
    """Add ``sibling_pdbs`` list to the entry."""
    entry = (pdb_data.get("data") or {}).get("entry") or {}
    doi = (entry.get("rcsb_primary_citation") or {}).get("pdbx_database_id_DOI")

    if not doi:
        entry["sibling_pdbs"] = []
        return

    siblings = _get_pdbs_from_doi(doi, session, cache, stats=stats)
    entry["sibling_pdbs"] = sorted(pid for pid in siblings if pid != pdb_id.upper())


def _get_pdbs_from_doi(
    doi: str,
    session: requests.Session,
    cache: JsonCache | None,
    stats: dict[str, int] | None = None,
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

    if stats is not None:
        stats["attempted"] += 1
    try:
        logger.info("Querying RCSB Search API for DOI: %s", doi)
        response = session.post(api_url, json=query, timeout=TIMEOUT_RCSB_SEARCH)
        response.raise_for_status()
        results = response.json()
        pdb_ids: list[str] = sorted(
            item["identifier"] for item in (results.get("result_set") or []) if "identifier" in item
        )
        if cache:
            cache.set(doi, pdb_ids)
        return pdb_ids
    except requests.exceptions.RequestException as exc:
        logger.error("RCSB Search API failed for DOI %s: %s", doi, exc)
        if stats is not None:
            stats["hard_failed"] += 1
        return []
