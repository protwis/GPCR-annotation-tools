"""API client wrappers for UniProt, PubChem, and RCSB GraphQL.

Network error handling: returns ``None`` (NOT ``True``) on
timeout/connection failure.  Callers translate ``None`` into an
``[API_UNAVAILABLE]`` warning.
"""

from __future__ import annotations

import html
import logging
import time
from typing import Any, Protocol

import requests

from gpcr_tools.config import (
    API_MAX_RETRIES,
    PUBCHEM_REST_URL,
    RCSB_GRAPHQL_URL,
    SLEEP_VALIDATION_RETRY,
    TIMEOUT_PUBCHEM_SYNONYMS,
    TIMEOUT_PUBCHEM_VALIDATION,
    TIMEOUT_RCSB_GRAPHQL_VALIDATION,
    TIMEOUT_UNIPROT_VALIDATION,
    UNIPROT_REST_URL,
    VALIDATION_RETRY_BACKOFF_FACTOR,
)
from gpcr_tools.validator.cache import ValidationCache


class SynonymCache(Protocol):
    """Structural cache contract for the PubChem synonym list.

    A CID maps to its list of synonym strings.  Any cache exposing
    ``has``/``get``/``set`` (the enrichment ``JsonCache``) satisfies this; only
    definitive (HTTP 200/404) results are ever stored, so a network failure
    never poisons the cache with a false negative.
    """

    def has(self, key: str) -> bool: ...

    def get(self, key: str) -> Any | None: ...

    def set(self, key: str, value: Any, *, allow_none: bool = False) -> None: ...


logger = logging.getLogger(__name__)


def check_uniprot_existence(
    entry_name: str,
    cache: ValidationCache,
) -> bool | None:
    """Validate whether a UniProt entry name exists.

    Returns ``True`` (HTTP 200) or ``False`` (definitive HTTP 404) on a verdict,
    and ``None`` when the service is unavailable -- a transient status (5xx/429),
    timeout, or network error. A transient failure is never cached and never
    reported as "does not exist", so a real entry is not called fake during an
    API outage.
    """
    clean_name = entry_name.split(".")[0].upper()
    key = f"uniprot:{clean_name.lower()}"

    cached = cache.get(key)
    if cached is not None:
        return cached

    url = f"{UNIPROT_REST_URL}/{clean_name}.txt"
    for attempt in range(API_MAX_RETRIES):
        try:
            resp = requests.head(url, timeout=TIMEOUT_UNIPROT_VALIDATION, allow_redirects=True)
            if resp.status_code == 200:
                cache.set(key, True)
                return True
            if resp.status_code == 404:
                cache.set(key, False)
                return False
            if resp.status_code == 400:
                # Bad Request: the query itself is malformed (e.g. a joined multi-slug
                # name like "a_human, b_human"). The service is fine; retrying the same
                # query just repeats the rejection. Abstain immediately -- we cannot know
                # existence from a malformed query -- and never cache it.
                logger.warning(
                    "UniProt rejected query (HTTP 400) for '%s' (malformed entry name?)",
                    entry_name,
                )
                return None
            # 5xx / 429 / other: the service is unavailable, not a verdict that the
            # entry is absent. Retry, then abstain -- never cache a transient status.
            if attempt == API_MAX_RETRIES - 1:
                logger.warning(
                    "UniProt unavailable (HTTP %s) for '%s'", resp.status_code, entry_name
                )
                return None
            time.sleep(SLEEP_VALIDATION_RETRY * VALIDATION_RETRY_BACKOFF_FACTOR**attempt)
        except (requests.RequestException, OSError) as exc:
            if attempt == API_MAX_RETRIES - 1:
                logger.warning("UniProt API error for '%s': %s", entry_name, exc)
                return None
            time.sleep(SLEEP_VALIDATION_RETRY * VALIDATION_RETRY_BACKOFF_FACTOR**attempt)
    return None


def check_pubchem_existence(
    cid: str,
    cache: ValidationCache,
) -> bool | None:
    """Validate whether a PubChem CID exists.

    Returns ``True`` (HTTP 200) or ``False`` (definitive HTTP 404, or a non-numeric
    id) on a verdict, and ``None`` when the service is unavailable -- a transient
    status (5xx/429), timeout, or network error. A transient failure is never
    cached and never reported as "does not exist".
    """
    clean_cid = "".join(filter(str.isdigit, str(cid)))
    if not clean_cid:
        return False  # Format error (non-numeric)

    key = f"pubchem:{clean_cid}"

    cached = cache.get(key)
    if cached is not None:
        return cached

    url = f"{PUBCHEM_REST_URL}/cid/{clean_cid}/description/JSON"
    for attempt in range(API_MAX_RETRIES):
        try:
            resp = requests.get(url, timeout=TIMEOUT_PUBCHEM_VALIDATION)
            if resp.status_code == 200:
                cache.set(key, True)
                return True
            if resp.status_code == 404:
                cache.set(key, False)
                return False
            if resp.status_code == 400:
                # Bad Request: a malformed query, not a transient outage. Abstain
                # immediately without retry; never cache it.
                logger.warning("PubChem rejected query (HTTP 400) for '%s' (malformed?)", cid)
                return None
            # 5xx / 429 / other: service unavailable, not a verdict. Retry, then
            # abstain -- never cache a transient status.
            if attempt == API_MAX_RETRIES - 1:
                logger.warning("PubChem unavailable (HTTP %s) for '%s'", resp.status_code, cid)
                return None
            time.sleep(SLEEP_VALIDATION_RETRY * VALIDATION_RETRY_BACKOFF_FACTOR**attempt)
        except (requests.RequestException, OSError) as exc:
            if attempt == API_MAX_RETRIES - 1:
                logger.warning("PubChem API error for '%s': %s", cid, exc)
                return None
            time.sleep(SLEEP_VALIDATION_RETRY * VALIDATION_RETRY_BACKOFF_FACTOR**attempt)
    return None


def _normalize_synonym(value: str) -> str:
    """Collapse a name to its comparison key: unescaped, lowercase, alnum-only.

    HTML entities are decoded and every non-alphanumeric character is dropped so
    that punctuation, spacing, and case differences between a PDB short name and
    a PubChem synonym do not cause a spurious mismatch.
    """
    unescaped = html.unescape(value)
    return "".join(ch for ch in unescaped.lower() if ch.isalnum())


def check_pubchem_synonym_match(
    cid: str,
    candidate_names: list[str],
    cache: SynonymCache,
) -> bool | None:
    """Check whether a PubChem CID's synonym list contains any candidate name.

    Confirms that a model-supplied CID actually denotes the reported molecule:
    an existence check alone passes a real-but-unrelated compound, so this asks
    whether the CID's own synonyms include the molecule's name (or one of its
    reported synonyms).

    Args:
        cid: The PubChem CID to verify.
        candidate_names: Names that should appear among the CID's synonyms --
            the union of the reported name and any reported synonyms.  Matching
            against this union (not the bare name) keeps the false-reject rate
            low, since many correct CIDs list only an IUPAC or lab-code name.
        cache: Synonym-list cache keyed by CID.

    Returns:
        ``True`` if any candidate matches a synonym; ``False`` if the CID's
        synonyms contain none of them (HTTP 200 with no overlap, or HTTP 404 /
        no synonyms on record); ``None`` when the question cannot be answered --
        empty candidate list, or a network/parse failure (network abstention,
        never treated as a mismatch).  Only definitive (HTTP 200/404) synonym
        lists are cached; a network failure is never cached.
    """
    clean_cid = "".join(filter(str.isdigit, str(cid)))
    if not clean_cid:
        return False  # Format error (non-numeric)

    candidate_keys = {norm for name in candidate_names if (norm := _normalize_synonym(str(name)))}
    if not candidate_keys:
        return None  # Nothing to compare against -> abstain, never reject.

    cached = cache.get(clean_cid) if cache.has(clean_cid) else None
    if cached is not None:
        # A cached miss is stored as ``[]`` (e.g. a 404), never ``None``; ``[] is not
        # None`` is True, so the empty intersection below correctly yields ``False``.
        synonym_keys = {_normalize_synonym(str(s)) for s in cached}
        return bool(candidate_keys & synonym_keys)

    url = f"{PUBCHEM_REST_URL}/cid/{clean_cid}/synonyms/JSON"
    for attempt in range(API_MAX_RETRIES):
        try:
            resp = requests.get(url, timeout=TIMEOUT_PUBCHEM_SYNONYMS)
        except (requests.RequestException, OSError) as exc:
            # Network abstention -- not a mismatch, not cached. Retry, then abstain.
            if attempt == API_MAX_RETRIES - 1:
                logger.warning("PubChem synonym API error for '%s': %s", cid, exc)
                return None
            time.sleep(SLEEP_VALIDATION_RETRY * VALIDATION_RETRY_BACKOFF_FACTOR**attempt)
            continue

        if resp.status_code == 404:
            # CID has no synonyms on record: definitive miss, safe to cache.
            cache.set(clean_cid, [])
            return False
        if resp.status_code == 400:
            # Bad Request: a malformed query, not a transient outage. Abstain
            # immediately without retry; never cache it.
            logger.warning("PubChem synonym rejected query (HTTP 400) for '%s' (malformed?)", cid)
            return None
        if resp.status_code != 200:
            # Transient/unexpected status -- abstain, do not cache. Retry first.
            if attempt == API_MAX_RETRIES - 1:
                logger.warning("PubChem synonym API status %d for '%s'", resp.status_code, cid)
                return None
            time.sleep(SLEEP_VALIDATION_RETRY * VALIDATION_RETRY_BACKOFF_FACTOR**attempt)
            continue

        # HTTP 200 is a definitive verdict. A malformed body will not improve on
        # retry, so a parse error abstains immediately rather than retrying.
        try:
            data = resp.json()
        except ValueError as exc:
            logger.warning("PubChem synonym parse error for '%s': %s", cid, exc)
            return None

        info_list = (data.get("InformationList") or {}).get("Information") or []
        synonyms: list[str] = info_list[0].get("Synonym") or [] if info_list else []
        cache.set(clean_cid, synonyms)  # Cache the confirmed (HTTP 200) list only.
        synonym_keys = {_normalize_synonym(str(s)) for s in synonyms}
        return bool(candidate_keys & synonym_keys)
    return None


# ---------------------------------------------------------------------------
# RCSB GraphQL
# ---------------------------------------------------------------------------

_GRAPHQL_URL = RCSB_GRAPHQL_URL

GRAPHQL_POLYMER_FEATURES_QUERY: str = """\
query structure($id: String!) {
  entry(entry_id: $id) {
    polymer_entities {
      rcsb_polymer_entity_container_identifiers {
        uniprot_ids
      }
      rcsb_polymer_entity_align {
        reference_database_name
        reference_database_accession
        aligned_regions {
          entity_beg_seq_id
          ref_beg_seq_id
          length
        }
      }
      uniprots {
        rcsb_id
        rcsb_uniprot_feature {
          type
          name
          description
          feature_positions {
            beg_seq_id
            end_seq_id
          }
        }
      }
      rcsb_polymer_entity_feature {
        type
        name
        reference_scheme
        feature_positions {
          beg_seq_id
          end_seq_id
        }
      }
      polymer_entity_instances {
        rcsb_polymer_entity_instance_container_identifiers {
          auth_asym_id
        }
        rcsb_polymer_instance_feature {
          type
          name
          feature_positions {
            beg_seq_id
            end_seq_id
          }
        }
      }
    }
  }
}
"""


def fetch_polymer_features(pdb_id: str) -> dict[str, Any] | None:
    """Fetch polymer entity/instance data from RCSB GraphQL.

    Returns the ``entry`` dict, or ``None`` on error.

    None-safety:
        Uses ``(data.get("data") or {}).get("entry")`` to handle
        ``{"data": null}`` responses.
    """
    payload = {
        "query": GRAPHQL_POLYMER_FEATURES_QUERY,
        "variables": {"id": pdb_id.upper()},
    }
    for attempt in range(API_MAX_RETRIES):
        try:
            resp = requests.post(
                _GRAPHQL_URL, json=payload, timeout=TIMEOUT_RCSB_GRAPHQL_VALIDATION
            )
            if resp.status_code == 400:
                # Bad Request: a malformed query, not a transient outage -- retrying
                # repeats the rejection. Abstain immediately.
                logger.warning("[%s] GraphQL rejected query (HTTP 400)", pdb_id)
                return None
            if resp.status_code != 200:
                # Transient/unexpected status -- retry, then abstain.
                if attempt == API_MAX_RETRIES - 1:
                    logger.warning("[%s] GraphQL returned status %d", pdb_id, resp.status_code)
                    return None
                time.sleep(SLEEP_VALIDATION_RETRY * VALIDATION_RETRY_BACKOFF_FACTOR**attempt)
                continue
            data = resp.json()
            if data.get("errors"):
                # A GraphQL-level error is a definitive server verdict, not a network
                # transient; retrying will not change it.
                logger.warning("[%s] GraphQL returned errors: %s", pdb_id, data["errors"])
                return None
            # None-safe: (data.get("data") or {}).get("entry")
            return (data.get("data") or {}).get("entry")
        except ValueError as exc:
            # Malformed body on an HTTP 200 -- a parse failure will not improve on
            # retry, so abstain immediately.
            logger.warning("[%s] GraphQL parse error: %s", pdb_id, exc)
            return None
        except (requests.RequestException, OSError) as exc:
            if attempt == API_MAX_RETRIES - 1:
                logger.warning("[%s] GraphQL fetch error: %s", pdb_id, exc)
                return None
            time.sleep(SLEEP_VALIDATION_RETRY * VALIDATION_RETRY_BACKOFF_FACTOR**attempt)
    return None


class PolymerFeaturesCacheLike(Protocol):
    """Structural cache contract for the polymer-features response.

    A PDB id maps to its GraphQL ``entry`` dict. ``get`` returns ``None`` on a
    miss or expiry; ``set`` stores only successful fetches. The concrete
    ``PolymerFeaturesCache`` in ``validator.cache`` satisfies this.
    """

    def get(self, key: str) -> dict[str, Any] | None: ...

    def set(self, key: str, value: dict[str, Any]) -> None: ...


def fetch_polymer_features_cached(
    pdb_id: str,
    cache: PolymerFeaturesCacheLike,
) -> dict[str, Any] | None:
    """Fetch polymer features, served from *cache* when fresh.

    A cache hit short-circuits the live request. On a miss the live fetch runs;
    a successful response is cached, while a failed (``None``) fetch is NOT
    cached, so a transient RCSB outage is never frozen as a fact and a later run
    re-probes. ``None`` is propagated on failure exactly as
    :func:`fetch_polymer_features`, so callers keep their fail-safe handling.
    """
    cached = cache.get(pdb_id)
    if cached is not None:
        return cached
    entry = fetch_polymer_features(pdb_id)
    if entry is not None:
        cache.set(pdb_id, entry)
    return entry


GRAPHQL_POLYMER_ALIGNMENT_QUERY: str = """\
query structure($id: String!) {
  entry(entry_id: $id) {
    polymer_entities {
      rcsb_polymer_entity_align {
        reference_database_name
        reference_database_accession
        aligned_regions { entity_beg_seq_id ref_beg_seq_id length }
      }
      polymer_entity_instances {
        rcsb_polymer_entity_instance_container_identifiers { auth_asym_id }
      }
    }
  }
}
"""


def fetch_polymer_alignment(
    pdb_id: str,
) -> dict[str, dict[str, list[tuple[int, int, int]]]] | None:
    """Fetch the RCSB SIFTS-derived entity->UniProt alignment, keyed by author chain.

    Returns ``{auth_chain: {uniprot_accession: [(entity_beg, ref_beg, length), ...]}}``
    -- multi-region per accession, so a fusion chain maps each segment to its own
    reference. ``None`` on network / parse failure.
    """
    payload = {"query": GRAPHQL_POLYMER_ALIGNMENT_QUERY, "variables": {"id": pdb_id.upper()}}
    data: dict[str, Any] | None = None
    for attempt in range(API_MAX_RETRIES):
        try:
            resp = requests.post(
                RCSB_GRAPHQL_URL, json=payload, timeout=TIMEOUT_RCSB_GRAPHQL_VALIDATION
            )
            if resp.status_code == 400:
                # Bad Request: a malformed query, not a transient outage -- retrying
                # repeats the rejection. Abstain immediately.
                logger.warning("[%s] alignment GraphQL rejected query (HTTP 400)", pdb_id)
                return None
            if resp.status_code != 200:
                # Transient/unexpected status -- retry, then abstain.
                if attempt == API_MAX_RETRIES - 1:
                    logger.warning("[%s] alignment GraphQL status %d", pdb_id, resp.status_code)
                    return None
                time.sleep(SLEEP_VALIDATION_RETRY * VALIDATION_RETRY_BACKOFF_FACTOR**attempt)
                continue
            data = resp.json()
            if data.get("errors"):
                # A GraphQL-level error is a definitive server verdict, not a network
                # transient; retrying will not change it.
                logger.warning("[%s] alignment GraphQL errors: %s", pdb_id, data["errors"])
                return None
            break
        except ValueError as exc:
            # Malformed body on an HTTP 200 -- abstain immediately, retry will not help.
            logger.warning("[%s] alignment GraphQL parse error: %s", pdb_id, exc)
            return None
        except (requests.RequestException, OSError) as exc:
            if attempt == API_MAX_RETRIES - 1:
                logger.warning("[%s] alignment GraphQL fetch error: %s", pdb_id, exc)
                return None
            time.sleep(SLEEP_VALIDATION_RETRY * VALIDATION_RETRY_BACKOFF_FACTOR**attempt)
    if data is None:
        return None

    entry = (data.get("data") or {}).get("entry") or {}
    chains: dict[str, dict[str, list[tuple[int, int, int]]]] = {}
    for entity in entry.get("polymer_entities") or []:
        if not isinstance(entity, dict):
            continue
        regions: dict[str, list[tuple[int, int, int]]] = {}
        for align in entity.get("rcsb_polymer_entity_align") or []:
            if not isinstance(align, dict) or align.get("reference_database_name") != "UniProt":
                continue
            acc = align.get("reference_database_accession")
            if not acc:
                continue
            for region in align.get("aligned_regions") or []:
                beg = region.get("entity_beg_seq_id")
                ref = region.get("ref_beg_seq_id")
                length = region.get("length")
                if beg is not None and ref is not None and length is not None:
                    regions.setdefault(acc, []).append((int(beg), int(ref), int(length)))
        if not regions:
            continue
        for inst in entity.get("polymer_entity_instances") or []:
            if not isinstance(inst, dict):
                continue
            auth = (inst.get("rcsb_polymer_entity_instance_container_identifiers") or {}).get(
                "auth_asym_id"
            )
            if auth:
                chains[auth] = regions
    return chains
