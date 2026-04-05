"""API client wrappers for UniProt, PubChem, and RCSB GraphQL.

Network error handling: returns ``None`` (NOT ``True``) on
timeout/connection failure.  Callers translate ``None`` into an
``[API_UNAVAILABLE]`` warning.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

from gpcr_tools.config import API_MAX_RETRIES
from gpcr_tools.validator.cache import ValidationCache

logger = logging.getLogger(__name__)


def check_uniprot_existence(
    entry_name: str,
    cache: ValidationCache,
) -> bool | None:
    """Validate whether a UniProt entry name exists.

    Returns ``True``/``False`` on success, ``None`` on network error.
    """
    clean_name = entry_name.split(".")[0].upper()
    key = f"uniprot:{clean_name.lower()}"

    cached = cache.get(key)
    if cached is not None:
        return cached

    url = f"https://rest.uniprot.org/uniprotkb/{clean_name}.txt"
    for attempt in range(API_MAX_RETRIES):
        try:
            resp = requests.head(url, timeout=5, allow_redirects=True)
            is_valid = resp.status_code == 200
            cache.set(key, is_valid)
            return is_valid
        except (requests.RequestException, OSError) as exc:
            if attempt == API_MAX_RETRIES - 1:
                logger.warning("UniProt API error for '%s': %s", entry_name, exc)
                return None
            time.sleep(1)
    return None


def check_pubchem_existence(
    cid: str,
    cache: ValidationCache,
) -> bool | None:
    """Validate whether a PubChem CID exists.

    Returns ``True``/``False`` on success, ``None`` on network error.
    """
    clean_cid = "".join(filter(str.isdigit, str(cid)))
    if not clean_cid:
        return False  # Format error (non-numeric)

    key = f"pubchem:{clean_cid}"

    cached = cache.get(key)
    if cached is not None:
        return cached

    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{clean_cid}/description/JSON"
        resp = requests.get(url, timeout=5)
        is_valid = resp.status_code == 200
        cache.set(key, is_valid)
        return is_valid
    except (requests.RequestException, OSError) as exc:
        logger.warning("PubChem API error for '%s': %s", cid, exc)
        return None


# ---------------------------------------------------------------------------
# RCSB GraphQL
# ---------------------------------------------------------------------------

_GRAPHQL_URL = "https://data.rcsb.org/graphql"

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

    Blood Lesson 1 — None-safety:
        Uses ``(data.get("data") or {}).get("entry")`` to handle
        ``{"data": null}`` responses.
    """
    payload = {
        "query": GRAPHQL_POLYMER_FEATURES_QUERY,
        "variables": {"id": pdb_id.upper()},
    }
    try:
        resp = requests.post(_GRAPHQL_URL, json=payload, timeout=15)
        if resp.status_code != 200:
            logger.warning("[%s] GraphQL returned status %d", pdb_id, resp.status_code)
            return None
        data = resp.json()
        if data.get("errors"):
            logger.warning("[%s] GraphQL returned errors: %s", pdb_id, data["errors"])
            return None
        # Blood Lesson 1: (data.get("data") or {}).get("entry")
        return (data.get("data") or {}).get("entry")
    except (requests.RequestException, OSError, ValueError) as exc:
        logger.warning("[%s] GraphQL fetch error: %s", pdb_id, exc)
        return None
