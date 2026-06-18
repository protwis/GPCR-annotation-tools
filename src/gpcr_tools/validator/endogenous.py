"""Ligand-intrinsic endogenous classification (GtoPdb-derived, no AI, no GPCRdb).

``is_endogenous`` asks of the LIGAND itself -- is this compound an endogenous
ligand of any target in the Guide to PHARMACOLOGY? -- independent of the receptor
a given structure happens to show it in. Backed by a shipped flat set of
endogenous-ligand InChIKeys + PubChem CIDs (built by
``scripts/build_endogenous_table.py`` from GtoPdb, CC-BY-SA 4.0).

Upstream-independent: GtoPdb is the source GPCRdb itself imports endogenous data
from, so this never touches GPCRdb's downstream per-structure curation.
"""

from __future__ import annotations

import gzip
import importlib.resources
import json
import logging
from functools import lru_cache

from gpcr_tools.config import ENDOGENOUS_DATA_FILE

logger = logging.getLogger(__name__)

ENDOGENOUS_TRUE = "true"
ENDOGENOUS_FALSE = "false"
ENDOGENOUS_UNKNOWN = "unknown"


@lru_cache(maxsize=1)
def load_endogenous_table() -> tuple[frozenset[str], frozenset[str]]:
    """``(endogenous InChIKeys, endogenous PubChem CIDs)`` from the shipped GtoPdb
    set; empty sets on any failure (a missing artifact degrades to all-unknown)."""
    try:
        src = importlib.resources.files("gpcr_tools") / "data" / ENDOGENOUS_DATA_FILE
        with (
            importlib.resources.as_file(src) as path,
            gzip.open(path, "rt", encoding="utf-8") as f,
        ):
            data = json.load(f)
        inchikeys = frozenset(data.get("inchikeys") or [])
        cids = frozenset(str(c) for c in (data.get("pubchem_cids") or []))
        return inchikeys, cids
    except (OSError, json.JSONDecodeError, ModuleNotFoundError) as exc:
        logger.warning("[endogenous] could not load %s: %s", ENDOGENOUS_DATA_FILE, exc)
        return frozenset(), frozenset()


def classify_endogenous(inchikey: str | None, pubchem_cid: str | None) -> str:
    """Is this bound small molecule an endogenous ligand? ``true`` / ``false`` / ``unknown``.

    * ``true``  -- the ligand's InChIKey or PubChem CID is in the GtoPdb endogenous set.
    * ``false`` -- it is a real small molecule (has a usable identifier) but is NOT in
      the set (e.g. a synthetic agonist / analog).
    * ``unknown`` -- no usable identifier (a peptide, ion, buffer, or unmatched ligand),
      or the table could not be loaded. Never a fabricated true/false.
    """
    inchikeys, cids = load_endogenous_table()
    if not inchikeys and not cids:
        return ENDOGENOUS_UNKNOWN
    ik = (inchikey or "").strip()
    cid = str(pubchem_cid).strip() if pubchem_cid is not None else ""
    if cid == "0":  # PubChem CID "0"/0 conventionally means "no CID" -> not an identifier
        cid = ""
    if not ik and not cid:
        return ENDOGENOUS_UNKNOWN
    if (ik and ik in inchikeys) or (cid and cid in cids):
        return ENDOGENOUS_TRUE
    return ENDOGENOUS_FALSE
