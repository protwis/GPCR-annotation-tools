"""GPCRdb generic-numbering lookup + contact mapping (no DB, no network).

A shipped static table (built once from GPCRdb, keyed by UniProt accession) maps
each receptor sequence position to its GPCRdb generic number (e.g. ``3x32``) and
segment (e.g. ``TM3``). This is a sequence-level reference -- like the gpcrdb
slug already injected during enrichment -- not GPCRdb's downstream per-structure
curation, so it keeps the tool upstream-independent.

The table lets a ligand's receptor contact residues be expressed as generic
numbers + segments, which the site classifier turns into a ``site_ref``. The
mapping goes structure entity index (``label_seq``) -> UniProt position (via the
RCSB alignment's aligned regions) -> generic number, with an amino-acid identity
gate that drops a contact whose residue does not match the reference (guards
against mutations / mis-alignment so a bad mapping degrades rather than misleads).
"""

from __future__ import annotations

import gzip
import importlib.resources
import json
import logging
from functools import lru_cache
from typing import Any

from gpcr_tools.config import SITE_REF_DATA_FILE

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_numbering_table() -> dict[str, Any]:
    """Load the shipped generic-numbering table (cached). Empty dict on failure.

    Shape: ``{accession: {"c": class_slug, "e": entry_name,
    "r": {sequence_number: [generic_number|None, segment|None, amino_acid]}}}``.
    """
    try:
        src = importlib.resources.files("gpcr_tools") / "data" / SITE_REF_DATA_FILE
        with importlib.resources.as_file(src) as path, gzip.open(path, "rt", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError, ModuleNotFoundError) as exc:
        logger.warning("[site_ref] could not load generic-numbering table: %s", exc)
        return {}


def receptor_class(accession: str) -> str | None:
    """GPCRdb class slug (e.g. ``001``) for *accession*, or None if not in table."""
    rec = load_numbering_table().get(accession)
    return rec.get("c") if isinstance(rec, dict) else None


def map_uniprot_position(label_seq: int, aligned_regions: list[tuple[int, int, int]]) -> int | None:
    """Map an entity SEQRES index to a UniProt position via the aligned regions.

    *aligned_regions* are ``(entity_beg_seq_id, ref_beg_seq_id, length)`` triples
    from RCSB (multi-region, so fusion inserts map only within their own region).
    """
    for entity_beg, ref_beg, length in aligned_regions:
        if entity_beg <= label_seq <= entity_beg + length - 1:
            return ref_beg + (label_seq - entity_beg)
    return None


def map_contacts(
    contacts: list[tuple[int, str]],
    aligned_regions: list[tuple[int, int, int]],
    accession: str,
) -> tuple[set[str], set[str], int, int]:
    """Map ligand contact residues to generic numbers + segments for *accession*.

    *contacts* are ``(label_seq, amino_acid_one_letter)`` of the receptor residues
    the ligand touches. Returns ``(generic_numbers, segments, mapped, gate_fails)``
    where *mapped* contacts passed the amino-acid identity gate and *gate_fails*
    did not (a residue that does not match the reference -- a mutation or a bad
    mapping). An accession absent from the table yields empty sets.
    """
    rec = load_numbering_table().get(accession)
    residues = rec.get("r") if isinstance(rec, dict) else None
    if not isinstance(residues, dict):
        return set(), set(), 0, 0

    generic_numbers: set[str] = set()
    segments: set[str] = set()
    mapped = 0
    gate_fails = 0
    for label_seq, amino_acid in contacts:
        uniprot_pos = map_uniprot_position(label_seq, aligned_regions)
        if uniprot_pos is None:
            continue
        entry = residues.get(str(uniprot_pos))
        if not entry:
            continue
        generic_number, segment, reference_aa = entry[0], entry[1], entry[2]
        if amino_acid != reference_aa:
            gate_fails += 1
            continue
        mapped += 1
        if generic_number:
            generic_numbers.add(generic_number)
        if segment:
            segments.add(segment)
    return generic_numbers, segments, mapped, gate_fails
