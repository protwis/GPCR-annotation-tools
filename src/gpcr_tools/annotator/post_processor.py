"""Post-processing pipeline for raw Gemini annotation responses.

Unwraps protobuf Composite objects, removes empty signaling-partner blocks,
standardises auxiliary-protein names, and lowercases all ``uniprot_entry_name``
values at every depth in the result tree.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from gpcr_tools.config import EMPTY_VALUES

# ---------------------------------------------------------------------------
# Auxiliary protein name normalisation
# ---------------------------------------------------------------------------

AUX_PROTEIN_NAME_MAPPING: dict[str, str] = {
    "Nb35": "Nanobody-35",
    "Nanobody 35": "Nanobody-35",
    "nb35": "Nanobody-35",
    "Nanobody35": "Nanobody-35",
}


def _unwrap_composite(data: Any) -> Any:
    """Convert internal proto Composite objects to plain Python types."""
    # Using duck typing because the actual protobuf types vary depending on the SDK version
    if isinstance(data, Mapping) and not isinstance(data, dict):
        return {str(k): _unwrap_composite(v) for k, v in data.items()}
    if isinstance(data, Sequence) and not isinstance(data, str | bytes | list | tuple):
        return [_unwrap_composite(x) for x in data]
    if isinstance(data, dict):
        return {str(k): _unwrap_composite(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_unwrap_composite(x) for x in data]
    return data


def is_meaningfully_empty(val: Any) -> bool:
    """Return ``True`` if *val* is effectively empty or explicitly null-like."""
    if val is None:
        return True
    if isinstance(val, dict | list | str) and not val:
        return True
    if isinstance(val, str) and val.lower().strip() in EMPTY_VALUES:
        return True
    return isinstance(val, str) and val.lower().strip() in {"not present", "missing"}


def _is_signaling_partners_empty(sp: dict[str, Any]) -> bool:
    """Check whether the signaling_partners block is effectively empty.

    Ignores keys like ``note`` when all substantive fields are empty.
    """
    if not sp:
        return True

    for key, val in sp.items():
        if key == "note":
            continue
        if not is_meaningfully_empty(val):
            return False

    return True


def _standardize_auxiliary_name(name: str | None) -> str | None:
    """Standardise common auxiliary protein names.

    The explicit :data:`AUX_PROTEIN_NAME_MAPPING` is checked first; only if no
    match is found do we fall through to the BRIL and nanobody regex patterns.
    """
    if not name:
        return name

    # 1. Explicit lookup table
    mapped = AUX_PROTEIN_NAME_MAPPING.get(name)
    if mapped is not None:
        return mapped

    # 2. Standardize BRIL variants
    if re.search(r"\bbril\b", name, flags=re.IGNORECASE) or name.lower() == "cytochrome b562 ril":
        return "BRIL"

    # 3. Standardize Nb / Nanobody
    match = re.match(r"^nb\s*(-)?\s*(\d+)$", name, flags=re.IGNORECASE)
    if match:
        return f"Nanobody-{match.group(2)}"

    return name


def _recursive_lowercase_uniprot(data: Any) -> Any:
    """Recursively walk *data* and lowercase every ``uniprot_entry_name`` value."""
    if isinstance(data, dict):
        new_dict: dict[str, Any] = {}
        for key, value in data.items():
            if key == "uniprot_entry_name" and isinstance(value, str):
                new_dict[key] = value.lower()
            else:
                new_dict[key] = _recursive_lowercase_uniprot(value)
        return new_dict
    if isinstance(data, list):
        return [_recursive_lowercase_uniprot(item) for item in data]
    return data


def post_process_annotation(raw_data: Any) -> dict[str, Any]:
    """Apply all post-processing rules to a raw Gemini annotation response.

    Processing order:
    1. Unwrap protobuf Composite wrappers to plain Python dicts/lists.
    2. Delete the ``signaling_partners`` block when it is meaningfully empty.
    3. Standardise auxiliary protein names via
       :func:`_standardize_auxiliary_name`.
    4. Recursively lowercase every ``uniprot_entry_name`` at any depth.
    """
    data = _unwrap_composite(raw_data)

    if not isinstance(data, dict):
        return {"error": "Root object is not a dictionary", "raw": data}

    # 2. Delete signaling_partners if meaningfully empty
    sp = data.get("signaling_partners")
    if isinstance(sp, dict) and _is_signaling_partners_empty(sp):
        del data["signaling_partners"]

    # 3. Standardize auxiliary protein names
    aux = data.get("auxiliary_proteins")
    if isinstance(aux, list):
        for entry in aux:
            if isinstance(entry, dict):
                name = entry.get("name")
                if isinstance(name, str):
                    entry["name"] = _standardize_auxiliary_name(name)

    # 4. Recursive lowercase of all uniprot_entry_name values
    result: dict[str, Any] = _recursive_lowercase_uniprot(data)

    return result
