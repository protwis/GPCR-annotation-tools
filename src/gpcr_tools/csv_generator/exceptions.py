"""Custom exceptions for the CSV generator."""

from __future__ import annotations

from collections.abc import Sequence


class CsvSchemaMismatchError(Exception):
    """Raised when an existing CSV file has headers that don't match the current schema."""

    def __init__(
        self,
        filename: str,
        expected_fields: Sequence[str],
        found_fields: Sequence[str],
    ):
        self.filename = filename
        self.expected_fields = expected_fields
        self.found_fields = found_fields

        if len(expected_fields) != len(found_fields):
            detail = f"Expected {len(expected_fields)} columns, found {len(found_fields)}."
        else:
            missing = set(expected_fields) - set(found_fields)
            extra = set(found_fields) - set(expected_fields)
            parts: list[str] = [f"Column count matches ({len(expected_fields)}) but names differ."]
            if missing:
                parts.append(f"Missing: {sorted(missing)}")
            if extra:
                parts.append(f"Unexpected: {sorted(extra)}")
            if not missing and not extra:
                parts.append("Column order differs.")
            detail = " ".join(parts)

        self.message = (
            f"{filename} has outdated headers. {detail} "
            f"Delete the file and re-run to apply new schema."
        )
        super().__init__(self.message)
