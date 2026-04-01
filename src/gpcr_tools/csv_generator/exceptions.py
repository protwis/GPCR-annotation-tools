"""Custom exceptions for the CSV generator."""


class CsvSchemaMismatchError(Exception):
    """Raised when an existing CSV file has headers that don't match the current schema."""

    def __init__(self, filename: str, expected_len: int, found_len: int):
        self.filename = filename
        self.expected_len = expected_len
        self.found_len = found_len
        self.message = (
            f"{filename} has outdated headers. "
            f"Expected {expected_len} columns, found {found_len}. "
            f"Delete the file and re-run to apply new schema."
        )
        super().__init__(self.message)
