"""Tests for fetcher/targets.py — PDB ID target file reader."""

from __future__ import annotations

from pathlib import Path

from gpcr_tools.fetcher.targets import read_targets


class TestReadTargets:
    """Target file parsing."""

    def test_simple_list(self, tmp_path: Path) -> None:
        f = tmp_path / "ids.txt"
        f.write_text("7W55\n8ABC\n9XYZ\n")
        assert read_targets(f) == ["7W55", "8ABC", "9XYZ"]

    def test_comments_and_blanks_ignored(self, tmp_path: Path) -> None:
        f = tmp_path / "ids.txt"
        f.write_text("# comment\n\n7W55\n  \n# another\n8ABC\n")
        assert read_targets(f) == ["7W55", "8ABC"]

    def test_uppercased(self, tmp_path: Path) -> None:
        f = tmp_path / "ids.txt"
        f.write_text("7w55\n8abc\n")
        assert read_targets(f) == ["7W55", "8ABC"]

    def test_duplicates_removed(self, tmp_path: Path) -> None:
        f = tmp_path / "ids.txt"
        f.write_text("7W55\n8ABC\n7w55\n")
        assert read_targets(f) == ["7W55", "8ABC"]

    def test_empty_file(self, tmp_path: Path) -> None:
        f = tmp_path / "ids.txt"
        f.write_text("# only comments\n\n")
        assert read_targets(f) == []

    def test_missing_file(self, tmp_path: Path) -> None:
        f = tmp_path / "nonexistent.txt"
        assert read_targets(f) == []
