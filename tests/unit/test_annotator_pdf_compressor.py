import subprocess
from unittest.mock import MagicMock

import pytest

from gpcr_tools.annotator import pdf_compressor


def test_compress_pdf_if_needed_skipped_when_small(tmp_path):
    small_pdf = tmp_path / "small.pdf"
    small_pdf.write_bytes(b"%PDF mock small")
    output_pdf = tmp_path / "output.pdf"

    result = pdf_compressor.compress_pdf_if_needed(small_pdf, output_pdf)
    assert result == small_pdf
    assert not output_pdf.exists()


def test_compress_pdf_if_needed_fails_no_gs(tmp_path, monkeypatch):
    large_pdf = tmp_path / "large.pdf"
    large_pdf.write_bytes(b"0" * (pdf_compressor.PDF_COMPRESSION_THRESHOLD_BYTES + 100))
    output_pdf = tmp_path / "output.pdf"

    monkeypatch.setattr("shutil.which", lambda cmd: None)

    with pytest.raises(RuntimeError, match="Ghostscript"):
        pdf_compressor.compress_pdf_if_needed(large_pdf, output_pdf)


def test_compress_pdf_if_needed_success(tmp_path, monkeypatch):
    large_pdf = tmp_path / "large.pdf"
    large_pdf.write_bytes(b"0" * (pdf_compressor.PDF_COMPRESSION_THRESHOLD_BYTES + 100))
    output_pdf = tmp_path / "output.pdf"

    monkeypatch.setattr("shutil.which", lambda cmd: "/usr/bin/gs")

    # Mock subprocess to just touch the output file mimicking gs
    def mock_run(cmd, *args, **kwargs):
        assert cmd[0] == "/usr/bin/gs"
        assert f"-sOutputFile={output_pdf}" in cmd
        assert str(large_pdf) in cmd
        output_pdf.write_bytes(b"%PDF compressed")
        return MagicMock()

    monkeypatch.setattr("subprocess.run", mock_run)

    result = pdf_compressor.compress_pdf_if_needed(large_pdf, output_pdf)
    assert result == output_pdf
    assert output_pdf.exists()


def test_compress_pdf_if_needed_subprocess_fails(tmp_path, monkeypatch):
    large_pdf = tmp_path / "large.pdf"
    large_pdf.write_bytes(b"0" * (pdf_compressor.PDF_COMPRESSION_THRESHOLD_BYTES + 100))
    output_pdf = tmp_path / "output.pdf"

    monkeypatch.setattr("shutil.which", lambda cmd: "/usr/bin/gs")

    def mock_run_fail(cmd, *args, **kwargs):
        raise subprocess.CalledProcessError(1, cmd, stderr=b"Ghostscript error")

    monkeypatch.setattr("subprocess.run", mock_run_fail)

    with pytest.raises(RuntimeError, match="Failed to compress PDF"):
        pdf_compressor.compress_pdf_if_needed(large_pdf, output_pdf)
