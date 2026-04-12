"""Ghostscript-based PDF compression for large files before Gemini upload."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from gpcr_tools.config import PDF_COMPRESSION_THRESHOLD_BYTES

logger = logging.getLogger(__name__)


def compress_pdf_if_needed(input_path: Path, output_path: Path) -> Path:
    """Compress a PDF via Ghostscript when it exceeds the size threshold.

    Returns the path to the compressed file if compression occurred,
    or the original *input_path* if it was below the threshold.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"PDF not found: {input_path}")

    original_size = input_path.stat().st_size
    if original_size <= PDF_COMPRESSION_THRESHOLD_BYTES:
        return input_path

    gs_binary = shutil.which("gs")
    if not gs_binary:
        raise RuntimeError(
            "Ghostscript ('gs') is not installed but is required to compress "
            "large PDFs before sending them to the Gemini API. "
            "Install it with: 'brew install ghostscript' or 'apt-get install ghostscript'."
        )

    logger.info(
        "[%s] PDF is %.1f MB, compressing...",
        input_path.stem,
        original_size / 1024 / 1024,
    )

    cmd = [
        gs_binary,
        "-sDEVICE=pdfwrite",
        "-dCompatibilityLevel=1.4",
        "-dPDFSETTINGS=/ebook",
        "-dNOPAUSE",
        "-dQUIET",
        "-dBATCH",
        f"-sOutputFile={output_path}",
        str(input_path),
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        logger.error("Error compressing PDF: %s", e.stderr.decode())
        raise RuntimeError(f"Failed to compress PDF {input_path}") from e

    compressed_size = output_path.stat().st_size
    ratio = (1 - (compressed_size / original_size)) * 100
    logger.info(
        "[%s] Compressed to %.1f MB (%.1f%% reduction).",
        input_path.stem,
        compressed_size / 1024 / 1024,
        ratio,
    )

    return output_path
