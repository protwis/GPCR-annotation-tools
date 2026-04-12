"""Multi-tier PDF downloader for GPCR papers.

Downloads open-access PDFs via a tiered fallback strategy:
  Tier 0 — CrossRef metadata enrichment (extract PMID/PMCID)
  Tier 1 — Unpaywall (best OA PDF link)
  Tier 2 — NCBI PMC OA interface (PDF link from XML)
  Fallback — mark as ``"fallback_paywalled"``

Reads enriched JSON from ``enriched/{pdb_id}.json``, writes PDFs to
``papers/{pdb_id}.pdf``, and updates ``state/download_log.json``
via atomic write after each PDB.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import tempfile
import time
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from gpcr_tools.config import (
    CROSSREF_API_URL,
    DL_STATUS_ABSTRACT_ONLY,
    HTTP_RETRY_ALLOWED_METHODS,
    HTTP_RETRY_BACKOFF_FACTOR,
    HTTP_RETRY_CONNECT,
    HTTP_RETRY_READ,
    HTTP_RETRY_STATUS_FORCELIST,
    HTTP_RETRY_TOTAL,
    NCBI_EUTILS_EFETCH_URL,
    NCBI_PMC_OA_URL,
    PDF_DOWNLOAD_CHUNK_SIZE,
    SLEEP_NCBI_RATE_LIMIT,
    TIMEOUT_CROSSREF,
    TIMEOUT_NCBI_EUTILS,
    TIMEOUT_NCBI_PMC_OA,
    TIMEOUT_PDF_DOWNLOAD,
    TIMEOUT_UNPAYWALL,
    UNPAYWALL_API_URL,
    get_config,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PDF_MAGIC = b"%PDF"


# ---------------------------------------------------------------------------
# Session builder
# ---------------------------------------------------------------------------


def _build_session(email: str) -> requests.Session:
    """Build a requests Session with retry adapter and polite headers."""
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": f"LitFetcher/2.0 (mailto:{email})",
            "From": email,
        }
    )
    retry = Retry(
        total=HTTP_RETRY_TOTAL,
        read=HTTP_RETRY_READ,
        connect=HTTP_RETRY_CONNECT,
        backoff_factor=HTTP_RETRY_BACKOFF_FACTOR,
        status_forcelist=HTTP_RETRY_STATUS_FORCELIST,
        allowed_methods=list(HTTP_RETRY_ALLOWED_METHODS),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


# ---------------------------------------------------------------------------
# Download log (atomic read/write)
# ---------------------------------------------------------------------------


def _read_download_log() -> dict[str, Any]:
    """Read the download log, returning empty dict if absent or corrupt."""
    cfg = get_config()
    path = cfg.download_log_file
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read download log: %s", exc)
        return {}


def _update_download_log(pdb_id: str, entry: dict[str, Any]) -> None:
    """Atomic read-modify-write for the download log.

    Follows the same pattern as ``aggregator/runner._update_aggregate_log``.
    """
    cfg = get_config()
    path = cfg.download_log_file
    path.parent.mkdir(parents=True, exist_ok=True)

    log_data = _read_download_log()
    log_data[pdb_id] = entry

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=str(path.parent),
            suffix=".tmp",
            delete=False,
            encoding="utf-8",
        ) as fd:
            tmp_path = fd.name
            json.dump(log_data, fd, indent=2, ensure_ascii=False)
        os.replace(tmp_path, str(path))
        tmp_path = None
    except OSError as exc:
        logger.error("Failed to write download log: %s", exc)
    finally:
        if tmp_path is not None:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Tier API functions
# ---------------------------------------------------------------------------


def _fetch_crossref_metadata(doi: str, session: requests.Session) -> dict[str, str | None]:
    """Tier 0: Enrich with CrossRef metadata (PMID, PMCID)."""
    url = f"{CROSSREF_API_URL}/{doi}"
    try:
        response = session.get(url, timeout=TIMEOUT_CROSSREF)
        if response.status_code == 200:
            data = response.json().get("message") or {}
            pmid = data.get("PMID")
            pmcid: str | None = None
            for link in data.get("link") or []:
                link_url = link.get("URL") or ""
                if "www.ncbi.nlm.nih.gov/pmc/articles/PMC" in link_url:
                    match = re.search(r"PMC(\d+)", link_url)
                    if match:
                        pmcid = match.group(1)
            return {"pmid": pmid, "pmcid": pmcid}
    except requests.exceptions.RequestException as exc:
        logger.warning("[CrossRef] Failed for DOI %s: %s", doi, exc)
    return {"pmid": None, "pmcid": None}


def _fetch_unpaywall_pdf_url(doi: str, session: requests.Session) -> str | None:
    """Tier 1: Get OA PDF URL from Unpaywall."""
    url = f"{UNPAYWALL_API_URL}/{doi}"
    try:
        response = session.get(url, timeout=TIMEOUT_UNPAYWALL)
        if response.status_code == 200:
            data = response.json()
            oa_location = data.get("best_oa_location") or {}
            pdf_url = oa_location.get("url_for_pdf")
            if pdf_url:
                return pdf_url  # type: ignore[no-any-return]
    except requests.exceptions.RequestException as exc:
        logger.warning("[Unpaywall] Failed for DOI %s: %s", doi, exc)
    return None


def _fetch_pmc_oa_pdf_url(pmcid: str, session: requests.Session) -> str | None:
    """Tier 2: Get PDF URL from NCBI PMC OA interface."""
    pmcid_num = pmcid.upper().replace("PMC", "")
    url = f"{NCBI_PMC_OA_URL}?id=PMC{pmcid_num}"
    try:
        response = session.get(url, timeout=TIMEOUT_NCBI_PMC_OA)
        time.sleep(SLEEP_NCBI_RATE_LIMIT)
        if response.status_code == 200:
            root = ET.fromstring(response.content)
            for record in root.findall(".//record"):
                for link in record.findall('.//link[@format="pdf"]'):
                    pdf_href = link.get("href")
                    if pdf_href:
                        # Convert FTP to HTTPS if needed
                        if pdf_href.startswith("ftp://ftp.ncbi.nlm.nih.gov/"):
                            pdf_href = pdf_href.replace("ftp://", "https://", 1)
                        return pdf_href
    except (requests.exceptions.RequestException, ET.ParseError) as exc:
        logger.warning("[PMC OA] Failed for PMCID %s: %s", pmcid, exc)
    return None


def _fetch_abstract_from_ncbi(pmid: str | int, session: requests.Session) -> str | None:
    """Fetch an abstract from PubMed via NCBI E-utilities efetch.

    Parameters
    ----------
    pmid:
        PubMed identifier (string or integer).
    session:
        Requests session to use for the HTTP call.

    Returns
    -------
    str | None
        Cleaned abstract text, or ``None`` on any failure.
    """
    url = f"{NCBI_EUTILS_EFETCH_URL}?db=pubmed&id={pmid}&rettype=abstract&retmode=text"
    try:
        response = session.get(url, timeout=TIMEOUT_NCBI_EUTILS)
        time.sleep(SLEEP_NCBI_RATE_LIMIT)
        if response.status_code == 200 and response.text:
            clean_text = re.sub(r"^\s*\d+\.\s*", "", response.text.strip())
            clean_text = "\n".join(line.strip() for line in clean_text.split("\n") if line.strip())
            return clean_text
    except requests.exceptions.RequestException as exc:
        logger.warning("[NCBI abstract] Failed for PMID %s: %s", pmid, exc)
    return None


def _download_file(url: str, output_path: Path, session: requests.Session) -> bool:
    """Download a file from *url* to *output_path* with streaming."""
    try:
        response = session.get(url, timeout=TIMEOUT_PDF_DOWNLOAD, stream=True)
        response.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=PDF_DOWNLOAD_CHUNK_SIZE):
                f.write(chunk)

        # Validate that the downloaded content is actually a PDF
        with open(output_path, "rb") as f:
            header = f.read(len(_PDF_MAGIC))
        if not header.startswith(_PDF_MAGIC):
            logger.warning("Downloaded content from %s is not a PDF (header: %r)", url, header[:16])
            with contextlib.suppress(OSError):
                output_path.unlink()
            return False

        return True
    except requests.exceptions.RequestException as exc:
        logger.warning("Download failed for %s: %s", url, exc)
        with contextlib.suppress(OSError):
            output_path.unlink()
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def download_paper_for_pdb(
    pdb_id: str,
    *,
    session: requests.Session | None = None,
    email: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Download the paper for a single PDB ID.

    Return the download log entry dict.
    """
    cfg = get_config()
    pdb_id = pdb_id.upper()
    final_pdf = cfg.papers_dir / f"{pdb_id}.pdf"
    enriched_path = cfg.enriched_dir / f"{pdb_id}.json"

    now = datetime.now(UTC).isoformat()

    # Input guard
    if not enriched_path.exists():
        logger.warning("[%s] Enriched data not found, skipping", pdb_id)
        entry: dict[str, Any] = {
            "status": "skipped_no_enriched_data",
            "source": None,
            "file_path": None,
            "doi": None,
            "pmid": None,
            "pmcid": None,
            "timestamp": now,
        }
        _update_download_log(pdb_id, entry)
        return entry

    # Resumability
    if final_pdf.exists() and not force:
        logger.info("[%s] PDF already exists, skipping", pdb_id)
        entry = {
            "status": "skipped_already_downloaded",
            "source": None,
            "file_path": str(final_pdf),
            "doi": None,
            "pmid": None,
            "pmcid": None,
            "timestamp": now,
        }
        _update_download_log(pdb_id, entry)
        return entry

    # Read enriched data
    try:
        with open(enriched_path, encoding="utf-8") as f:
            pdb_data: dict[str, Any] = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("[%s] Failed to read enriched JSON: %s", pdb_id, exc)
        entry = {
            "status": "failed_no_data",
            "source": None,
            "file_path": None,
            "doi": None,
            "pmid": None,
            "pmcid": None,
            "timestamp": now,
        }
        _update_download_log(pdb_id, entry)
        return entry

    # Resolve session
    resolved_email = email or os.environ.get("GPCR_EMAIL_FOR_APIS") or ""
    if not resolved_email:
        logger.error("GPCR_EMAIL_FOR_APIS is not set")
        entry = {
            "status": "failed_no_data",
            "source": None,
            "file_path": None,
            "doi": None,
            "pmid": None,
            "pmcid": None,
            "timestamp": now,
        }
        _update_download_log(pdb_id, entry)
        return entry

    sess = session or _build_session(resolved_email)

    # Extract identifiers from enriched data
    entry_data = (pdb_data.get("data") or {}).get("entry") or {}
    doi = (entry_data.get("rcsb_primary_citation") or {}).get("pdbx_database_id_DOI")
    pmid = (entry_data.get("rcsb_entry_container_identifiers") or {}).get("pubmed_id")
    pmcid = (entry_data.get("pubmed") or {}).get("rcsb_pubmed_central_id")

    if not doi:
        logger.info("[%s] No DOI found", pdb_id)
        entry = {
            "status": "failed_no_doi",
            "source": None,
            "file_path": None,
            "doi": None,
            "pmid": pmid,
            "pmcid": pmcid,
            "timestamp": now,
        }
        _update_download_log(pdb_id, entry)
        return entry

    # Tier 0: CrossRef metadata enrichment
    crossref = _fetch_crossref_metadata(doi, sess)
    pmid = crossref.get("pmid") or pmid
    pmcid = crossref.get("pmcid") or pmcid

    # Tier 1: Unpaywall
    pdf_url: str | None = None
    source: str | None = None

    pdf_url = _fetch_unpaywall_pdf_url(doi, sess)
    if pdf_url:
        source = "unpaywall_pdf"

    # Tier 2: NCBI PMC OA
    if not pdf_url and pmcid:
        pdf_url = _fetch_pmc_oa_pdf_url(str(pmcid), sess)
        if pdf_url:
            source = "ncbi_pmc_oa_pdf"

    # Download if we have a URL
    if pdf_url:
        cfg.papers_dir.mkdir(parents=True, exist_ok=True)
        temp_pdf = cfg.papers_dir / f"{pdb_id}_temp.pdf"

        if _download_file(pdf_url, temp_pdf, sess):
            os.replace(str(temp_pdf), str(final_pdf))
            logger.info("[%s] Downloaded PDF → %s", pdb_id, final_pdf)
            entry = {
                "status": "success_pdf_downloaded",
                "source": source,
                "file_path": str(final_pdf),
                "doi": doi,
                "pmid": pmid,
                "pmcid": pmcid,
                "timestamp": now,
            }
            _update_download_log(pdb_id, entry)
            return entry
        else:
            # Clean up temp file
            with contextlib.suppress(OSError):
                temp_pdf.unlink()

    # Tier 3: NCBI abstract fallback
    if pmid:
        abstract = _fetch_abstract_from_ncbi(str(pmid), sess)
        if abstract:
            abstracts_dir = cfg.papers_dir / "abstracts"
            abstracts_dir.mkdir(parents=True, exist_ok=True)
            abstract_path = abstracts_dir / f"{pdb_id}.txt"
            try:
                abstract_path.write_text(abstract, encoding="utf-8")
                logger.info("[%s] Saved abstract → %s", pdb_id, abstract_path)
                entry = {
                    "status": DL_STATUS_ABSTRACT_ONLY,
                    "source": "ncbi_abstract",
                    "file_path": str(abstract_path),
                    "doi": doi,
                    "pmid": pmid,
                    "pmcid": pmcid,
                    "timestamp": now,
                }
                _update_download_log(pdb_id, entry)
                return entry
            except OSError as exc:
                logger.warning("[%s] Failed to write abstract: %s", pdb_id, exc)

    # Fallback: paywalled
    logger.info("[%s] All download tiers failed, marking paywalled", pdb_id)
    entry = {
        "status": "fallback_paywalled",
        "source": None,
        "file_path": None,
        "doi": doi,
        "pmid": pmid,
        "pmcid": pmcid,
        "timestamp": now,
    }
    _update_download_log(pdb_id, entry)
    return entry
