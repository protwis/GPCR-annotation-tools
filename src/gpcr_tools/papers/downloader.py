"""Multi-tier PDF downloader for GPCR papers.

Trusts ONLY the DOI RCSB tags on the primary citation (no title-search or
citation-table recovery -- attaching the wrong paper is worse than none), then
tries candidate PDF URLs as a true fallback chain -- a URL that resolves but
yields a non-PDF (an HTML challenge)
or a 403/404 does not end the search, only an exhausted chain marks the paper
paywalled:
  CrossRef metadata — the article's PMID and a direct publisher PDF link
  Unpaywall — best OA PDF link
  PMC open-access S3 bucket — by a PMCID resolved from the DOI/PMID via the ID Converter
  Fallback — mark ``"fallback_paywalled"`` (left for manual download)

Reads enriched JSON from ``enriched/{pdb_id}.json``, writes PDFs to
``papers/{pdb_id}.pdf``, and updates ``state/download_log.json``
via atomic write after each PDB.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from gpcr_tools.config import (
    CROSSREF_API_URL,
    DL_STATUS_FAILED_NO_DATA,
    DL_STATUS_FAILED_NO_DOI,
    DL_STATUS_PAYWALLED,
    DL_STATUS_SKIPPED_EXISTS,
    DL_STATUS_SKIPPED_NO_ENRICHED,
    DL_STATUS_SUCCESS,
    HTTP_RETRY_ALLOWED_METHODS,
    HTTP_RETRY_BACKOFF_FACTOR,
    HTTP_RETRY_CONNECT,
    HTTP_RETRY_READ,
    HTTP_RETRY_STATUS_FORCELIST,
    HTTP_RETRY_TOTAL,
    NCBI_IDCONV_URL,
    PDF_DOWNLOAD_CHUNK_SIZE,
    PDF_MIN_VALID_BYTES,
    PMC_S3_BASE_URL,
    SLEEP_NCBI_RATE_LIMIT,
    TIMEOUT_CROSSREF,
    TIMEOUT_NCBI_PMC_OA,
    TIMEOUT_PDF_DOWNLOAD,
    TIMEOUT_UNPAYWALL,
    UNPAYWALL_API_URL,
    get_config,
)
from gpcr_tools.papers.storage import canonical_pdf_path, resolve_pdf_path

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
    """Tier 0: CrossRef metadata -- the article's PMID and a direct PDF link.

    Many gold/hybrid-OA publishers (notably Nature-brand journals) expose a direct
    ``application/pdf`` link in CrossRef even when Unpaywall carries no PDF URL, so
    capturing it here is the single largest DOI-only recovery.

    Deliberately does NOT extract a PMCID from ``link[]``: a CrossRef link can
    point at a *reference's* PMC article, not this one, and a wrong PMCID would
    fetch the WRONG paper. The PMCID is resolved authoritatively from the DOI via
    the ID Converter (``_resolve_pmcid``) instead.
    """
    url = f"{CROSSREF_API_URL}/{doi}"
    try:
        response = session.get(url, timeout=TIMEOUT_CROSSREF)
        if response.status_code == 200:
            data = response.json().get("message") or {}
            pmid = data.get("PMID")
            pdf_url: str | None = None
            for link in data.get("link") or []:
                link_url = link.get("URL") or ""
                content_type = link.get("content-type") or ""
                if pdf_url is None and (
                    "application/pdf" in content_type or link_url.lower().endswith(".pdf")
                ):
                    pdf_url = link_url
            return {"pmid": pmid, "crossref_pdf_url": pdf_url}
    except requests.exceptions.RequestException as exc:
        logger.warning("[CrossRef] Failed for DOI %s: %s", doi, exc)
    return {"pmid": None, "crossref_pdf_url": None}


def _fetch_unpaywall_pdf_url(
    doi: str,
    session: requests.Session,
    email: str | None = None,
) -> str | None:
    """Tier 1: Get OA PDF URL from Unpaywall."""
    url = f"{UNPAYWALL_API_URL}/{doi}"
    params: dict[str, str] = {}
    if email:
        params["email"] = email
    try:
        response = session.get(url, params=params, timeout=TIMEOUT_UNPAYWALL)
        if response.status_code == 200:
            data = response.json()
            oa_location = data.get("best_oa_location") or {}
            pdf_url = oa_location.get("url_for_pdf")
            if pdf_url:
                return pdf_url  # type: ignore[no-any-return]
    except requests.exceptions.RequestException as exc:
        logger.warning("[Unpaywall] Failed for DOI %s: %s", doi, exc)
    return None


def _fetch_pmc_s3_pdf_url(pmcid: str, session: requests.Session) -> str | None:
    """Resolve the article PDF in the PMC open-access S3 bucket.

    Replaces the retired NCBI OA FTP interface (the legacy oa_pdf paths now 404,
    and the web reader endpoint is behind a bot challenge). Confirms via the
    per-article metadata JSON that a PDF exists, then returns the canonical S3 PDF
    URL. ``None`` when the article is not in the open-access subset.
    """
    pmcid_norm = pmcid.upper()
    if not pmcid_norm.startswith("PMC"):
        pmcid_norm = f"PMC{pmcid_norm}"
    meta_url = f"{PMC_S3_BASE_URL}/metadata/{pmcid_norm}.1.json"
    try:
        response = session.get(meta_url, timeout=TIMEOUT_NCBI_PMC_OA)
        if response.status_code != 200:
            return None
        # ``pdf_url`` is the authoritative object path as an ``s3://bucket/key``
        # URI with an optional ``?md5=`` query; map it to the public HTTPS object
        # (this also carries the correct version, so no version is assumed). Null
        # when the article has no PDF in the open-access subset.
        pdf_field = (response.json() or {}).get("pdf_url")
        if not pdf_field:
            return None
        pdf_field = str(pdf_field)
        if pdf_field.startswith("s3://"):
            key = pdf_field.removeprefix("s3://").split("/", 1)[1].split("?", 1)[0]
            return f"{PMC_S3_BASE_URL}/{key}"
        if pdf_field.startswith("http"):
            return pdf_field.split("?", 1)[0]
        return f"{PMC_S3_BASE_URL}/{pdf_field.lstrip('/').split('?', 1)[0]}"
    except (requests.exceptions.RequestException, ValueError, IndexError) as exc:
        logger.warning("[PMC S3] Failed for PMCID %s: %s", pmcid, exc)
    return None


def _resolve_pmcid(doi: str | None, pmid: str | None, session: requests.Session) -> str | None:
    """Resolve a PMCID authoritatively via the NCBI ID Converter.

    Queries by DOI first (then PMID) so the PMC open-access download uses a PMCID
    that provably belongs to THIS paper -- never one guessed from a CrossRef
    reference link or an unverified metadata field, which could fetch the WRONG
    paper. Returns ``PMC...`` or ``None`` (no S3 attempt, the paper stays
    paywalled, which is safe).
    """
    for ident in (doi, pmid):
        if not ident:
            continue
        params = {"ids": str(ident), "format": "json", "tool": "litfetcher"}
        try:
            response = session.get(NCBI_IDCONV_URL, params=params, timeout=TIMEOUT_NCBI_PMC_OA)
            time.sleep(SLEEP_NCBI_RATE_LIMIT)  # the ID Converter rate-limits aggressively
            if response.status_code == 200:
                records = response.json().get("records") or []
                rec = records[0] if records else {}
                # Trust the PMCID only when the converter echoed back the exact id
                # we queried (it returns one record per id with ``requested-id``),
                # so the PMCID provably belongs to THIS paper and a future batched
                # query could never positionally cross-bind a wrong PMCID.
                if rec.get("pmcid") and str(rec.get("requested-id")) == str(ident):
                    return str(rec["pmcid"])
        except (requests.exceptions.RequestException, ValueError) as exc:
            logger.warning("[ID Converter] Failed for %s: %s", ident, exc)
    return None


def _download_file(url: str, output_path: Path, session: requests.Session) -> bool:
    """Download a file from *url* to *output_path* with streaming."""
    try:
        response = session.get(url, timeout=TIMEOUT_PDF_DOWNLOAD, stream=True)
        response.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=PDF_DOWNLOAD_CHUNK_SIZE):
                f.write(chunk)

        # Validate the downloaded content: real PDF magic AND a plausible size.
        # The magic check rejects HTML bot-challenge pages; the size floor rejects
        # tiny error stubs / a stream truncated after the first chunk (which would
        # otherwise be promoted to the final path and never retried).
        with open(output_path, "rb") as f:
            header = f.read(len(_PDF_MAGIC))
        size = output_path.stat().st_size
        if not header.startswith(_PDF_MAGIC) or size < PDF_MIN_VALID_BYTES:
            logger.warning(
                "Downloaded content from %s is not a valid PDF (header: %r, %d bytes)",
                url,
                header[:16],
                size,
            )
            with contextlib.suppress(OSError):
                output_path.unlink()
            return False

        return True
    except (requests.exceptions.RequestException, OSError) as exc:
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
    enriched_path = cfg.enriched_dir / f"{pdb_id}.json"

    now = datetime.now(UTC).isoformat()

    # Input guard
    if not enriched_path.exists():
        logger.warning("[%s] Enriched data not found, skipping", pdb_id)
        entry: dict[str, Any] = {
            "status": DL_STATUS_SKIPPED_NO_ENRICHED,
            "source": None,
            "file_path": None,
            "doi": None,
            "pmid": None,
            "pmcid": None,
            "timestamp": now,
        }
        _update_download_log(pdb_id, entry)
        return entry

    # Resumability — a paper already on disk under EITHER the canonical DOI name
    # (shared by same-DOI siblings) or the legacy per-PDB name counts as present,
    # so a sibling's already-downloaded canonical file is not re-fetched.
    existing_pdf = resolve_pdf_path(pdb_id, _read_download_log())
    if existing_pdf is not None and not force:
        logger.info("[%s] PDF already exists, skipping", pdb_id)
        # Preserve any identifiers a prior run already resolved -- the log is
        # full-replace, so nulling them here would lose the DOI that same-paper
        # sibling grouping (manual workflow Phase 1) depends on across re-runs.
        prior = _read_download_log().get(pdb_id)
        prior = prior if isinstance(prior, dict) else {}
        entry = {
            "status": DL_STATUS_SKIPPED_EXISTS,
            "source": None,
            "file_path": str(existing_pdf),
            "doi": prior.get("doi"),
            "pmid": prior.get("pmid"),
            "pmcid": prior.get("pmcid"),
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
            "status": DL_STATUS_FAILED_NO_DATA,
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
            "status": DL_STATUS_FAILED_NO_DATA,
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
    # PMCID is NOT read from metadata here: the enriched ``rcsb_pubmed_central_id``
    # and CrossRef links can carry a wrong/reference PMCID. It is resolved
    # authoritatively from the DOI only if/when the PMC-S3 tier is reached.
    pmcid: str | None = None

    if not doi:
        # Trust ONLY the DOI RCSB itself tags on the primary citation. No title
        # search / heuristic recovery: attaching the wrong paper is worse than
        # none, so a missing DOI falls straight through to the manual workflow.
        logger.info("[%s] No DOI found", pdb_id)
        entry = {
            "status": DL_STATUS_FAILED_NO_DOI,
            "source": None,
            "file_path": None,
            "doi": None,
            "pmid": pmid,
            "pmcid": pmcid,
            "timestamp": now,
        }
        _update_download_log(pdb_id, entry)
        return entry

    # Save to the canonical DOI-named file (one per paper, shared by every
    # same-DOI sibling) when DOI storage is on; otherwise the per-PDB name. The
    # DOI is confirmed present above, so siblings collapse onto one physical file.
    final_pdf = canonical_pdf_path(pdb_id, doi)

    # Tier 0: CrossRef metadata (the article's PMID + a direct publisher PDF link).
    crossref = _fetch_crossref_metadata(doi, sess)
    pmid = crossref.get("pmid") or pmid

    cfg.papers_dir.mkdir(parents=True, exist_ok=True)
    temp_pdf = cfg.papers_dir / f"{pdb_id}_temp.pdf"

    # A sibling may already have downloaded this paper to the same canonical file
    # while this PDB was being processed; if so, don't re-download (idempotent).
    if final_pdf.exists() and not force:
        entry = {
            "status": DL_STATUS_SKIPPED_EXISTS,
            "source": None,
            "file_path": str(final_pdf),
            "doi": doi,
            "pmid": pmid,
            "pmcid": pmcid,
            "timestamp": now,
        }
        _update_download_log(pdb_id, entry)
        return entry

    # Ordered candidate resolvers, tried as a TRUE fallback chain: a URL that
    # resolves but yields a non-PDF (e.g. an HTML bot challenge) or a 403/404 does
    # NOT end the search -- only an exhausted chain marks the paper paywalled.
    def _pmc_s3_url() -> str | None:
        # Resolve the PMCID authoritatively from the DOI/PMID (never an unverified
        # field), so the S3 download cannot fetch a different paper. The resolved
        # PMCID is what gets logged. No resolution -> no S3 attempt -> paywalled.
        nonlocal pmcid
        pmcid = _resolve_pmcid(doi, str(pmid) if pmid else None, sess)
        return _fetch_pmc_s3_pdf_url(pmcid, sess) if pmcid else None

    resolvers: list[tuple[str, Any]] = [
        ("crossref_pdf", lambda: crossref.get("crossref_pdf_url")),
        ("unpaywall_pdf", lambda: _fetch_unpaywall_pdf_url(doi, sess, email=resolved_email)),
        ("pmc_s3_pdf", _pmc_s3_url),
    ]
    for source, resolve in resolvers:
        pdf_url = resolve()
        if not pdf_url:
            continue
        if _download_file(pdf_url, temp_pdf, sess):
            os.replace(str(temp_pdf), str(final_pdf))
            logger.info("[%s] Downloaded PDF → %s (%s)", pdb_id, final_pdf, source)
            entry = {
                "status": DL_STATUS_SUCCESS,
                "source": source,
                "file_path": str(final_pdf),
                "doi": doi,
                "pmid": pmid,
                "pmcid": pmcid,
                "timestamp": now,
            }
            _update_download_log(pdb_id, entry)
            return entry
        with contextlib.suppress(OSError):
            temp_pdf.unlink()

    # Every open-access route exhausted. We deliberately do NOT fall back to an
    # abstract: the annotator needs the full paper, so the PDB is left paywalled
    # for manual download (the watcher ingests a hand-dropped PDF) and otherwise
    # skipped downstream.
    logger.info("[%s] No open-access PDF available, marking paywalled", pdb_id)
    entry = {
        "status": DL_STATUS_PAYWALLED,
        "source": None,
        "file_path": None,
        "doi": doi,
        "pmid": pmid,
        "pmcid": pmcid,
        "timestamp": now,
    }
    _update_download_log(pdb_id, entry)
    return entry
