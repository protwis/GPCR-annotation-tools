"""Annotation runner -- single-PDB, batch submission, and recovery."""

from __future__ import annotations

import concurrent.futures
import contextlib
import json
import logging
import os
import tempfile
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from google.genai.errors import APIError

from gpcr_tools.annotator.detect_orchestrator import build_tool_config, build_tool_for_signals
from gpcr_tools.annotator.gemini_client import get_client
from gpcr_tools.annotator.pdf_compressor import compress_pdf_if_needed
from gpcr_tools.annotator.post_processor import post_process_annotation
from gpcr_tools.annotator.prompt_builder import build_prompt_parts
from gpcr_tools.annotator.schema import ANNOTATION_TOOL
from gpcr_tools.code_version import get_code_version
from gpcr_tools.config import (
    ANNOTATOR_FUNCTION_NAME,
    BATCH_REGISTRY_VERSION,
    BATCH_STATUS_DOWNLOADED,
    BATCH_STATUS_FAILED,
    BATCH_STATUS_RECOVERED,
    BATCH_STATUS_SUBMITTED,
    CLOUD_CLEANUP,
    GEMINI_BASE_BACKOFF,
    GEMINI_BATCH_MAX_REQUESTS,
    GEMINI_BATCH_PACK_REQUESTS,
    GEMINI_DEFAULT_RUNS,
    GEMINI_FILE_TTL_HOURS,
    GEMINI_MAX_RETRIES,
    GEMINI_MAX_WORKERS,
    SLEEP_GEMINI_429,
    UPLOAD_DEDUP,
    get_config,
    get_gemini_model_name,
    model_run_subdir,
    sanitize_doi,
)
from gpcr_tools.detector.signals import SEVERITY_ADVISORY
from gpcr_tools.detector.stage import load_detect_signals
from gpcr_tools.papers.storage import content_hash, resolve_doi, resolve_pdf_path

logger = logging.getLogger(__name__)


def _registry_fresh_uri(entry: Any, now: datetime) -> str | None:
    """Return a still-valid cached upload URI from a registry *entry*, or None.

    Entries are ``{"uri": ..., "uploaded_at": <iso>}``. A URI older than the
    Files-API TTL — or a legacy bare-string entry whose age is unknown — is
    treated as expired so the caller re-uploads instead of embedding a dead
    fileUri into the batch request.
    """
    if isinstance(entry, dict):
        uri = entry.get("uri")
        uploaded_at = entry.get("uploaded_at")
        if uri and uploaded_at:
            try:
                age = now - datetime.fromisoformat(uploaded_at)
            except ValueError:
                return None
            if age < timedelta(hours=GEMINI_FILE_TTL_HOURS):
                return str(uri)
    return None


def _safe_job_name(job_name: str) -> str:
    """Filesystem-safe token for a provider job name (names contain '/')."""
    return job_name.replace("/", "_")


def _read_download_log_safe(config: Any) -> dict[str, Any]:
    """Read the download log (DOI source for upload dedup), tolerant of absence."""
    path = config.download_log_file
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _resolve_upload(
    client: Any,
    config: Any,
    registry: dict[str, Any],
    upload_key: str,
    pdb_id: str,
    pdf_file: Path,
    doi: str,
    now: datetime,
) -> tuple[str | None, str | None]:
    """Return ``(fileUri, file_name)`` for *pdf_file*, reusing a cached upload when safe.

    A cached upload under *upload_key* is reused only when (a) it is within the
    Files-API TTL AND (b) the safety gate passes — the cached entry's recorded
    content hash equals this PDF's hash, OR (lacking a stored hash) the entry's
    DOI matches. On a hash disagreement the cache is NOT reused: a fresh per-PDB
    upload is made instead (fail safe — never serve the wrong bytes to a sibling),
    and the disagreement is logged. On any upload error returns ``(None, None)``.
    """
    this_hash = content_hash(pdf_file)
    cached = registry.get(upload_key)
    cached_uri = _registry_fresh_uri(cached, now)
    if cached_uri and isinstance(cached, dict):
        cached_hash = cached.get("content_hash")
        # Byte-identity is the strong gate; same-DOI is the fallback when an older
        # entry carries no hash. A recorded hash that disagrees fails the gate.
        if cached_hash and this_hash and cached_hash != this_hash:
            logger.warning(
                "[%s] Cached upload for key %s has a different content hash than this "
                "PDF; uploading separately (fail-safe).",
                pdb_id,
                upload_key,
            )
        else:
            return cached_uri, cached.get("name")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_pdf = Path(tmp_dir) / f"{pdb_id}_compressed.pdf"
        try:
            actual_pdf = compress_pdf_if_needed(pdf_file, tmp_pdf)
            uploaded_file = client.files.upload(
                file=str(actual_pdf), config={"mime_type": "application/pdf"}
            )
        except Exception as e:
            logger.error("[%s] Failed to upload PDF: %s", pdb_id, e)
            return None, None

    # Stamp the REAL upload time (not the submit-time ``now``) so the TTL check
    # measures the file's actual age; record the deletable file ``name``, the DOI,
    # and the content hash so a sibling can verify byte-identity before reuse.
    registry[upload_key] = {
        "uri": uploaded_file.uri,
        "name": uploaded_file.name,
        "uploaded_at": datetime.now(UTC).isoformat(),
        "doi": doi or None,
        "content_hash": this_hash,
    }
    logger.info("[%s] Uploaded PDF to %s (key %s)", pdb_id, uploaded_file.uri, upload_key)
    return uploaded_file.uri, uploaded_file.name


def _load_job_registry(config: Any) -> dict[str, Any]:
    """Return the batch-job registry, or a fresh empty one.

    The registry (``state/batch_jobs.json``) tracks every submitted batch job
    keyed by job name, so a sharded submission's multiple jobs are all
    trackable and recoverable. Tolerant: a missing or corrupt file yields an
    empty registry rather than raising.
    """
    reg_file = config.batch_jobs_registry_file
    if reg_file.exists():
        loaded: Any = None
        try:
            loaded = json.loads(reg_file.read_text())
        except (json.JSONDecodeError, OSError):
            loaded = None
        if isinstance(loaded, dict) and isinstance(loaded.get("jobs"), dict):
            return loaded
    return {"version": BATCH_REGISTRY_VERSION, "jobs": {}}


def _save_job_registry(config: Any, registry: dict[str, Any]) -> None:
    """Persist the job registry atomically (tmp + os.replace)."""
    reg_file = config.batch_jobs_registry_file
    os.makedirs(reg_file.parent, exist_ok=True)
    tmp = reg_file.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(registry, f, indent=2)
    os.replace(tmp, reg_file)


def _register_job(config: Any, entry: dict[str, Any]) -> None:
    """Insert or replace one job entry, keyed by ``entry['job_name']``."""
    registry = _load_job_registry(config)
    registry["jobs"][entry["job_name"]] = entry
    _save_job_registry(config, registry)


def _update_job_status(config: Any, job_name: str, **fields: Any) -> None:
    """Patch fields on one job entry (no-op if the job is unknown)."""
    registry = _load_job_registry(config)
    entry = registry["jobs"].get(job_name)
    if entry is None:
        return
    entry.update(fields)
    _save_job_registry(config, registry)


def _files_referenced_by_live_jobs(registry: dict[str, Any], exclude_job: str) -> set[str]:
    """File names still referenced by a non-terminal (live) job other than
    *exclude_job*.

    Only RECOVERED jobs are treated as released. A FAILED job's inputs are still
    counted as "live" here, so a file it shares is not deleted by another job's
    cleanup — it ages out later via the TTL orphan sweep. This is the ref-count
    gate: a shared file is deleted only when no still-live job (other than the one
    being cleaned up) references it. Conservative — it over-retains, never deletes
    a file a live job might still need.
    """
    live: set[str] = set()
    for name, entry in registry.get("jobs", {}).items():
        if name == exclude_job:
            continue
        if not isinstance(entry, dict):
            continue
        if entry.get("status") == BATCH_STATUS_RECOVERED:
            continue
        for fname in entry.get("uploaded_file_names") or []:
            if fname:
                live.add(str(fname))
        src = entry.get("batch_src_file_name")
        if src:
            live.add(str(src))
    return live


def _delete_remote_file(client: Any, name: str) -> None:
    """Best-effort delete of one Files-API file; a failure is logged, never raised."""

    with contextlib.suppress(Exception):
        client.files.delete(name=name)
        logger.info("Deleted uploaded file %s", name)


def _cleanup_terminal_job_uploads(config: Any, client: Any, job_name: str) -> None:
    """Delete a terminal job's uploaded inputs, ref-counted across jobs.

    The job's JSONL source is its own (never shared) so it is always deleted; each
    per-PDB PDF upload is deleted only when no other still-live job references it.
    Best-effort: a delete failure never propagates. No-op when cleanup is disabled.
    """
    if not CLOUD_CLEANUP:
        return
    registry = _load_job_registry(config)
    entry = registry.get("jobs", {}).get(job_name)
    if not isinstance(entry, dict):
        return
    # Idempotent: a job whose uploads were already released is skipped, so a
    # repeated poll does not re-issue deletes for the same (now-gone) files.
    if entry.get("uploads_cleaned"):
        return
    still_referenced = _files_referenced_by_live_jobs(registry, exclude_job=job_name)
    for fname in entry.get("uploaded_file_names") or []:
        if fname and str(fname) not in still_referenced:
            _delete_remote_file(client, str(fname))
    src = entry.get("batch_src_file_name")
    if src:
        _delete_remote_file(client, str(src))
    _update_job_status(config, job_name, uploads_cleaned=True)


def _sweep_orphan_uploads(config: Any, client: Any) -> None:
    """Best-effort sweep of orphaned Files-API uploads at submit start.

    Deletes only files older than the TTL AND not referenced by any non-terminal
    job, so a file an in-flight job still needs is never removed. Guarded so any
    failure (listing or deleting) never aborts the submission that follows.
    """

    registry = _load_job_registry(config)
    referenced = _files_referenced_by_live_jobs(registry, exclude_job="")
    now = datetime.now(UTC)
    try:
        files = list(client.files.list())
    except Exception as exc:
        logger.warning("Orphan sweep skipped — could not list uploaded files: %s", exc)
        return
    for f in files:
        name = getattr(f, "name", None)
        if not name or str(name) in referenced:
            continue
        created = getattr(f, "create_time", None)
        # Only delete a file we can prove is past the TTL; an unknown age is left
        # alone (fail safe — never delete a file that might still be in use).
        if created is None:
            continue
        try:
            created_dt = (
                created if isinstance(created, datetime) else datetime.fromisoformat(str(created))
            )
            # A naive timestamp would make the aware-`now` subtraction raise — and
            # the sweep must never abort the submission, so normalise and keep the
            # comparison inside the guard.
            if created_dt.tzinfo is None:
                created_dt = created_dt.replace(tzinfo=UTC)
            within_ttl = now - created_dt < timedelta(hours=GEMINI_FILE_TTL_HOURS)
        except (ValueError, TypeError):
            continue
        if within_ttl:
            continue
        with contextlib.suppress(Exception):
            client.files.delete(name=str(name))
            logger.info("Swept orphaned upload %s", name)


def _chunk_request_groups(
    groups: list[list[dict[str, Any]]], max_per_chunk: int
) -> list[list[dict[str, Any]]]:
    """Pack per-PDB request *groups* into chunks of at most *max_per_chunk*.

    A single PDB's runs are never split across chunks; a group larger than the
    cap becomes its own (over-cap) chunk rather than being divided.
    """
    chunks: list[list[dict[str, Any]]] = []
    chunk: list[dict[str, Any]] = []
    for reqs in groups:
        if chunk and len(chunk) + len(reqs) > max_per_chunk:
            chunks.append(chunk)
            chunk = []
        chunk.extend(reqs)
        if len(chunk) >= max_per_chunk:
            chunks.append(chunk)
            chunk = []
    if chunk:
        chunks.append(chunk)
    return chunks


def _chunk_group_metas(
    groups: list[dict[str, Any]], max_per_chunk: int
) -> list[list[dict[str, Any]]]:
    """Pack per-PDB group *metas* (``{pdb_id, requests, file_name}``) into chunks
    of at most *max_per_chunk* requests.

    Same invariant as :func:`_chunk_request_groups` — a single PDB's runs are
    never split, an over-cap group becomes its own chunk — but it preserves each
    group's metadata (the uploaded file name) so a chunk carries the names of the
    inputs its job references.
    """
    chunks: list[list[dict[str, Any]]] = []
    chunk: list[dict[str, Any]] = []
    chunk_size = 0
    for group in groups:
        n = len(group["requests"])
        if chunk and chunk_size + n > max_per_chunk:
            chunks.append(chunk)
            chunk = []
            chunk_size = 0
        chunk.append(group)
        chunk_size += n
        if chunk_size >= max_per_chunk:
            chunks.append(chunk)
            chunk = []
            chunk_size = 0
    if chunk:
        chunks.append(chunk)
    return chunks


def _order_groups_by_paper(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Order per-PDB groups so same-paper PDBs are adjacent (packed into one job).

    Same-paper PDBs share an uploaded fileUri (see the upload-dedup path), so
    placing them adjacently lets the chunker keep them in one job, which keeps a
    shared upload referenced by the fewest jobs. Groups are keyed by the shared
    fileUri's file ``name`` (the dedup key); a group with no recorded name keeps
    its original relative order. Order within a paper is preserved (stable).
    """
    order: dict[str, int] = {}
    for group in groups:
        key = group.get("file_name") or group["pdb_id"]
        if key not in order:
            order[key] = len(order)
    return sorted(groups, key=lambda g: order[g.get("file_name") or g["pdb_id"]])


def _submit_batch_chunk(
    config: Any,
    client: Any,
    *,
    model_name: str,
    prompt_id: str | None,
    chunk_requests: list[dict[str, Any]],
    chunk_index: int,
    chunk_count: int,
    detect_advisory_by_pdb: dict[str, list[str]],
    created_at: str,
    uploaded_file_names: list[str] | None = None,
) -> str:
    """Submit one chunk as a batch job and register it; return the job name.

    Raises on upload/create failure so the caller can isolate a single chunk's
    failure; the temp JSONL is always cleaned up. The job entry records the
    Files-API names of the inputs this job references (the per-PDB PDF uploads in
    *uploaded_file_names* plus the JSONL source), so terminal cleanup can delete
    them once no non-terminal job still references them.
    """
    os.makedirs(config.pipeline_runs_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".jsonl") as f:
        for req in chunk_requests:
            f.write(json.dumps(req) + "\n")
        tmp_jsonl = Path(f.name)

    try:
        batch_src_file = client.files.upload(
            file=str(tmp_jsonl), config={"mime_type": "application/jsonl"}
        )
        if not batch_src_file.name:
            raise ValueError("Uploaded file has no name")

        batch_job = client.batches.create(model=model_name, src=batch_src_file.name)
        if not batch_job.name:
            raise ValueError("Created batch job has no name")
        logger.info(
            "Batch chunk %d/%d submitted: %s (%d requests)",
            chunk_index + 1,
            chunk_count,
            batch_job.name,
            len(chunk_requests),
        )

        # The registry carries the provenance (model / prompt / code_version /
        # advisories) that recover_batch stamps onto each result, so a job's
        # results are always attributed to the model that produced them, with
        # no dependence on a shared sidecar a later submission could overwrite.
        _register_job(
            config,
            {
                "job_name": batch_job.name,
                "status": BATCH_STATUS_SUBMITTED,
                "model_requested": model_name,
                "prompt": prompt_id,
                "code_version": get_code_version(),
                "chunk_index": chunk_index,
                "chunk_count": chunk_count,
                "request_count": len(chunk_requests),
                "created_at": created_at,
                "raw_output_file": None,
                "recovered_at": None,
                "detect_advisory": detect_advisory_by_pdb,
                # Uploaded inputs to delete once the job is terminal (ref-counted):
                # the per-PDB PDF uploads + this job's JSONL source.
                "uploaded_file_names": sorted(uploaded_file_names or []),
                "batch_src_file_name": batch_src_file.name,
            },
        )

        # Back-compat: keep the single-file pointer to the most recent job so a
        # rollback to pre-registry code still finds a job to poll. The registry
        # is authoritative; this mirror is deprecated.
        tmp_job_file = config.current_batch_job_file.with_suffix(".tmp")
        with open(tmp_job_file, "w") as fj:
            fj.write(batch_job.name)
        os.replace(tmp_job_file, config.current_batch_job_file)
        return str(batch_job.name)
    finally:
        if tmp_jsonl.exists():
            os.remove(tmp_jsonl)


def run_single_pdb(
    pdb_id: str,
    enriched_data: dict,
    prompt_text: str,
    pdf_path: Path,
    num_runs: int = GEMINI_DEFAULT_RUNS,
    model_name: str | None = None,
    prompt_id: str | None = None,
    temperature: float | None = None,
    thinking_level: str | None = None,
) -> None:
    """Run annotation for a single PDB entry using parallel Gemini calls.

    Uploads the PDF once, then fans out *num_runs* independent generation
    requests via a thread pool.  Completed runs are persisted atomically
    so the process is safely resumable.
    """
    model_name = model_name or get_gemini_model_name()
    config = get_config()
    out_dir = config.ai_results_dir / pdb_id / model_run_subdir(model_name)

    # Check resumability
    os.makedirs(out_dir, exist_ok=True)
    completed_runs = 0
    for n in range(1, num_runs + 1):
        if (out_dir / f"run_{n}.json").exists():
            completed_runs += 1

    if completed_runs >= num_runs:
        logger.info("[%s] All %d runs already completed. Skipping.", pdb_id, num_runs)
        return

    client = get_client()

    # Compress PDF if needed
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_pdf = Path(tmp_dir) / f"{pdb_id}_compressed.pdf"
        try:
            actual_pdf = compress_pdf_if_needed(pdf_path, tmp_pdf)
        except Exception as e:
            logger.error("[%s] PDF compression failed: %s", pdb_id, e)
            return

        # Upload PDF
        try:
            uploaded_file = client.files.upload(
                file=str(actual_pdf), config={"mime_type": "application/pdf"}
            )
        except Exception as e:
            logger.error("[%s] Failed to upload PDF: %s", pdb_id, e)
            return

        try:
            # Detect signals route advisory evidence into the prompt + tool; with
            # none (or only review signals) the prompt and config are unchanged.
            detect_signals = load_detect_signals(pdb_id)
            advisory_kinds = sorted(
                {s.kind for s in detect_signals if s.severity == SEVERITY_ADVISORY}
            )
            parts = build_prompt_parts(
                pdb_id, enriched_data, prompt_text, detect_signals=detect_signals
            )
            run_config = build_tool_config(
                detect_signals, temperature=temperature, thinking_level=thinking_level
            )
            contents: list[Any] = [*parts, uploaded_file]

            def do_run(run_num: int) -> None:
                out_file = out_dir / f"run_{run_num}.json"
                if out_file.exists():
                    return

                retries = 0
                while retries < GEMINI_MAX_RETRIES:
                    try:
                        # get a potentially rotated client
                        run_client = get_client()
                        response = run_client.models.generate_content(
                            model=model_name,
                            contents=contents,
                            config=run_config,
                        )

                        if not response.function_calls:
                            raise ValueError("No function calls returned by the model")

                        # Extract the first function call
                        fc = response.function_calls[0]
                        if fc.name != ANNOTATOR_FUNCTION_NAME:
                            raise ValueError(f"Unexpected function call: {fc.name}")

                        args = fc.args
                        if args is None:
                            raise ValueError("Function call missing arguments")

                        # Process and save
                        final_data = post_process_annotation(args)
                        final_data["_provenance"] = {
                            "model_requested": model_name,
                            "model_served": getattr(response, "model_version", None),
                            "prompt": prompt_id,
                            "code_version": get_code_version(),
                            "detect_advisory": advisory_kinds,
                            "run": run_num,
                            "mode": "single",
                            "timestamp": datetime.now(UTC).isoformat(),
                        }

                        # Atomic write
                        tmp_out = out_file.with_suffix(".tmp")
                        with open(tmp_out, "w") as f:
                            json.dump(final_data, f, indent=2)
                        os.replace(tmp_out, out_file)
                        logger.info("[%s] Run %d complete.", pdb_id, run_num)
                        return

                    except APIError as e:
                        retries += 1
                        if e.code == 429:
                            # Rate-limited — longer sleep before retry
                            time.sleep(SLEEP_GEMINI_429 * (2 ** (retries - 1)))
                        else:
                            time.sleep(GEMINI_BASE_BACKOFF * (2 ** (retries - 1)))
                    except Exception as exc:
                        logger.warning(
                            "[%s] Run %d attempt %d failed: %s",
                            pdb_id,
                            run_num,
                            retries + 1,
                            exc,
                        )
                        retries += 1
                        time.sleep(GEMINI_BASE_BACKOFF * (2 ** (retries - 1)))

                logger.error(
                    "[%s] Run %d failed after %d retries.", pdb_id, run_num, GEMINI_MAX_RETRIES
                )

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(num_runs, GEMINI_MAX_WORKERS)
            ) as executor:
                # Consume the iterator so any exception escaping a worker
                # surfaces here instead of being silently discarded.
                list(executor.map(do_run, range(1, num_runs + 1)))

        finally:
            with contextlib.suppress(Exception):
                if uploaded_file.name:
                    client.files.delete(name=uploaded_file.name)


def build_and_submit_batch(
    targets: list[str],
    prompt_text: str,
    num_runs: int = GEMINI_DEFAULT_RUNS,
    model_name: str | None = None,
    prompt_id: str | None = None,
    temperature: float | None = None,
    thinking_level: str | None = None,
) -> None:
    """Build a JSONL payload for all *targets* and submit it to the Gemini Batch API."""
    model_name = model_name or get_gemini_model_name()
    config = get_config()
    client = get_client()

    # Best-effort sweep of orphaned Files-API uploads before adding more, so a
    # long-running corpus submission doesn't accumulate past the 20 GB cap. Never
    # aborts submission (guarded inside the helper).
    if CLOUD_CLEANUP:
        _sweep_orphan_uploads(config, client)

    # Prepare batch requests
    now = datetime.now(UTC)
    # Each group carries the PDB id, its requests, and the Files-API file *name*
    # of the PDF it references (None for a TTL-cached reuse whose name predates
    # this field), so terminal cleanup can delete a job's uploaded inputs.
    request_groups: list[dict[str, Any]] = []
    registry = {}

    # Check if uploaded files registry exists
    reg_file = config.uploaded_files_registry_file
    if reg_file.exists():
        try:
            with open(reg_file) as f:
                registry = json.load(f)
        except json.JSONDecodeError:
            pass

    # The download log resolves a PDB's DOI (so same-paper PDBs share one upload).
    download_log = _read_download_log_safe(config)

    # Per-PDB advisory signal kinds, recorded into the job provenance so
    # recover_batch can stamp each result with the advisories active at submit.
    detect_advisory_by_pdb: dict[str, list[str]] = {}
    for pdb_id in targets:
        enriched_file = config.enriched_dir / f"{pdb_id}.json"
        pdf_file = resolve_pdf_path(pdb_id, download_log)

        if not enriched_file.exists() or pdf_file is None:
            logger.warning("[%s] Missing enriched data or PDF, skipping batch prep.", pdb_id)
            continue

        try:
            with open(enriched_file) as f:
                enriched_data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            # One unreadable enriched file must not abort the whole batch.
            logger.warning("[%s] Skipping — unreadable enriched JSON: %s", pdb_id, exc)
            continue

        # Determine runs to do
        out_dir = config.ai_results_dir / pdb_id / model_run_subdir(model_name)
        os.makedirs(out_dir, exist_ok=True)
        runs_to_do = [n for n in range(1, num_runs + 1) if not (out_dir / f"run_{n}.json").exists()]

        if not runs_to_do:
            continue

        # Dedup the upload by canonical paper: same-DOI PDBs share one fileUri.
        # The registry key is the sanitized DOI when dedup is on and a DOI exists,
        # else the PDB id (reproducing the per-PDB behaviour byte-for-byte).
        doi = resolve_doi(pdb_id, download_log) if UPLOAD_DEDUP else ""
        upload_key = f"doi:{sanitize_doi(doi)}" if (UPLOAD_DEDUP and doi) else pdb_id

        pdf_uri, pdf_file_name = _resolve_upload(
            client, config, registry, upload_key, pdb_id, pdf_file, doi, now
        )
        if not pdf_uri:
            continue

        detect_signals = load_detect_signals(pdb_id)
        detect_advisory_by_pdb[pdb_id] = sorted(
            {s.kind for s in detect_signals if s.severity == SEVERITY_ADVISORY}
        )
        parts = build_prompt_parts(
            pdb_id, enriched_data, prompt_text, detect_signals=detect_signals
        )
        tool_for_pdb = build_tool_for_signals(ANNOTATION_TOOL, detect_signals)

        # We need to construct the request dict for the batch API.
        # The schema for the batch API contents is identical to generate_content.
        # Requests are grouped per PDB so chunking never splits one structure's
        # runs across batch jobs.
        pdb_requests: list[dict[str, Any]] = []
        for n in runs_to_do:
            req_id = f"{pdb_id}__run_{n:02d}"

            # Construct the contents array. The File Data needs a specific format.
            contents_batch: list[dict[str, Any]] = []
            for part in parts:
                if isinstance(part, str):
                    contents_batch.append({"parts": [{"text": part}]})
            contents_batch.append(
                {"parts": [{"fileData": {"fileUri": pdf_uri, "mimeType": "application/pdf"}}]}
            )

            # The tool schema must be provided as a dict (per-PDB: augmented when
            # an incidental-candidate signal is present, identical to base otherwise).
            assert tool_for_pdb.function_declarations is not None
            fn_decl = tool_for_pdb.function_declarations[0]
            tool_dict = {
                "functionDeclarations": [
                    {
                        "name": fn_decl.name,
                        "description": fn_decl.description,
                        "parameters": fn_decl.parameters.model_dump(exclude_none=True)
                        if fn_decl.parameters
                        else {},
                    }
                ]
            }

            # Per-request payload. The model is set once at the batch-job level;
            # repeating it per request is rejected as a mismatch. Temperature and
            # thinking level, however, are per-request (generationConfig) -- each
            # omitted entirely when not set, so the default behaviour is unchanged.
            request_payload: dict[str, Any] = {
                "contents": contents_batch,
                "tools": [tool_dict],
                "toolConfig": {"functionCallingConfig": {"mode": "ANY"}},
            }
            generation_config: dict[str, Any] = {}
            if temperature is not None:
                generation_config["temperature"] = temperature
            if thinking_level is not None:
                generation_config["thinkingConfig"] = {"thinkingLevel": thinking_level.upper()}
            if generation_config:
                request_payload["generationConfig"] = generation_config

            # "key" (not "id") is echoed back in the output for correlation.
            pdb_requests.append({"key": req_id, "request": request_payload})

        if pdb_requests:
            request_groups.append(
                {"pdb_id": pdb_id, "requests": pdb_requests, "file_name": pdf_file_name}
            )

    # Save updated registry
    tmp_reg = reg_file.with_suffix(".tmp")
    with open(tmp_reg, "w") as f:
        json.dump(registry, f, indent=2)
    os.replace(tmp_reg, reg_file)

    total_requests = sum(len(group["requests"]) for group in request_groups)
    if not total_requests:
        logger.info("No batch requests to submit. All done!")
        return

    # When dedup is on, pack same-paper PDBs adjacently so they share a job (and a
    # single upload reference). The cap is the packing target; over-large papers
    # still span consecutive jobs. With dedup off, preserve target order and the
    # hard per-job ceiling, reproducing the previous behaviour.
    if UPLOAD_DEDUP:
        request_groups = _order_groups_by_paper(request_groups)
        pack_cap = GEMINI_BATCH_PACK_REQUESTS
    else:
        pack_cap = GEMINI_BATCH_MAX_REQUESTS

    # Shard into jobs so one oversized submission can't be rejected wholesale or
    # sit in the queue past the provider's 48-hour expiry. Each PDB's runs stay
    # within a single job; the file names a chunk references travel with it so
    # terminal cleanup can delete the job's uploaded inputs.
    chunks = _chunk_group_metas(request_groups, pack_cap)
    submitted = 0
    for chunk_index, chunk in enumerate(chunks):
        chunk_requests = [req for group in chunk for req in group["requests"]]
        chunk_file_names = sorted({group["file_name"] for group in chunk if group.get("file_name")})
        # Scope the advisory map to THIS chunk's PDBs, so each job entry records
        # only the PDBs that job actually contains — letting the run manifest
        # attribute an incomplete PDB to the specific job that failed (not blame
        # every submitted PDB whenever any one job fails).
        chunk_advisory = {
            g["pdb_id"]: detect_advisory_by_pdb.get(g["pdb_id"], [])
            for g in chunk
            if g.get("pdb_id")
        }
        try:
            _submit_batch_chunk(
                config,
                client,
                model_name=model_name,
                prompt_id=prompt_id,
                chunk_requests=chunk_requests,
                chunk_index=chunk_index,
                chunk_count=len(chunks),
                detect_advisory_by_pdb=chunk_advisory,
                created_at=now.isoformat(),
                uploaded_file_names=chunk_file_names,
            )
            submitted += 1
        except Exception as exc:
            # One chunk's failure must not lose the chunks already submitted:
            # each is registered the moment it is created and recovered
            # independently, and the remaining outstanding runs are simply
            # re-chunked on the next run (completed runs are skipped).
            logger.error(
                "Batch chunk %d/%d failed to submit: %s", chunk_index + 1, len(chunks), exc
            )
    logger.info("Submitted %d/%d batch chunk(s).", submitted, len(chunks))


def check_batch_status() -> None:
    """Poll the Gemini Batch API for all tracked jobs and download finished ones."""
    config = get_config()
    client = get_client()

    registry = _load_job_registry(config)

    # Migration: a workspace from before the registry has only the single-file
    # pointer. Adopt that in-flight job so it is tracked and recovered like any
    # other (model/prompt are back-filled from its sidecar at recover time).
    if not registry["jobs"] and config.current_batch_job_file.exists():
        legacy_name = config.current_batch_job_file.read_text().strip()
        if legacy_name:
            _register_job(
                config,
                {
                    "job_name": legacy_name,
                    "status": BATCH_STATUS_SUBMITTED,
                    "model_requested": None,
                    "prompt": None,
                    "code_version": None,
                    "chunk_index": 0,
                    "chunk_count": 1,
                    "request_count": None,
                    "created_at": None,
                    "raw_output_file": None,
                    "recovered_at": None,
                    "detect_advisory": {},
                },
            )
            registry = _load_job_registry(config)

    pending = [e for e in registry["jobs"].values() if e.get("status") == BATCH_STATUS_SUBMITTED]
    if not pending:
        logger.info("No active batch job found in state.")
        return

    # The SDK exposes terminal states as JOB_STATE_* on ``job.state.name``.
    # Some terminal states carry results to download; others do not.
    succeeded_states = ("JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED")
    failed_states = ("JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED")

    downloaded_any = False
    for entry in pending:
        job_name = entry["job_name"]
        try:
            job = client.batches.get(name=job_name)
        except Exception as e:
            logger.error("Failed to get batch job %s: %s", job_name, e)
            continue

        state = job.state.name if job.state else ""
        logger.info("Batch Job %s is in state: %s", job_name, state)

        if state in failed_states:
            expired_note = (
                " -- it ran or waited past the provider's 48-hour limit; resubmit or split the batch"
                if state == "JOB_STATE_EXPIRED"
                else ""
            )
            logger.error(
                "Batch job %s ended without results (%s)%s: %s",
                job_name,
                state,
                expired_note,
                job.error,
            )
            _update_job_status(config, job_name, status=BATCH_STATUS_FAILED)
            # A failed/expired job will never produce results — release its
            # uploaded inputs now (ref-counted so a shared file a live job still
            # needs is kept).
            _cleanup_terminal_job_uploads(config, client, job_name)
            continue

        if state not in succeeded_states:
            logger.info(
                "Batch job %s is not finished yet (state %s); try again later.", job_name, state
            )
            continue

        if not (job.dest and job.dest.file_name):
            logger.error("Batch job %s reported %s but exposed no result file.", job_name, state)
            continue

        try:
            os.makedirs(config.pipeline_runs_dir, exist_ok=True)
            raw_out_file = config.pipeline_runs_dir / f"raw_output_{_safe_job_name(job_name)}.jsonl"
            logger.info("Downloading %s to %s", job.dest.file_name, raw_out_file)
            content = client.files.download(file=job.dest.file_name)
            with open(raw_out_file, "wb") as f_out:
                f_out.write(content)
            _update_job_status(
                config, job_name, status=BATCH_STATUS_DOWNLOADED, raw_output_file=str(raw_out_file)
            )
            downloaded_any = True
        except Exception as e:  # surface any download failure without crashing
            logger.error("Failed to download batch results for %s: %s", job_name, e)

    # Recover whenever any downloaded-but-not-yet-recovered job exists -- not
    # only the ones downloaded this round -- so a crash between a download and
    # its recovery is healed on the next poll rather than stranding the result.
    registry = _load_job_registry(config)
    if downloaded_any or any(
        e.get("status") == BATCH_STATUS_DOWNLOADED for e in registry["jobs"].values()
    ):
        logger.info("Download(s) complete. Running recovery to parse results.")
        recover_batch()

    # After recovery, release the uploaded inputs of every now-recovered job whose
    # results are safely on disk. Ref-counted, so a PDF shared with a still-live
    # job is kept until that job is terminal too.
    if CLOUD_CLEANUP:
        registry = _load_job_registry(config)
        for entry in registry["jobs"].values():
            if entry.get("status") == BATCH_STATUS_RECOVERED:
                _cleanup_terminal_job_uploads(config, client, entry["job_name"])


def recover_batch() -> None:
    """Re-process raw JSONL batch output into individual per-run JSON files."""
    config = get_config()
    runs_dir = config.pipeline_runs_dir

    if not runs_dir.exists():
        logger.info("No pipeline runs directory found.")
        return

    registry = _load_job_registry(config)
    # Map a downloaded raw-output filename to its authoritative job entry, so
    # each result is attributed to the model/prompt of the job that produced
    # it -- not a shared sidecar a later submission may have overwritten.
    by_raw = {
        Path(e["raw_output_file"]).name: e
        for e in registry["jobs"].values()
        if e.get("raw_output_file")
    }

    def _load_provenance(raw_file: Path) -> dict:
        # Legacy/migration fallback for raw outputs with no registry entry
        # (downloaded before the registry existed): match the per-job sidecar
        # (raw_output_<job>.jsonl -> _batch_provenance_<job>.json), then the
        # legacy shared file. Without the per-job match, a stale raw file from
        # an earlier job would be stamped with a later job's model.
        job_suffix = raw_file.stem.removeprefix("raw_output_")
        for prov_file in (
            runs_dir / f"_batch_provenance_{job_suffix}.json",
            runs_dir / "_batch_provenance.json",
        ):
            if prov_file.exists():
                try:
                    loaded = json.loads(prov_file.read_text())
                except (json.JSONDecodeError, OSError):
                    return {}
                return loaded if isinstance(loaded, dict) else {}
        return {}

    for raw_file in runs_dir.glob("raw_output_*.jsonl"):
        entry = by_raw.get(raw_file.name)
        batch_meta: dict[str, Any]
        if entry is not None and entry.get("model_requested"):
            batch_meta = {
                "model_requested": entry.get("model_requested"),
                "prompt": entry.get("prompt"),
                "code_version": entry.get("code_version"),
                "detect_advisory": entry.get("detect_advisory") or {},
            }
        else:
            # No registry entry, OR a migration-adopted entry whose real model
            # lives only in the legacy per-job sidecar (the adopted entry has
            # model_requested=None). Resolve from the sidecar / shared file so
            # the result keeps its true model and per-model output directory.
            batch_meta = _load_provenance(raw_file)
        logger.info("Processing %s...", raw_file.name)
        with open(raw_file) as f:
            for line_no, line in enumerate(f, 1):
                try:
                    data = json.loads(line)
                    req_id = data.get("key") or data.get("id")
                    if not req_id or "__run_" not in req_id:
                        continue

                    # rpartition (not split("__")) so a PDB id that itself
                    # contains "__" doesn't unpack-error and drop the run.
                    pdb_id, _, run_part = req_id.rpartition("__run_")
                    run_num = int(run_part)

                    out_dir = (
                        config.ai_results_dir
                        / pdb_id
                        / model_run_subdir(batch_meta.get("model_requested"))
                    )
                    out_file = out_dir / f"run_{run_num}.json"
                    # Resume-by-existence: never clobber an already-recovered
                    # run. Makes recover idempotent and immune to a stale or
                    # re-downloaded raw file overwriting good results.
                    if out_file.exists():
                        continue

                    response_obj = data.get("response", {})
                    candidates = response_obj.get("candidates") or []
                    if not candidates:
                        logger.warning(
                            "[%s] Run %d: no candidates in batch response (line %d)",
                            pdb_id,
                            run_num,
                            line_no,
                        )
                        continue

                    content = candidates[0].get("content") or {}
                    parts = content.get("parts") or []
                    matched = False
                    for part in parts:
                        fc = part.get("functionCall")
                        if fc and fc.get("name") == ANNOTATOR_FUNCTION_NAME:
                            args = fc.get("args")
                            if args is None:
                                logger.warning(
                                    "[%s] Run %d: function call has no args (line %d)",
                                    pdb_id,
                                    run_num,
                                    line_no,
                                )
                                break
                            final_data = post_process_annotation(args)
                            final_data["_provenance"] = {
                                "model_requested": batch_meta.get("model_requested"),
                                "model_served": response_obj.get("modelVersion"),
                                "prompt": batch_meta.get("prompt"),
                                # From the submission record: the code that built
                                # and submitted the batch, not the recovery run.
                                "code_version": batch_meta.get("code_version"),
                                "detect_advisory": (batch_meta.get("detect_advisory") or {}).get(
                                    pdb_id, []
                                ),
                                "run": run_num,
                                "mode": "batch",
                                "timestamp": datetime.now(UTC).isoformat(),
                            }

                            os.makedirs(out_dir, exist_ok=True)
                            tmp_out = out_file.with_suffix(".tmp")
                            with open(tmp_out, "w") as f_out:
                                json.dump(final_data, f_out, indent=2)
                            os.replace(tmp_out, out_file)
                            matched = True
                            break

                    if not matched:
                        logger.warning(
                            "[%s] Run %d: no matching function call in response (line %d)",
                            pdb_id,
                            run_num,
                            line_no,
                        )
                except Exception as e:
                    logger.error(
                        "Row-level Error Isolation: Failed to process line %d in %s: %s",
                        line_no,
                        raw_file.name,
                        e,
                    )
                    continue

        if entry is not None:
            _update_job_status(
                config,
                entry["job_name"],
                status=BATCH_STATUS_RECOVERED,
                recovered_at=datetime.now(UTC).isoformat(),
            )


def discover_annotation_targets(num_runs: int, model_name: str) -> list[str]:
    """Enriched PDB IDs that still need annotation runs for *model_name*.

    A PDB counts as done when its per-model run directory already holds
    *num_runs* run files; those are excluded. Model-aware so it matches the
    namespaced output layout written by the runners.
    """
    config = get_config()
    enriched_pdbs = {p.stem.upper() for p in config.enriched_dir.glob("*.json")}
    done: set[str] = set()
    if config.ai_results_dir.exists():
        for d in config.ai_results_dir.iterdir():
            if not d.is_dir():
                continue
            model_dir = d / model_run_subdir(model_name)
            completed = sum(
                1 for n in range(1, num_runs + 1) if (model_dir / f"run_{n}.json").exists()
            )
            if completed >= num_runs:
                done.add(d.name.upper())
    return sorted(enriched_pdbs - done)


def run_annotation_stage(
    pdb_id: str | None = None,
    targets_file: str | None = None,
    prompt_file: str | None = None,
    model: str | None = None,
    num_runs: int = GEMINI_DEFAULT_RUNS,
    batch: bool = False,
    temperature: float | None = None,
    thinking_level: str | None = None,
) -> None:
    """Resolve targets / prompt / model and run annotation (single or batch).

    Shared by the ``annotate`` and ``pipeline`` commands. Auto-discovers
    enriched PDBs that still need runs when no explicit target is given.
    Raises ``FileNotFoundError`` when no prompt is available.
    """
    config = get_config()

    # Fail fast on a stale / missing storage contract BEFORE the expensive AI
    # calls. Previously only the interactive curate step validated the contract,
    # so a layout mismatch surfaced only after annotation had already run.
    from gpcr_tools.workspace import validate_contract

    validate_contract(config)

    model_name = model or get_gemini_model_name()

    if pdb_id:
        pdb_ids = [pdb_id.upper()]
    elif targets_file:
        from gpcr_tools.fetcher.targets import read_targets

        pdb_ids = read_targets(Path(targets_file))
    else:
        pdb_ids = discover_annotation_targets(num_runs, model_name)

    if prompt_file:
        prompt_text = Path(prompt_file).read_text(encoding="utf-8")
        prompt_id = Path(prompt_file).stem
    elif config.default_prompt_file.exists():
        prompt_text = config.default_prompt_file.read_text(encoding="utf-8")
        prompt_id = config.default_prompt_file.stem
    else:
        raise FileNotFoundError(
            f"Default prompt file not found at {config.default_prompt_file}; "
            "create it or pass a prompt file."
        )

    if batch:
        build_and_submit_batch(
            pdb_ids,
            prompt_text,
            num_runs=num_runs,
            model_name=model_name,
            prompt_id=prompt_id,
            temperature=temperature,
            thinking_level=thinking_level,
        )
        return

    download_log = _read_download_log_safe(config)
    for pid in pdb_ids:
        enriched_path = config.enriched_dir / f"{pid}.json"
        if not enriched_path.exists():
            logger.warning("Skipping %s: no enriched data at %s", pid, enriched_path)
            continue
        try:
            with open(enriched_path, encoding="utf-8") as fh:
                enriched_data = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Skipping %s: unreadable enriched JSON: %s", pid, exc)
            continue
        # Resolve the PDF via the canonical DOI-named file (with a per-PDB
        # fallback), passing the download log so single-mode can resolve a paper
        # whose DOI lives only in the log (a same-DOI sibling's virtual coverage).
        pdf_path = resolve_pdf_path(pid, download_log)
        if pdf_path is None:
            logger.warning("Skipping %s: no PDF found in %s", pid, config.papers_dir)
            continue
        run_single_pdb(
            pdb_id=pid,
            enriched_data=enriched_data,
            prompt_text=prompt_text,
            pdf_path=pdf_path,
            num_runs=num_runs,
            model_name=model_name,
            prompt_id=prompt_id,
            temperature=temperature,
            thinking_level=thinking_level,
        )
