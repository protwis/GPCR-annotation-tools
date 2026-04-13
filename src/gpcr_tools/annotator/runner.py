"""Annotation runner -- single-PDB, batch submission, and recovery."""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from google.genai.errors import APIError

from gpcr_tools.annotator.gemini_client import get_client
from gpcr_tools.annotator.pdf_compressor import compress_pdf_if_needed
from gpcr_tools.annotator.post_processor import post_process_annotation
from gpcr_tools.annotator.prompt_builder import build_prompt_parts
from gpcr_tools.annotator.schema import ANNOTATION_TOOL, TOOL_CONFIG
from gpcr_tools.config import (
    ANNOTATOR_FUNCTION_NAME,
    GEMINI_BASE_BACKOFF,
    GEMINI_DEFAULT_RUNS,
    GEMINI_MAX_RETRIES,
    GEMINI_MAX_WORKERS,
    SLEEP_GEMINI_429,
    TIMEOUT_BATCH_RESULT_DOWNLOAD,
    get_config,
    get_gemini_model_name,
)

logger = logging.getLogger(__name__)


def run_single_pdb(
    pdb_id: str,
    enriched_data: dict,
    prompt_text: str,
    pdf_path: Path,
    num_runs: int = GEMINI_DEFAULT_RUNS,
    model_name: str | None = None,
) -> None:
    """Run annotation for a single PDB entry using parallel Gemini calls.

    Uploads the PDF once, then fans out *num_runs* independent generation
    requests via a thread pool.  Completed runs are persisted atomically
    so the process is safely resumable.
    """
    model_name = model_name or get_gemini_model_name()
    config = get_config()
    out_dir = config.ai_results_dir / pdb_id

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
            uploaded_file = client.files.upload(file=str(actual_pdf))
        except Exception as e:
            logger.error("[%s] Failed to upload PDF: %s", pdb_id, e)
            return

        try:
            parts = build_prompt_parts(pdb_id, enriched_data, prompt_text)
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
                            config=TOOL_CONFIG,
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
                executor.map(do_run, range(1, num_runs + 1))

        finally:
            import contextlib

            with contextlib.suppress(Exception):
                if uploaded_file.name:
                    client.files.delete(name=uploaded_file.name)


def build_and_submit_batch(
    targets: list[str],
    prompt_text: str,
    num_runs: int = GEMINI_DEFAULT_RUNS,
    model_name: str | None = None,
) -> None:
    """Build a JSONL payload for all *targets* and submit it to the Gemini Batch API."""
    model_name = model_name or get_gemini_model_name()
    config = get_config()
    client = get_client()

    # Prepare batch requests
    requests = []
    registry = {}

    # Check if uploaded files registry exists
    reg_file = config.uploaded_files_registry_file
    if reg_file.exists():
        try:
            with open(reg_file) as f:
                registry = json.load(f)
        except json.JSONDecodeError:
            pass

    for pdb_id in targets:
        enriched_file = config.enriched_dir / f"{pdb_id}.json"
        pdf_file = config.papers_dir / f"{pdb_id}.pdf"

        if not enriched_file.exists() or not pdf_file.exists():
            logger.warning("[%s] Missing enriched data or PDF, skipping batch prep.", pdb_id)
            continue

        with open(enriched_file) as f:
            enriched_data = json.load(f)

        # Determine runs to do
        out_dir = config.ai_results_dir / pdb_id
        os.makedirs(out_dir, exist_ok=True)
        runs_to_do = [n for n in range(1, num_runs + 1) if not (out_dir / f"run_{n}.json").exists()]

        if not runs_to_do:
            continue

        # Upload or get PDF
        pdf_uri = registry.get(pdb_id)
        if not pdf_uri:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_pdf = Path(tmp_dir) / f"{pdb_id}_compressed.pdf"
                try:
                    actual_pdf = compress_pdf_if_needed(pdf_file, tmp_pdf)
                    uploaded_file = client.files.upload(file=str(actual_pdf))
                    pdf_uri = uploaded_file.uri
                    registry[pdb_id] = pdf_uri
                    logger.info("[%s] Uploaded PDF to %s", pdb_id, pdf_uri)
                except Exception as e:
                    logger.error("[%s] Failed to upload PDF: %s", pdb_id, e)
                    continue

        parts = build_prompt_parts(pdb_id, enriched_data, prompt_text)

        # We need to construct the request dict for the batch API
        # The schema for the batch API contents is identical to generate_content
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

            # The tool schema must be provided as a dict
            assert ANNOTATION_TOOL.function_declarations is not None
            fn_decl = ANNOTATION_TOOL.function_declarations[0]
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

            requests.append(
                {
                    "id": req_id,
                    "request": {
                        "model": model_name,
                        "contents": contents_batch,
                        "tools": [tool_dict],
                        "toolConfig": {"functionCallingConfig": {"mode": "ANY"}},
                    },
                }
            )

    # Save updated registry
    tmp_reg = reg_file.with_suffix(".tmp")
    with open(tmp_reg, "w") as f:
        json.dump(registry, f, indent=2)
    os.replace(tmp_reg, reg_file)

    if not requests:
        logger.info("No batch requests to submit. All done!")
        return

    # Write JSONL
    os.makedirs(config.pipeline_runs_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".jsonl") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")
        tmp_jsonl = Path(f.name)

    try:
        # Upload JSONL to Gemini
        batch_src_file = client.files.upload(file=str(tmp_jsonl))
        if not batch_src_file.name:
            raise ValueError("Uploaded file has no name")
        logger.info("Uploaded batch JSONL source: %s", batch_src_file.name)

        # Submit batch
        batch_job = client.batches.create(model=model_name, src=batch_src_file.name)
        if not batch_job.name:
            raise ValueError("Created batch job has no name")
        logger.info("Batch submitted successfully! Job Name: %s", batch_job.name)

        # Save job name
        tmp_job_file = config.current_batch_job_file.with_suffix(".tmp")
        with open(tmp_job_file, "w") as f:
            f.write(batch_job.name)
        os.replace(tmp_job_file, config.current_batch_job_file)

    finally:
        if tmp_jsonl.exists():
            os.remove(tmp_jsonl)


def check_batch_status() -> None:
    """Poll the Gemini Batch API for the current job and download results when complete."""
    config = get_config()
    job_file = config.current_batch_job_file

    if not job_file.exists():
        logger.info("No active batch job found in state.")
        return

    with open(job_file) as f:
        job_name = f.read().strip()

    client = get_client()
    try:
        job = client.batches.get(name=job_name)
    except Exception as e:
        logger.error("Failed to get batch job %s: %s", job_name, e)
        return

    logger.info("Batch Job %s is in state: %s", job_name, job.state)

    if job.state in ("SUCCEEDED", "FAILED", "PARTIALLY_SUCCEEDED"):
        logger.info("Batch has completed. Downloading results...")
        if hasattr(job, "output_uri") and job.output_uri:
            import requests

            try:
                os.makedirs(config.pipeline_runs_dir, exist_ok=True)
                safe_name = job_name.replace("/", "_")
                raw_out_file = config.pipeline_runs_dir / f"raw_output_{safe_name}.jsonl"
                logger.info("Downloading from %s to %s", job.output_uri, raw_out_file)

                response = requests.get(job.output_uri, timeout=TIMEOUT_BATCH_RESULT_DOWNLOAD)
                response.raise_for_status()
                with open(raw_out_file, "wb") as f_out:
                    f_out.write(response.content)
                logger.info("Download complete. Running recovery to parse results.")
                recover_batch()
            except requests.exceptions.RequestException as e:
                logger.error("Failed to download results: %s", e)
        else:
            logger.info("No output URI found for this job.")


def recover_batch() -> None:
    """Re-process raw JSONL batch output into individual per-run JSON files."""
    config = get_config()
    runs_dir = config.pipeline_runs_dir

    if not runs_dir.exists():
        logger.info("No pipeline runs directory found.")
        return

    for raw_file in runs_dir.glob("raw_output_*.jsonl"):
        logger.info("Processing %s...", raw_file.name)
        with open(raw_file) as f:
            for line_no, line in enumerate(f, 1):
                try:
                    data = json.loads(line)
                    req_id = data.get("id")
                    if not req_id or "__run_" not in req_id:
                        continue

                    pdb_id, run_part = req_id.split("__")
                    run_num = int(run_part.replace("run_", ""))

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

                            out_dir = config.ai_results_dir / pdb_id
                            os.makedirs(out_dir, exist_ok=True)
                            out_file = out_dir / f"run_{run_num}.json"

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
