"""Read-only operational reports over pipeline outputs.

Each function returns the report as a string so it is easy to test; the CLI
prints the returned text.  No mutation, no external calls.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gpcr_tools.code_version import get_code_version
from gpcr_tools.config import (
    CHIMERA_A5_ANCHOR_MIN_SCORE,
    GEMINI_DEFAULT_RUNS,
    RUN_MANIFEST_JSON_NAME,
    RUN_MANIFEST_MD_NAME,
    get_config,
    get_gemini_model_name,
    model_run_subdir,
)

logger = logging.getLogger(__name__)


def _read_json(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return None


def _read_json_dict(path: Path) -> dict[str, Any]:
    """Read a JSON object, returning {} if the file is missing, unreadable, or
    not a JSON object — a corrupt-but-parseable non-dict must not crash a report."""
    data = _read_json(path)
    return data if isinstance(data, dict) else {}


def _validation_log_files() -> list[Path]:
    vdir = get_config().aggregated_dir / "validation_logs"
    return sorted(vdir.glob("*_validation.json")) if vdir.is_dir() else []


def report_pdf_coverage() -> str:
    """Summarise paper-PDF coverage from the download log: how many PDB entries
    landed in each outcome (downloaded, paywalled, no DOI, ...)."""
    cfg = get_config()
    log = _read_json_dict(cfg.download_log_file)
    entries = [e for e in log.values() if isinstance(e, dict)]
    if not entries:
        return "PDF coverage: no download log found (run 'fetch-papers' first)."

    counts = Counter(e.get("status") or "unknown" for e in entries)
    total = sum(counts.values())
    lines = [f"PDF coverage report ({total} PDB entr{'y' if total == 1 else 'ies'}):", ""]
    for status, n in counts.most_common():
        pct = 100 * n / total
        lines.append(f"  {n:4d} ({pct:5.1f}%)  {status}")
    return "\n".join(lines)


def report_full_audit() -> str:
    """Summarise validation warnings and chimera conflicts across all aggregated
    PDBs."""
    files = _validation_log_files()
    if not files:
        return "Full audit: no validation logs found (run 'aggregate' first)."

    with_warnings: list[str] = []
    with_conflicts: list[str] = []
    chimera_status: Counter[str] = Counter()
    for f in files:
        pdb = f.name.removesuffix("_validation.json")
        data = _read_json_dict(f)
        chimera_status[data.get("chimera_status") or "unknown"] += 1
        if data.get("critical_warnings"):
            with_warnings.append(pdb)
        if data.get("algo_conflicts"):
            with_conflicts.append(pdb)

    lines = [f"Full validation audit ({len(files)} PDB(s)):", ""]
    lines.append(f"  PDBs with critical warnings: {len(with_warnings)}")
    if with_warnings:
        lines.append(f"    {', '.join(sorted(with_warnings))}")
    lines.append(f"  PDBs with algo conflicts:    {len(with_conflicts)}")
    if with_conflicts:
        lines.append(f"    {', '.join(sorted(with_conflicts))}")
    lines.append("  Chimera status:")
    for status, n in chimera_status.most_common():
        lines.append(f"    {n:4d}  {status}")
    return "\n".join(lines)


def report_tail_analysis() -> str:
    """Summarise the G-protein alpha5 identity analysis: the score
    distribution, status breakdown, and which structures to review.

    (The historical report also catalogued alpha5 sequences and candidate
    pools; those need per-run data not kept in the validation logs and are out
    of scope here.)"""
    files = _validation_log_files()
    if not files:
        return "alpha5 analysis: no validation logs found (run 'aggregate' first)."

    score_dist: Counter[Any] = Counter()
    status_dist: Counter[str] = Counter()
    flagged: list[tuple[str, Any]] = []
    for f in files:
        pdb = f.name.removesuffix("_validation.json")
        data = _read_json_dict(f)
        score = data.get("chimera_score")
        status = data.get("chimera_status") or "unknown"
        score_dist[score] += 1
        status_dist[status] += 1
        # A non-success status or a sub-anchor alpha5 score (the window did not
        # confidently match any reference) is worth a curator's eye.
        if status != "success" or (
            isinstance(score, (int, float)) and score < CHIMERA_A5_ANCHOR_MIN_SCORE
        ):
            flagged.append((pdb, score))

    lines = [
        f"G-protein alpha5 identity analysis ({len(files)} PDB(s)):",
        "",
        "  Score distribution:",
    ]
    for score in sorted((s for s in score_dist if isinstance(s, (int, float))), reverse=True):
        lines.append(f"    score {score}: {score_dist[score]}")
    if score_dist.get(None):
        lines.append(f"    score n/a: {score_dist[None]}")
    lines.append("  Status:")
    for status, n in status_dist.most_common():
        lines.append(f"    {n:4d}  {status}")
    lines.append(
        f"  Flagged for review (non-success or score < {CHIMERA_A5_ANCHOR_MIN_SCORE}): "
        f"{len(flagged)}"
    )
    for pdb, score in sorted(flagged):
        lines.append(f"    {pdb}: score={score}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Run manifest — full situational awareness for a curator
# ---------------------------------------------------------------------------

# A leading ``[TYPE]`` token on a validation warning / conflict string is the
# warning's category (e.g. ``[HALLUCINATION ALERT] at ...``). Bucketing on it
# turns a flat list of free-text warnings into a typed count breakdown without
# coupling the report to every individual alert-prefix constant.
_WARNING_TYPE_RE = re.compile(r"^\s*\[([^\]]+)\]")
_UNTYPED_WARNING_LABEL = "(untyped)"


def _read_json_dict_at(path: Path) -> dict[str, Any]:
    """Read a JSON object at *path*, returning {} if missing/unreadable/non-dict.

    A standalone helper (not :func:`_read_json_dict`) so a manifest run mid-pipeline,
    where any directory or file may be absent, never raises.
    """
    if not path.is_file():
        return {}
    return _read_json_dict(path)


def _warning_type(message: Any) -> str:
    """Category of a validation warning / conflict from its leading ``[TYPE]``.

    Falls back to ``(untyped)`` for a warning with no bracketed prefix, so every
    warning is counted in the breakdown rather than silently dropped.
    """
    match = _WARNING_TYPE_RE.match(str(message))
    return match.group(1).strip() if match else _UNTYPED_WARNING_LABEL


def _manifest_targets(cfg: Any) -> dict[str, Any]:
    """The full target list from ``targets.txt`` (count + ids); robust to absence."""
    from gpcr_tools.fetcher.targets import read_targets

    ids = read_targets(cfg.targets_file) if cfg.targets_file.is_file() else []
    return {"count": len(ids), "pdb_ids": ids}


def _manifest_no_pdf(cfg: Any) -> dict[str, Any]:
    """PDBs with no paper PDF, grouped by the not-run reason.

    For a PDB present in the download log the reason is its log status. A target
    PDB absent from the log entirely (never fetched) is reported under a dedicated
    ``not_in_download_log`` bucket, so the manifest accounts for every target.
    """
    from gpcr_tools.fetcher.targets import read_targets
    from gpcr_tools.papers.storage import resolve_pdf_path

    log = _read_json_dict_at(cfg.download_log_file)
    logged = {str(k).upper() for k in log}
    by_reason: dict[str, list[str]] = {}
    for pdb_id, entry in log.items():
        if not isinstance(entry, dict):
            continue
        # Resolve the PDF the same way every reader does (canonical DOI-named
        # file or legacy per-PDB), so a PDB covered by a same-DOI sibling's
        # download is not falsely reported as missing.
        if resolve_pdf_path(pdb_id, log) is not None:
            continue
        reason = entry.get("status") or "unknown"
        by_reason.setdefault(reason, []).append(pdb_id.upper())
    # Targets that never made it into the download log at all (e.g. fetch not yet
    # run for them): classify under ``not_in_download_log`` rather than dropping.
    targets = read_targets(cfg.targets_file) if cfg.targets_file.is_file() else []
    for pid in targets:
        if pid.upper() in logged or resolve_pdf_path(pid, log) is not None:
            continue
        by_reason.setdefault("not_in_download_log", []).append(pid.upper())
    return {
        "count": sum(len(v) for v in by_reason.values()),
        "by_reason": {reason: sorted(set(ids)) for reason, ids in sorted(by_reason.items())},
    }


def _manifest_run_counts(cfg: Any, model_name: str, num_runs: int) -> dict[str, Any]:
    """Per-PDB AI run counts; flags PDBs that ran but produced < *num_runs* runs.

    The incomplete reason is derived from what is recorded on disk: a batch job
    state (EXPIRED / FAILED) covering the PDB, else the aggregate-log failure, else
    ``unknown`` (a per-run bust reason such as MAX_TOKENS is not persisted in the
    saved run JSON, so it cannot be attributed here).
    """
    ai_dir = cfg.ai_results_dir
    failed_pdbs = _failed_job_pdbs(cfg)
    agg_log = _read_json_dict_at(cfg.state_dir / "aggregate_log.json")

    per_pdb: dict[str, int] = {}
    if ai_dir.is_dir():
        for d in sorted(p for p in ai_dir.iterdir() if p.is_dir()):
            model_dir = d / model_run_subdir(model_name)
            count = len(list(model_dir.glob("run_*.json"))) if model_dir.is_dir() else 0
            if count == 0:
                # Fall back to the legacy flat layout (pre-model-namespacing).
                count = len(list(d.glob("run_*.json")))
            per_pdb[d.name.upper()] = count

    incomplete: list[dict[str, Any]] = []
    for pdb_id, count in sorted(per_pdb.items()):
        if 0 < count < num_runs:
            agg = agg_log.get(pdb_id)
            reason = "unknown"
            if pdb_id in failed_pdbs:
                reason = "batch_job_failed_or_expired"
            elif isinstance(agg, dict) and agg.get("status") == "failed":
                reason = "aggregate_failed"
            incomplete.append({"pdb_id": pdb_id, "runs": count, "reason": reason})
    return {
        "expected_runs": num_runs,
        "model": model_name,
        "per_pdb": per_pdb,
        "incomplete_count": len(incomplete),
        "incomplete": incomplete,
    }


def _failed_job_pdbs(cfg: Any) -> set[str]:
    """The set of PDB ids that were part of a FAILED batch job.

    Each job entry records its PDBs under ``detect_advisory`` (keyed by every PDB
    submitted in that job), so a manifest can attribute an incomplete PDB to the
    specific job that died — rather than blaming every incomplete PDB whenever any
    job failed. Tolerant of an absent registry; a PDB not found here degrades to an
    ``unknown`` reason.
    """
    reg = _read_json_dict_at(cfg.batch_jobs_registry_file)
    jobs = reg.get("jobs")
    if not isinstance(jobs, dict):
        return set()
    pdbs: set[str] = set()
    for entry in jobs.values():
        if isinstance(entry, dict) and entry.get("status") == "failed":
            advisory = entry.get("detect_advisory")
            if isinstance(advisory, dict):
                pdbs.update(str(p).upper() for p in advisory)
    return pdbs


def _manifest_quality(cfg: Any) -> dict[str, Any]:
    """Validation-quality stats: one-click-acceptable vs gated, with a typed
    breakdown of warning / conflict categories and their counts."""
    files = _validation_log_files()
    acceptable: list[str] = []
    gated: list[str] = []
    warning_types: Counter[str] = Counter()
    conflict_types: Counter[str] = Counter()
    for f in files:
        pdb = f.name.removesuffix("_validation.json")
        data = _read_json_dict(f)
        warnings = [w for w in (data.get("critical_warnings") or []) if w]
        conflicts = [c for c in (data.get("algo_conflicts") or []) if c]
        for w in warnings:
            warning_types[_warning_type(w)] += 1
        for c in conflicts:
            conflict_types[_warning_type(c)] += 1
        # One-click-acceptable = nothing gates the curator: no critical warning and
        # no algo conflict. Anything else is gated (needs a manual look).
        if warnings or conflicts:
            gated.append(pdb)
        else:
            acceptable.append(pdb)
    return {
        "validated_count": len(files),
        "acceptable_count": len(acceptable),
        "acceptable": sorted(acceptable),
        "gated_count": len(gated),
        "gated": sorted(gated),
        "warning_types": dict(warning_types.most_common()),
        "conflict_types": dict(conflict_types.most_common()),
    }


def _manifest_provenance(cfg: Any, model_name: str, num_runs: int) -> dict[str, Any]:
    """Commit/version, model, prompt version, run timestamp, and N runs.

    The prompt version is the default prompt file's stem when present (the same
    value the runners stamp into ``_provenance``); ``None`` when no prompt is set.
    """
    prompt_id = cfg.default_prompt_file.stem if cfg.default_prompt_file.is_file() else None
    return {
        "code_version": get_code_version(),
        "model": model_name,
        "prompt": prompt_id,
        "expected_runs": num_runs,
        "generated_at": datetime.now(UTC).isoformat(),
    }


def build_run_manifest(
    model_name: str | None = None, num_runs: int | None = None
) -> dict[str, Any]:
    """Assemble the full run-manifest record as a JSON-serialisable dict.

    Robust to a partial / mid-pipeline workspace: every section degrades to empty
    counts rather than raising when a directory or file is absent.
    """
    cfg = get_config()
    model_name = model_name or get_gemini_model_name()
    num_runs = num_runs if num_runs is not None else GEMINI_DEFAULT_RUNS
    return {
        "provenance": _manifest_provenance(cfg, model_name, num_runs),
        "targets": _manifest_targets(cfg),
        "no_pdf": _manifest_no_pdf(cfg),
        "run_counts": _manifest_run_counts(cfg, model_name, num_runs),
        "quality": _manifest_quality(cfg),
    }


def _render_id_list(ids: list[str], *, limit: int = 50) -> str:
    """Render a PDB-id list for the human report, truncating very long lists so a
    corpus-scale manifest stays readable (the full list is always in the JSON)."""
    if not ids:
        return "(none)"
    if len(ids) <= limit:
        return ", ".join(ids)
    shown = ", ".join(ids[:limit])
    return f"{shown}, … (+{len(ids) - limit} more; see run_manifest.json)"


def render_run_manifest_md(manifest: dict[str, Any]) -> str:
    """Render the manifest dict as a human-readable Markdown report."""
    prov = manifest["provenance"]
    targets = manifest["targets"]
    no_pdf = manifest["no_pdf"]
    run_counts = manifest["run_counts"]
    quality = manifest["quality"]

    lines: list[str] = [
        "# GPCR annotation run manifest",
        "",
        f"- Generated: {prov['generated_at']}",
        f"- Code version: {prov['code_version']}",
        f"- Model: {prov['model']}",
        f"- Prompt: {prov['prompt']}",
        f"- Runs per PDB (expected): {prov['expected_runs']}",
        "",
        "## Targets",
        "",
        f"{targets['count']} target PDB(s).",
        "",
        f"{_render_id_list(targets['pdb_ids'])}",
        "",
        "## No PDF / not run",
        "",
        f"{no_pdf['count']} PDB(s) with no paper PDF, grouped by reason:",
        "",
    ]
    if no_pdf["by_reason"]:
        for reason, ids in no_pdf["by_reason"].items():
            lines.append(f"- **{reason}** ({len(ids)}): {_render_id_list(ids)}")
    else:
        lines.append("- (none)")
    lines += [
        "",
        "## Ran but incomplete",
        "",
        f"{run_counts['incomplete_count']} PDB(s) with fewer than "
        f"{run_counts['expected_runs']} runs (model {run_counts['model']}):",
        "",
    ]
    if run_counts["incomplete"]:
        lines.append("| PDB | runs | reason |")
        lines.append("|-----|------|--------|")
        for item in run_counts["incomplete"]:
            lines.append(f"| {item['pdb_id']} | {item['runs']} | {item['reason']} |")
    else:
        lines.append("- (none)")
    lines += [
        "",
        "## Quality",
        "",
        f"{quality['validated_count']} PDB(s) validated.",
        f"- One-click-acceptable (no critical warnings/conflicts): {quality['acceptable_count']}",
        f"- Gated (need review): {quality['gated_count']}",
        "",
        "### Critical-warning types",
        "",
    ]
    if quality["warning_types"]:
        lines.append("| Type | Count |")
        lines.append("|------|-------|")
        for wtype, n in quality["warning_types"].items():
            lines.append(f"| {wtype} | {n} |")
    else:
        lines.append("- (none)")
    lines += ["", "### Algo-conflict types", ""]
    if quality["conflict_types"]:
        lines.append("| Type | Count |")
        lines.append("|------|-------|")
        for ctype, n in quality["conflict_types"].items():
            lines.append(f"| {ctype} | {n} |")
    else:
        lines.append("- (none)")
    lines.append("")
    return "\n".join(lines)


def _atomic_write(path: Path, text: str) -> None:
    """Write *text* to *path* atomically (tmp + os.replace), creating parents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=str(path.parent), suffix=".tmp", delete=False, encoding="utf-8"
    ) as fd:
        tmp_path = fd.name
        fd.write(text)
    os.replace(tmp_path, str(path))


def write_run_manifest(
    model_name: str | None = None, num_runs: int | None = None
) -> tuple[Path, Path]:
    """Build the manifest and write both ``run_manifest.json`` and
    ``run_manifest.md`` to the workspace output dir; return their paths."""
    cfg = get_config()
    manifest = build_run_manifest(model_name=model_name, num_runs=num_runs)
    json_path = cfg.output_dir / RUN_MANIFEST_JSON_NAME
    md_path = cfg.output_dir / RUN_MANIFEST_MD_NAME
    _atomic_write(json_path, json.dumps(manifest, indent=2))
    _atomic_write(md_path, render_run_manifest_md(manifest))
    return json_path, md_path


def report_run_manifest() -> str:
    """Write the run manifest to disk and return a short human summary for the CLI."""
    json_path, md_path = write_run_manifest()
    manifest = _read_json_dict_at(json_path)
    targets = manifest.get("targets", {})
    no_pdf = manifest.get("no_pdf", {})
    run_counts = manifest.get("run_counts", {})
    quality = manifest.get("quality", {})
    return "\n".join(
        [
            "Run manifest written:",
            f"  {json_path}",
            f"  {md_path}",
            "",
            f"  Targets: {targets.get('count', 0)}",
            f"  No PDF / not run: {no_pdf.get('count', 0)}",
            f"  Ran but incomplete: {run_counts.get('incomplete_count', 0)}",
            f"  Validated: {quality.get('validated_count', 0)} "
            f"(acceptable {quality.get('acceptable_count', 0)}, "
            f"gated {quality.get('gated_count', 0)})",
        ]
    )
