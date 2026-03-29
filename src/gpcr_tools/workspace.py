"""Workspace initialization, contract validation, and startup checks.

Implements the v3.1 storage contract lifecycle:
  - ``init_workspace``  - create the full directory tree + contract file
  - ``validate_contract`` - verify contract file existence and version
  - ``validate_paths``  - verify absolute paths, no overlap / collision
  - ``print_path_summary`` - emit one-line-per-path summary to stderr
  - ``ensure_runtime_dirs`` - create missing operational dirs post-validation
  - ``startup_checks``  - orchestrate the full startup sequence
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from gpcr_tools.config import OVERRIDE_VARS, WorkspaceConfig, get_config

SUPPORTED_CONTRACT_VERSION = 1

# Directories created by ``init_workspace`` (relative to workspace root).
_INIT_DIRS: list[str] = [
    "contract",
    "raw",
    "raw/pdb_json",
    "raw/structure_files",
    "enriched",
    "papers",
    "ai_results",
    "aggregated",
    "aggregated/logs",
    "aggregated/validation_logs",
    "output",
    "output/csv",
    "output/audit",
    "cache",
    "state",
    "state/pipeline_runs",
    "tmp",
]


# ----------------------------------------------------------------------- #
#  Workspace initialisation                                                #
# ----------------------------------------------------------------------- #


def init_workspace(workspace_root: Path | None = None) -> None:
    """Create the full v3.1 workspace tree and contract file.

    Safe to run multiple times.  Refuses to overwrite a contract whose
    version differs from ``SUPPORTED_CONTRACT_VERSION``.
    """
    if workspace_root is None:
        workspace_root = get_config().workspace
    workspace_root = Path(workspace_root).resolve()

    contract_file = workspace_root / "contract" / "storage_contract.json"

    if contract_file.exists():
        _check_existing_contract(contract_file)

    for subdir in _INIT_DIRS:
        (workspace_root / subdir).mkdir(parents=True, exist_ok=True)

    if not contract_file.exists():
        contract_data = {
            "storage_contract_version": SUPPORTED_CONTRACT_VERSION,
            "created_by": "gpcr-tools",
            "created_at_utc": datetime.now(UTC).isoformat(),
        }
        with open(contract_file, "w", encoding="utf-8") as f:
            json.dump(contract_data, f, indent=2)
            f.write("\n")

    print(f"[workspace] initialized → {workspace_root}", file=sys.stderr)


def _check_existing_contract(contract_file: Path) -> None:
    """Abort if an existing contract file is incompatible."""
    try:
        with open(contract_file, encoding="utf-8") as f:
            existing = json.load(f)
    except json.JSONDecodeError:
        _abort(f"Existing contract file is not valid JSON: {contract_file}")

    version = existing.get("storage_contract_version")
    if version is None:
        _abort(f"Existing contract file missing 'storage_contract_version': {contract_file}")
    if version != SUPPORTED_CONTRACT_VERSION:
        _abort(
            f"Workspace contract version is {version}, "
            f"but this tool supports version {SUPPORTED_CONTRACT_VERSION}.\n"
            f"Cannot upgrade or downgrade automatically."
        )


# ----------------------------------------------------------------------- #
#  Contract validation                                                     #
# ----------------------------------------------------------------------- #


def validate_contract(config: WorkspaceConfig | None = None) -> None:
    """Ensure the contract file exists, parses, and has a supported version."""
    if config is None:
        config = get_config()

    cf = config.contract_file

    if not cf.exists():
        _abort(
            f"Workspace contract file not found: {cf}\n"
            "Please initialize the workspace first:\n\n"
            "    gpcr-tools init-workspace\n"
        )

    try:
        with open(cf, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        _abort(f"Contract file is not valid JSON: {cf}\nDetail: {exc}")

    version = data.get("storage_contract_version")
    if version is None:
        _abort(f"Contract file missing 'storage_contract_version': {cf}")

    if version != SUPPORTED_CONTRACT_VERSION:
        _abort(
            f"Unsupported contract version {version} "
            f"(this tool supports version {SUPPORTED_CONTRACT_VERSION}).\n"
            f"Contract file: {cf}"
        )


# ----------------------------------------------------------------------- #
#  Path validation                                                         #
# ----------------------------------------------------------------------- #


def validate_paths(config: WorkspaceConfig | None = None) -> None:
    """Validate resolved paths: absolute, no overlaps, no collisions.

    Rules enforced (see strategy §5):
      - all resolved paths are absolute
      - non-overridden paths must resolve under GPCR_WORKSPACE
      - no two top-level dirs may be the same path
      - no two top-level dirs may have a parent/child relationship
      - no top-level dir may equal the workspace root
    """
    if config is None:
        config = get_config()

    if not config.workspace.is_absolute():
        _abort(f"Workspace path is not absolute: {config.workspace}")

    named_dirs: dict[str, Path] = {
        "raw": config.raw_dir,
        "enriched": config.enriched_dir,
        "papers": config.papers_dir,
        "ai_results": config.ai_results_dir,
        "aggregated": config.aggregated_dir,
        "output": config.output_dir,
        "cache": config.cache_dir,
        "state": config.state_dir,
        "tmp": config.tmp_dir,
    }

    for name, path in named_dirs.items():
        if not path.is_absolute():
            _abort(f"{name} path is not absolute: {path}")

    for name, path in named_dirs.items():
        if path == config.workspace:
            _abort(f"{name} path must not equal the workspace root: {path}")

    # Pairwise collision / nesting check
    names = list(named_dirs.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a_name, a_path = names[i], named_dirs[names[i]]
            b_name, b_path = names[j], named_dirs[names[j]]
            if a_path == b_path:
                _abort(f"Path collision — {a_name} and {b_name} both resolve to: {a_path}")
            if a_path.is_relative_to(b_path) or b_path.is_relative_to(a_path):
                _abort(
                    f"Path overlap — {a_name} ({a_path}) and "
                    f"{b_name} ({b_path}) have a parent/child relationship."
                )

    # Non-overridden paths must be under workspace
    for name, path in named_dirs.items():
        env_var = OVERRIDE_VARS[name]
        if os.environ.get(env_var) is None and not path.is_relative_to(config.workspace):
            _abort(
                f"{name} path ({path}) is not under workspace "
                f"({config.workspace}) and was not explicitly overridden via {env_var}."
            )


# ----------------------------------------------------------------------- #
#  Path summary                                                            #
# ----------------------------------------------------------------------- #


def print_path_summary(config: WorkspaceConfig | None = None) -> None:
    """Print resolved workspace paths to stderr."""
    if config is None:
        config = get_config()

    lines = [
        ("root", config.workspace),
        ("raw", config.raw_dir),
        ("enriched", config.enriched_dir),
        ("papers", config.papers_dir),
        ("ai_results", config.ai_results_dir),
        ("aggregated", config.aggregated_dir),
        ("output", config.output_dir),
        ("cache", config.cache_dir),
        ("state", config.state_dir),
        ("tmp", config.tmp_dir),
        ("contract", config.contract_file),
    ]
    for label, path in lines:
        print(f"[workspace] {label:<14s} → {path}", file=sys.stderr)


# ----------------------------------------------------------------------- #
#  Runtime directory creation                                              #
# ----------------------------------------------------------------------- #


def ensure_runtime_dirs(config: WorkspaceConfig | None = None) -> None:
    """Create missing non-contract operational directories.

    Must only be called **after** contract validation succeeds.
    Does NOT create ``contract/`` or ``storage_contract.json``.
    """
    if config is None:
        config = get_config()

    dirs = [
        config.raw_dir,
        config.raw_dir / "pdb_json",
        config.raw_dir / "structure_files",
        config.enriched_dir,
        config.papers_dir,
        config.ai_results_dir,
        config.aggregated_dir,
        config.aggregated_dir / "logs",
        config.aggregated_dir / "validation_logs",
        config.output_dir,
        config.csv_output_dir,
        config.audit_output_dir,
        config.cache_dir,
        config.state_dir,
        config.pipeline_runs_dir,
        config.tmp_dir,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------- #
#  Full startup sequence                                                   #
# ----------------------------------------------------------------------- #


def startup_checks(config: WorkspaceConfig | None = None) -> WorkspaceConfig:
    """Execute the full startup validation sequence.

    1. Resolve configuration
    2. Validate absolute paths and overlap rules
    3. Validate contract file
    4. Print path summary to stderr
    5. Create any missing operational directories

    Returns the validated config for convenience.
    """
    if config is None:
        config = get_config()

    validate_paths(config)
    validate_contract(config)
    print_path_summary(config)
    ensure_runtime_dirs(config)

    return config


# ----------------------------------------------------------------------- #
#  Helpers                                                                 #
# ----------------------------------------------------------------------- #


def _abort(msg: str) -> None:  # pragma: no cover  (tested via sys.exit side-effect)
    """Print an error to stderr and exit."""
    print(f"Error: {msg}", file=sys.stderr)
    sys.exit(1)
