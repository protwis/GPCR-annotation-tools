# Development Standards & Philosophy

**Repository:** `GPCR-annotation-tools`
**Maintained since:** 2026
**Purpose:** Production-grade, human-in-the-loop curation suite for GPCR structural biology.

---

## Core Principles

### 1. One Epic at a Time

Never modify multiple modules or features simultaneously. Each Epic is an atomic unit of work with a clear scope, deliverable, and test criteria. Complete one before starting the next.

### 2. Proof-of-Concept Before Production

When introducing new logic (especially domain-specific validation, data transformation, or API integration), write a standalone PoC script first. Verify the concept works in isolation. Only integrate into the main codebase when there are no remaining unknowns.

### 3. Minimum Changes, Maximum Clarity

Every change should be the smallest diff that achieves the goal. No drive-by refactors. No "while I'm here" cleanups. If adjacent code needs improvement, file it as a separate task.

### 4. Test-Driven Verification

Every new function must ship with at least one test. Every bug fix must ship with a regression test that would have caught the bug. No exceptions.

---

## Code Quality Standards

### Python Version & Tooling

- **Minimum Python:** 3.11
- **Linter & Formatter:** `ruff` (enforced in CI)
  - Rules: `E, W, F, I, N, UP, B, SIM, RUF`
  - Line length: 100
- **Type Checker:** `mypy` with `check_untyped_defs = true`
- **Test Framework:** `pytest` with `pytest-cov`
- **CI Gate:** All of the above must pass before merge.

### Module Architecture

The codebase follows a strict separation of concerns:

```
src/gpcr_tools/
├── config.py              # Constants, schemas, workspace resolution. No I/O.
├── workspace.py           # Contract validation, directory lifecycle.
├── __main__.py            # CLI entrypoint only. No business logic.
└── csv_generator/
    ├── app.py             # Orchestration. Ties modules together.
    ├── data_loader.py     # File I/O: JSON loading, PDB discovery.
    ├── review_engine.py   # Interactive review logic. No CSV, no file writes.
    ├── csv_writer.py      # Pure data transformation. No UI, no prompts.
    ├── audit.py           # Audit trail logging.
    ├── validation_display.py  # Validation alert rendering.
    └── ui.py              # Rich console helpers, panels, theming.
```

**Rules:**
- `csv_writer.py` must remain a **pure function module** — no Rich imports, no `console.print`, no `Prompt.ask`.
- `review_engine.py` must never write files directly — it returns data; callers write it.
- `config.py` must never perform I/O — it resolves paths and defines constants.
- New utility functions go into the module where they are used. Do not create a catch-all `utils.py` unless the function is genuinely shared across 3+ modules.

### Function Design

- **Type hints are mandatory** on all function signatures (parameters and return types).
- **Docstrings** are required for public functions. Use imperative mood ("Return the…", "Check whether…"). Skip docstrings for obvious internal helpers.
- **No mutable default arguments.** Use `None` + internal initialization:
  ```python
  def foo(items: list[str] | None = None) -> ...:
      if items is None:
          items = []
  ```
- **Prefer returning data over mutating arguments.** If a function must mutate, document it explicitly.

### Naming Conventions

- **Functions/variables:** `snake_case`
- **Classes:** `PascalCase`
- **Constants:** `UPPER_SNAKE_CASE`
- **Private helpers:** prefix with `_` (e.g., `_build_structure_note`)
- **Test functions:** `test_<what>_<scenario>` (e.g., `test_coerce_type_float_roundtrip`)

### Error Handling

- **At system boundaries** (file I/O, user input, API calls): catch specific exceptions, log actionable messages.
- **Internal logic:** let exceptions propagate. Do not wrap internal calls in try/except "just in case."
- **Never silently swallow errors.** If a PDB fails to load, record it in `processed_log.json` as `"failed"` and continue to the next.

---

## Testing Standards

### Coverage Requirements

- Every **new function** needs at least one happy-path test.
- Every **bug fix** needs a regression test that reproduces the original bug.
- Every **interactive prompt change** (new choices, changed defaults) needs a monkeypatch test verifying the prompt flow.

### Test Organization

```
tests/
├── conftest.py               # Shared fixtures (sample_pdb_data, workspace setup)
├── fixtures/                  # Static JSON test data
│   ├── sample_pdb_simple.json
│   ├── sample_pdb_controversy.json
│   └── ...
├── unit/                      # One test file per source module
│   ├── test_config.py
│   ├── test_csv_writer.py
│   ├── test_review_engine.py
│   └── ...
└── integration/               # End-to-end pipeline tests
    ├── test_csv_pipeline.py
    ├── test_curate_cli.py
    └── test_init_workspace.py
```

### Test Conventions

- **Fixtures over setup methods.** Use `conftest.py` for shared fixtures.
- **No `importlib.reload()`.** Use `reset_config()` between tests instead.
- **Monkeypatch for environment variables.** Never modify `os.environ` directly.
- **Assert specific values, not just truthiness.** `assert result == expected`, not `assert result`.
- **Test file naming:** `test_{module_name}.py` mirrors `{module_name}.py`.

### What to Test vs. What Not to Test

**Do test:**
- Data transformation logic (CSV mapping, sanitization, type coercion)
- Controversy detection and resolution logic
- Validation alert filtering and impact analysis
- Config resolution with various environment variable combinations
- Audit trail entry format and content

**Do not test:**
- Rich panel rendering aesthetics (visual, not logical)
- Third-party library internals
- File system existence checks that are already covered by workspace contract validation

---

## Git & PR Standards

### Commit Messages

- Use imperative mood: "Add coerce_type helper" not "Added coerce_type helper"
- First line: concise summary (< 72 chars)
- Body (if needed): explain **why**, not **what** (the diff shows what)

### Branch Strategy

- `main` is the stable branch. All PRs target `main`.
- Feature branches: `feature/<epic-name>` or `fix/<bug-name>`
- One PR per Epic. Do not bundle unrelated changes.

### PR Checklist

Before opening a PR, verify:
- [ ] `ruff check src/ tests/` passes
- [ ] `ruff format --check src/ tests/` passes
- [ ] `mypy src/` passes
- [ ] `pytest tests/ -v` passes with no failures
- [ ] New functions have tests
- [ ] No hardcoded paths (use `get_config()`)
- [ ] No secrets or API keys in committed files

---

## Domain-Specific Conventions

### Data Flow Integrity

The curation pipeline has a strict data flow:

```
aggregated/*.json → load → review → transform → CSV
                                  ↘ audit trail (JSONL)
```

- **Never mutate the original loaded data.** Always `copy.deepcopy()` before passing to the review engine.
- **Audit every human decision.** Accept, edit, skip, delete — all must appear in `audit_trail.jsonl`.
- **CSV output is append-only.** Rows are appended to existing CSVs; never overwrite.

### Validation & Alert Integration

Validation warnings flow through `validation_data` dict. To add a new alert source:
1. Inject warnings into `validation_data["critical_warnings"]` (list of strings).
2. Use path prefixes matching top-level block keys (e.g., `"receptor_info"`, `"ligands"`).
3. The existing `get_relevant_validation_warnings()` will pick them up automatically.

### CSV Schema Changes

When adding or removing CSV columns:
1. Update `CSV_SCHEMA` in `config.py`.
2. Update `transform_for_csv()` in `csv_writer.py`.
3. Update the corresponding test in `test_csv_writer.py`.
4. Update `README.md` output table.
5. Consider backward compatibility — existing CSV files in user workspaces may lack new columns.

---

## Anti-Patterns to Avoid

| Don't | Do Instead |
|-------|------------|
| Monolithic files (> 500 lines) | Split by responsibility |
| Global mutable state | Frozen dataclass + LRU cache |
| Hardcoded paths | `get_config()` resolution |
| Silent exception swallowing | Log error + record failure state |
| `Prompt.ask()` storing raw strings | `coerce_type()` to preserve original types |
| `q` (quit) cascading to abort entire PDB | Provide `s` (skip) at every interactive layer |
| Backward-compat shims (`__getattr__`, re-exports) | Clean removal + deprecation warning if needed |
