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
    ├── csv_writer.py      # Data transformation + CSV file appending. No UI, no prompts.
    ├── audit.py           # Audit trail logging.
    ├── validation_display.py  # Validation alert rendering.
    └── ui.py              # Rich console helpers, panels, theming.
```

**Rules:**
- `csv_writer.py` must remain **UI-free** — no Rich imports, no `console.print`, no `Prompt.ask`. It owns both data transformation and CSV file appending.
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
- **Exception handling symmetry:** when a custom exception is introduced (e.g. `CsvSchemaMismatchError`), it must be handled at **every** call site that can raise it — not just the one you're currently working on. Before merging, grep the codebase for all callers.

### Defensive Coding Idioms

This section codifies the most common low-level bug patterns in this codebase. They arise from processing AI-generated JSON and external API data where keys may be absent, `null`, or inconsistently formatted.

#### 1. None-Safety — `.get()` Does NOT Guard Against Explicit `None`

`dict.get(key, default)` returns the default **only when the key is missing**. If the key exists with value `None`, it returns `None`:

```python
# BAD — returns None when key is present but null
chain = data.get("chain_id", "?")          # {"chain_id": None} → None
info  = data.get("chain_id_override", {})  # {"chain_id_override": None} → None

# GOOD — handles both absent and null
chain = data.get("chain_id") or "?"
info  = data.get("chain_id_override") or {}
```

**Rule:** when processing external data (JSON from AI, API responses, user-edited fixtures), always use `x.get(key) or fallback` for strings, dicts, and lists. Use `x.get(key, default)` only when `None` is a semantically impossible value (e.g. internally constructed dicts with guaranteed schemas).

**Nested chains** are especially dangerous:

```python
# BAD — crashes if "override" key exists but is None
data.get("override", {}).get("applied")

# GOOD
(data.get("override") or {}).get("applied")
```

#### 2. Cross-Module String Constants — No Magic String Duplication

If a string is **produced** in one module and **consumed** (matched against) in another, it must be defined as a **shared constant** in `config.py`. Hardcoding the same magic string in two different files is a guaranteed consistency bug.

```python
# BAD — "ghost_ligand" in config.py, "ghost ligand" in validation_display.py
VALIDATION_FATAL_KEYWORDS = ("ghost_ligand", ...)   # config.py
if "ghost ligand" in text:                           # validation_display.py → NEVER MATCHES

# GOOD — single source of truth
# config.py
ALERT_TYPE_HALLUCINATION = "HALLUCINATION"
# producer (validation_display.py)
warnings.append(f"[{ALERT_TYPE_HALLUCINATION}] {msg}")
# consumer (validation_display.py)
is_hallucination = ALERT_TYPE_HALLUCINATION in warn_str
```

**Rule:** before hardcoding any identifier string (alert types, status codes, keyword tokens), search the codebase. If it already appears elsewhere, extract it to `config.py`. If you are introducing a new identifier that will be matched elsewhere, define the constant first.

#### 3. Python Type-Dispatch Traps

##### `bool` is a subclass of `int`

`isinstance(True, int)` evaluates to `True`. In type-dispatch chains, the `bool` branch must be checked **before** `int`, and must use `elif` (not bare `if`) to prevent fall-through:

```python
# BAD — True falls through bool branch into int branch
if isinstance(original, bool):
    ...  # no match → falls through
if isinstance(original, int):
    return int(new_str)  # converts a bool original to int

# GOOD — elif stops fall-through; unrecognized bool input returns string
if isinstance(original, bool):
    ...
    return new_str  # explicit fallback
elif isinstance(original, int):
    return int(new_str)
```

##### `json.loads()` erases container type

`json.loads('{"a":1}')` returns a `dict` regardless of whether the original was a `list`. Always verify the parsed type matches the original:

```python
# BAD — a list original can silently become a dict
if isinstance(original, (list, dict)):
    return json.loads(new_str)

# GOOD — verify type after parsing
if isinstance(original, (list, dict)):
    parsed = json.loads(new_str)
    if isinstance(parsed, type(original)):
        return parsed
```

#### 4. Truthiness vs. Identity — `if x:` vs. `if x is not None:`

When a variable holds a dict, list, or other container, `if x:` is `False` for both `None` and empty containers (`{}`, `[]`). If the distinction matters (e.g. "user quit" vs. "user accepted everything, resulting in `{}`"), use `is not None`:

```python
# BAD — empty dict {} from a valid review is silently skipped
if final_data:
    write_to_csv(final_data)

# GOOD — only skip on explicit None (user quit)
if final_data is not None:
    write_to_csv(final_data)
```

#### 5. Immutable Module-Level Constants

Module-level constants must use **immutable types** to prevent accidental cross-test contamination and runtime mutation:

| Mutable | Immutable Replacement |
|---------|----------------------|
| `set({...})` | `frozenset({...})` |
| `[...]` | `(...)` (tuple) |
| `dict` (for dispatch tables) | Acceptable as-is if not mutated; consider `MappingProxyType` for critical schemas |

#### 6. Multi-File Write Atomicity

When a single operation writes to multiple files (e.g. `append_to_csvs` iterating over CSV files), validate **all** preconditions (schema checks, permissions) **before** writing to any file. Otherwise a failure midway leaves orphaned partial writes:

```python
# BAD — structures.csv written, then ligands.csv fails schema check → partial state
for filename, rows in data.items():
    check_schema(filename)  # may raise
    write(filename, rows)

# GOOD — pre-flight all checks, then write
for filename, rows in data.items():
    check_schema(filename)  # raises before any write happens
for filename, rows in data.items():
    write(filename, rows)
```

---

## Testing Standards

### Coverage Requirements

- Every **new function** needs at least one happy-path test.
- Every **bug fix** needs a regression test that reproduces the original bug.
- Every **interactive prompt change** (new choices, changed defaults) needs a monkeypatch test verifying the prompt flow.

### Test Organization

```
tests/
├── conftest.py               # Shared fixtures (sample_pdb_data, workspace setup, real_pdb_workspace)
├── fixtures/
│   ├── sample_*.json         # Synthetic minimal fixtures for unit tests
│   └── real_pdbs/            # Real PDB data (see §Real Data Testing Strategy)
│       ├── {pdb_id}.json
│       ├── logs/
│       └── validation_logs/
├── unit/                      # One test file per source module
│   ├── test_config.py
│   ├── test_csv_writer.py
│   ├── test_review_engine.py
│   └── ...
└── integration/               # End-to-end pipeline tests + real PDB tests
    ├── test_csv_pipeline.py
    ├── test_curate_cli.py
    ├── test_init_workspace_cli.py
    ├── test_real_pdb_fixtures.py
    ├── test_real_pdb_pipeline.py
    ├── test_real_pdb_gating.py
    └── test_real_pdb_review_engine.py
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

### Real Data Testing Strategy

The test suite maintains a set of **real PDB entries** (`tests/fixtures/real_pdbs/`) that exercise the full spectrum of domain scenarios: clean data, controversies, critical warnings (ghost chains, hallucinated ligands, UniProt clashes), oligomer analysis, and complex multi-model disagreement.

#### The Live Chain Goal

The ultimate target is a **live-generated test chain** where tests execute actual pipeline stages rather than reading static fixture files. The only permanent static fixtures are:

1. **A single PDB ID list** — the canonical set of test PDBs (currently 9 entries).
2. **AI annotation run outputs** (`ai_results/`) — because AI annotation is non-deterministic, slow, and requires external API keys, it is the **one permanent断点 (breakpoint)** in the chain.

**Mandatory Data Density:** To ensure robust majority voting and discrepancy detection tests, every PDB in the canonical set MUST have a complete set of **10 AI annotation runs** (`run_01.json` through `run_10.json`) committed to `tests/fixtures/real_pdbs/ai_results/{PDB_ID}/`.

Everything downstream of that breakpoint — `aggregate → validate → curate` — runs live during tests. Everything upstream of annotation — `download → enrich → papers` — also runs live once migrated. The AI annotation outputs are the only static fixtures that persist in the final state.

```
download → enrich → papers → [AI annotation] → aggregate → validate → curate
  live       live     live     STATIC FIXTURE      live        live      live
```

#### Why Live Over Static

Static fixtures carry a silent risk: **format drift**. If `aggregate` changes its output schema but the static `aggregated/*.json` fixtures are not updated, the downstream `curate` tests still pass — against stale data. A live chain eliminates this class of bug entirely, because each stage's tests consume the actual output of the previous stage.

#### Migration: Progressive Fixture Retirement

The pipeline is migrated **back-to-front** (curate first, then validate, then aggregate, and so on). During migration, the fixture strategy follows a **progressive retirement** model:

1. **Before migrating a stage:** its output is a static fixture. Downstream tests read from that fixture.
2. **After migrating a stage:** its output is live-generated during tests. Downstream tests switch to consuming the live output. The old static fixture is retired.

**Example — migrating `aggregate + validate`:**

| Phase | `aggregate` input | `aggregate` output | `curate` reads from |
|-------|-------------------|--------------------|---------------------|
| Before migration | N/A | Static `aggregated/*.json` | Static fixture |
| After migration | Static `ai_results/` + `enriched/` fixtures | Live-generated | Live aggregate output |

When switching a downstream test from static to live input, run a **one-time equivalence check**: confirm that the live-generated output is semantically consistent with the old static fixture (or that differences are expected and documented). This catches regressions introduced during the migration itself.

#### Intermediate Fixture Rules

During the migration process:

- **Adding intermediate fixtures is expected.** When migrating `aggregate`, you will commit `ai_results/` and `enriched/` fixtures as the new static input. These are the "new断点" until those upstream stages are also migrated.
- **Downstream tests must be updated in the same PR.** When `aggregate` starts producing live output, the existing `curate` tests that previously read static `aggregated/*.json` must be rewritten to consume the live output. Do not leave two parallel fixture paths.
- **One断点 at a time moves forward.** Each migration PR should clearly shift the static/live boundary by exactly one stage. The PR description must state which fixtures are being retired and which are being introduced.

#### Fixture Selection Criteria

The canonical PDB set must collectively cover:

- At least one **clean entry** (no controversies, no critical warnings)
- At least one entry with **voting log controversies**
- At least one entry with **critical validation warnings** (ghost chains, hallucinated ligands, UniProt clashes)
- At least one entry with **oligomer analysis** data
- At least one entry that triggers **fix-mode auto-resolution** (trivial controversies only)
- At least one **complex entry** with multiple overlapping issues

When a new pipeline stage is migrated, verify that the existing PDB set provides sufficient coverage for that stage's logic. If not, add new PDB IDs — but prefer expanding coverage of existing IDs first.

#### Test Organization for Real Data

```
tests/
├── fixtures/
│   ├── real_pdbs/                    # Current static断点 fixtures
│   │   ├── {pdb_id}.json            # Main aggregated data (retiring as aggregate goes live)
│   │   ├── logs/                    # Voting logs (retiring as aggregate goes live)
│   │   ├── validation_logs/         # Validation logs (retiring as validate goes live)
│   │   ├── ai_results/              # AI annotation outputs (permanent static fixtures)
│   │   └── enriched/                # Enriched metadata (retiring as enrich goes live)
│   └── sample_*.json                # Synthetic minimal fixtures for unit tests
└── integration/
    ├── test_real_pdb_fixtures.py     # Fixture integrity (adapts as fixtures evolve)
    ├── test_real_pdb_pipeline.py     # End-to-end smoke tests (progressively goes live)
    ├── test_real_pdb_gating.py       # Mode availability per PDB
    └── test_real_pdb_review_engine.py # Interactive flow with monkeypatched prompts
```

Unit tests continue to use **synthetic fixtures** (`sample_*.json`) for isolation and speed. Real data tests live exclusively in `integration/`.

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
- [ ] **None-safety:** any `.get(key, default)` on external data uses `or` pattern (see §Defensive Coding Idioms §1)
- [ ] **String constants:** no new magic strings duplicated across files (see §Defensive Coding Idioms §2)
- [ ] **Exception symmetry:** any new custom exception is handled at all call sites (grep for the raising function)
- [ ] **Truthiness:** `if x:` vs `if x is not None:` used correctly for dicts/lists (see §Defensive Coding Idioms §4)

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
4. **Any new alert type string** (e.g. `"HALLUCINATION"`, `"GHOST_LIGAND"`) must be defined as a constant in `config.py` and referenced by both the producer and consumer — never hardcoded in two places independently.

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
| `.get(key, {})` on external JSON (None leaks through) | `.get(key) or {}` (see §Defensive Coding Idioms) |
| Same magic string hardcoded in 2+ files | Shared constant in `config.py` |
| `if isinstance(x, bool): ... if isinstance(x, int):` (bool⊂int fall-through) | `elif` chain with explicit fallback in bool branch |
| `if data:` when `{}` is a valid state | `if data is not None:` |
| Mutable module-level `set` / `list` constants | `frozenset` / `tuple` |
| Handle custom exception in one call site only | Handle at **all** call sites (grep before merging) |
| `re.findall(r"\[\d+\]", full_path)` extracting nested indices | Anchor to block prefix, extract only top-level index |
