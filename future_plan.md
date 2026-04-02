# Future Development Plan

## 1. Type Enforcement in Interactive Review

**Issue:** In `review_engine.py`'s `coerce_type` function, the system attempts to coerce user input back to the original data type. However, if the coercion fails (e.g., trying to parse "apple" when the original value was a boolean `True`), it silently falls back to returning the string.

**Risk:** While this matches the legacy behavior ("Fallback -> keeps as string"), it can lead to type degradation and data pollution in the output CSVs (e.g., a boolean column suddenly containing string values due to a typo).

**Action Plan:**
Refactor the prompt logic at the UI layer. When a user is editing a field and the input cannot be robustly coerced to the original type (especially for strict types like booleans or numbers), the UI should reject the input, display a validation error message ("Expected a boolean/number"), and ask the user to retry, rather than silently accepting a string fallback.

---

## 2. Validation Ordering: Receptor Validation vs Oligomer Chain Override

**Issue:** In the old `pdb-annotation/aggregate_results.py`, the execution order is:

```
validate_and_enrich_ligands()   # Step 5
validate_receptor_identity()    # Step 6 — validates receptor chain_id + uniprot
analyze_oligomer()              # Step 7 — may override chain_id via _apply_chain_override()
```

`analyze_oligomer()` contains `_apply_chain_override()`, which auto-corrects the receptor's `chain_id` and `uniprot_entry_name` when the AI hallucinated a non-GPCR chain or when a 7TM upgrade is warranted. However, `validate_receptor_identity()` runs **before** this override, meaning it validates the **pre-correction** chain — which may be the hallucinated one.

**Practical impact:** In the hallucination case, `validate_receptor_identity()` will correctly report `UNIPROT_CLASH` (the hallucinated chain's UniProt won't match the GPCR roster), so the problem is flagged — but for the wrong reason and against the wrong chain. The validation report is technically misleading: it says "chain X has a UniProt clash" when the real issue is "chain X shouldn't have been selected at all."

**Why not fix during migration:** This is the production behavior of the old codebase. Changing the execution order during migration would be a functional change, not a port, and could introduce subtle interactions we haven't tested for. The migration plan explicitly preserves the old ordering to maintain behavioral equivalence.

**Action plan:**
After the aggregate & validate migration is complete and verified against the old output (Epic 8 equivalence check), introduce a dedicated follow-up epic:

1. Move `analyze_oligomer()` (specifically the chain override step) to execute **before** `validate_receptor_identity()`.
2. After the override, re-run receptor validation against the corrected chain.
3. Add regression tests for the specific scenario: AI hallucinates a non-GPCR chain → oligomer overrides it → receptor validation runs on the corrected chain → report is accurate.
4. Verify that no other validators depend on the pre-override chain state.