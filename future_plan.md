# Future Development Plan

## 1. Type Enforcement in Interactive Review

**Issue:** In `review_engine.py`'s `coerce_type` function, the system attempts to coerce user input back to the original data type. However, if the coercion fails (e.g., trying to parse "apple" when the original value was a boolean `True`), it silently falls back to returning the string.

**Risk:** While this matches the legacy behavior ("Fallback -> keeps as string"), it can lead to type degradation and data pollution in the output CSVs (e.g., a boolean column suddenly containing string values due to a typo).

**Action Plan:**
Refactor the prompt logic at the UI layer. When a user is editing a field and the input cannot be robustly coerced to the original type (especially for strict types like booleans or numbers), the UI should reject the input, display a validation error message ("Expected a boolean/number"), and ask the user to retry, rather than silently accepting a string fallback.