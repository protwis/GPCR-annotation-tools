"""Contract test: every receptor state the annotator can emit must be a State
token the GPCRdb structure import accepts.

The CSV writer exports each state as ``state.value.capitalize()`` into the
``State`` column of structures.csv, which the downstream GPCRdb build imports
directly. A state value that capitalizes to a token the import does not accept
would silently land bad data in the structure table. This test pins the state
enum to the set of tokens confirmed accepted downstream, so any future edit to
the enum fails loudly until the new value is verified.

(Parallel to ``test_role_vocab_contract`` for ligand roles; states are exported
verbatim-capitalized rather than slugified, so the contract is the simpler
value -> capitalized-token mapping.)
"""

from __future__ import annotations

from gpcr_tools.annotator.schema import ANNOTATION_TOOL

# Each emittable structure state -> the State token written to structures.csv
# (the CSV writer capitalizes state.value). Every token here is a State value the
# GPCRdb structure import accepts.
STATE_TO_CSV_TOKEN = {
    "inactive": "Inactive",
    "active": "Active",
    "other": "Other",
    "intermediate": "Intermediate",
    "unknown": "Unknown",
}


def _extract_state_enum() -> list[str]:
    """Pull the structure-state enum out of the annotation tool schema."""
    params = ANNOTATION_TOOL.function_declarations[0].parameters
    state = params.properties["structure_info"].properties["state"]
    return state.properties["value"].enum


class TestStateVocabularyContract:
    def test_enum_matches_pinned_vocabulary(self) -> None:
        # If this fails the state enum drifted: confirm every new value is a State
        # token the GPCRdb import accepts before adding it to STATE_TO_CSV_TOKEN.
        enum = _extract_state_enum()
        assert enum is not None
        assert set(enum) == set(STATE_TO_CSV_TOKEN)

    def test_every_state_capitalizes_to_pinned_token(self) -> None:
        # Pin the transform the CSV writer applies (state.value.capitalize()), so
        # the model-facing enum value and the exported token stay in lockstep.
        for state, expected_token in STATE_TO_CSV_TOKEN.items():
            assert state.capitalize() == expected_token
