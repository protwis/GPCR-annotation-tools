from gpcr_tools.annotator.post_processor import (
    _is_signaling_partners_empty,
    _standardize_auxiliary_name,
    _unwrap_composite,
    is_meaningfully_empty,
    post_process_annotation,
)


def test_unwrap_composite():
    class DummyCompositeMap:
        def __init__(self, data):
            self.data = data

        def items(self):
            return self.data.items()

        def __iter__(self):
            return iter(self.data)

        def __getitem__(self, k):
            return self.data[k]

    # Python 3 map duck-typing mapping. Mapping check covers dicts.
    from collections.abc import Mapping, Sequence

    class DummyMapping(Mapping):
        def __init__(self, d):
            self.d = d

        def __getitem__(self, key):
            return self.d[key]

        def __iter__(self):
            return iter(self.d)

        def __len__(self):
            return len(self.d)

    class DummySequence(Sequence):
        def __init__(self, s):
            self.s = s

        def __getitem__(self, i):
            return self.s[i]

        def __len__(self):
            return len(self.s)

    # Test that a custom mapping converts to a python dict
    composite = DummyMapping({"key": DummySequence(["val"])})
    result = _unwrap_composite(composite)
    assert isinstance(result, dict)
    assert result == {"key": ["val"]}

    # Ensure native structures are preserved appropriately
    assert _unwrap_composite({"a": [1, {"b": 2}]}) == {"a": [1, {"b": 2}]}


def test_is_meaningfully_empty():
    assert is_meaningfully_empty(None) is True
    assert is_meaningfully_empty("") is True
    assert is_meaningfully_empty([]) is True
    assert is_meaningfully_empty({}) is True
    assert is_meaningfully_empty("N/A") is True
    assert is_meaningfully_empty(" none ") is True
    assert is_meaningfully_empty("-") is True
    assert is_meaningfully_empty("missing") is True
    assert is_meaningfully_empty("not present") is True

    assert is_meaningfully_empty("Valid string") is False
    assert is_meaningfully_empty(["content"]) is False
    assert is_meaningfully_empty(0) is False  # 0 is not a string but falsey
    assert is_meaningfully_empty(False) is False


def test_is_signaling_partners_empty():
    assert _is_signaling_partners_empty({}) is True
    assert _is_signaling_partners_empty({"note": "They are missing"}) is True
    assert _is_signaling_partners_empty({"g_protein": {}}) is True
    assert _is_signaling_partners_empty({"g_protein": {"alpha": "gnas"}}) is False


def test_standardize_auxiliary_name():
    assert _standardize_auxiliary_name("BRIL") == "BRIL"
    assert _standardize_auxiliary_name("bRiL") == "BRIL"
    assert _standardize_auxiliary_name("T4-Lysozyme-BRIL fusion") == "BRIL"
    assert _standardize_auxiliary_name("cytochrome b562 RIL") == "BRIL"

    # Nanobody standardisation
    assert _standardize_auxiliary_name("Nb35") == "Nanobody-35"
    assert _standardize_auxiliary_name("nb-12") == "Nanobody-12"
    assert _standardize_auxiliary_name("Nanobody") == "Nanobody"
    assert _standardize_auxiliary_name(None) is None


def test_post_process_annotation():
    raw_response = {
        "receptor_info": {
            "uniprot_entry_name": "OPSD_BOVIN",
        },
        "signaling_partners": {"note": "None found", "g_protein": {}},
        "auxiliary_proteins": [
            {"name": "Nb6", "type": "Nanobody"},
            {"name": "cytochrome b562 ril", "type": "Fusion"},
        ],
    }

    result = post_process_annotation(raw_response)

    # 1. Lowercase receptor_info uniprot
    assert result["receptor_info"]["uniprot_entry_name"] == "opsd_bovin"

    # 2. empty signaling partners deleted
    assert "signaling_partners" not in result

    # 3. auxiliary proteins parsed
    assert result["auxiliary_proteins"][0]["name"] == "Nanobody-6"
    assert result["auxiliary_proteins"][1]["name"] == "BRIL"


def test_post_process_signaling_partners_lower_uniprot():
    raw_response = {
        "signaling_partners": {
            "g_protein": {"alpha_subunit": {"uniprot_entry_name": "GNAS_HUMAN"}},
            "arrestin": {"uniprot_entry_name": "ARRB1_HUMAN"},
        }
    }
    result = post_process_annotation(raw_response)
    assert (
        result["signaling_partners"]["g_protein"]["alpha_subunit"]["uniprot_entry_name"]
        == "gnas_human"
    )
    assert result["signaling_partners"]["arrestin"]["uniprot_entry_name"] == "arrb1_human"
