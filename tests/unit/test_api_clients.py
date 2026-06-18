"""Unit tests for the existence-check robustness in validator.api_clients.

The key guarantee: a definitive HTTP 200/404 is a verdict (cached), but a
transient failure (5xx/429, timeout, network error) abstains (returns ``None``)
and is NEVER cached -- so a real id is not reported "does not exist" during an
API outage, and the persistent cache is never poisoned with a transient False.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import requests

from gpcr_tools.config import (
    API_MAX_RETRIES,
    SLEEP_VALIDATION_RETRY,
    VALIDATION_RETRY_BACKOFF_FACTOR,
)
from gpcr_tools.validator.api_clients import (
    check_pubchem_existence,
    check_pubchem_synonym_match,
    check_uniprot_existence,
    fetch_polymer_features_cached,
)


class _FakePolymerFeaturesCache:
    """Minimal dict-valued cache mirroring PolymerFeaturesCache's get/set."""

    def __init__(self) -> None:
        self._d: dict[str, dict] = {}

    def get(self, key: str) -> dict | None:
        return self._d.get(key)

    def set(self, key: str, value: dict) -> None:
        self._d[key] = value


class TestFetchPolymerFeaturesCached:
    def test_cache_hit_short_circuits_live_fetch(self) -> None:
        cache = _FakePolymerFeaturesCache()
        cache.set("7RKF", {"polymer_entities": ["cached"]})
        with patch("gpcr_tools.validator.api_clients.fetch_polymer_features") as mock_fetch:
            result = fetch_polymer_features_cached("7RKF", cache)
        assert result == {"polymer_entities": ["cached"]}
        mock_fetch.assert_not_called()

    def test_miss_fetches_live_and_stores(self) -> None:
        cache = _FakePolymerFeaturesCache()
        entry = {"polymer_entities": ["live"]}
        with patch(
            "gpcr_tools.validator.api_clients.fetch_polymer_features",
            return_value=entry,
        ):
            result = fetch_polymer_features_cached("7RKF", cache)
        assert result == entry
        assert cache.get("7RKF") == entry

    def test_failed_fetch_is_not_cached(self) -> None:
        # A transient failure (None) must never be frozen as a cached fact.
        cache = _FakePolymerFeaturesCache()
        with patch(
            "gpcr_tools.validator.api_clients.fetch_polymer_features",
            return_value=None,
        ):
            result = fetch_polymer_features_cached("7RKF", cache)
        assert result is None
        assert cache.get("7RKF") is None


class _FakeCache:
    """Minimal bool-valued cache mirroring ValidationCache's get/contains/set."""

    def __init__(self) -> None:
        self._d: dict[str, bool] = {}

    def get(self, key: str) -> bool | None:
        return self._d.get(key)

    def __contains__(self, key: str) -> bool:
        return key in self._d

    def set(self, key: str, value: bool) -> None:
        self._d[key] = value


def _resp(status: int) -> MagicMock:
    r = MagicMock()
    r.status_code = status
    return r


@patch("gpcr_tools.validator.api_clients.time.sleep", lambda *_: None)
class TestCheckUniprotExistence:
    @patch("gpcr_tools.validator.api_clients.requests.head")
    def test_200_is_true_and_cached(self, mock_head: MagicMock) -> None:
        mock_head.return_value = _resp(200)
        cache = _FakeCache()
        assert check_uniprot_existence("oprm_human", cache) is True
        assert cache.get("uniprot:oprm_human") is True

    @patch("gpcr_tools.validator.api_clients.requests.head")
    def test_404_is_false_and_cached(self, mock_head: MagicMock) -> None:
        mock_head.return_value = _resp(404)
        cache = _FakeCache()
        assert check_uniprot_existence("not_a_real_slug", cache) is False
        assert cache.get("uniprot:not_a_real_slug") is False

    @patch("gpcr_tools.validator.api_clients.requests.head")
    def test_503_abstains_and_not_cached(self, mock_head: MagicMock) -> None:
        # The anti-poisoning guarantee: a transient outage must NOT cache a False
        # and must NOT report a real entry as absent.
        mock_head.return_value = _resp(503)
        cache = _FakeCache()
        assert check_uniprot_existence("oprm_human", cache) is None
        assert "uniprot:oprm_human" not in cache

    @patch("gpcr_tools.validator.api_clients.requests.head")
    def test_network_error_abstains_and_not_cached(self, mock_head: MagicMock) -> None:
        mock_head.side_effect = requests.RequestException("boom")
        cache = _FakeCache()
        assert check_uniprot_existence("oprm_human", cache) is None
        assert "uniprot:oprm_human" not in cache

    @patch("gpcr_tools.validator.api_clients.requests.head")
    def test_400_is_malformed_query_not_retried_not_cached(self, mock_head: MagicMock) -> None:
        # HTTP 400 = the query itself is malformed (e.g. a comma-joined multi-slug
        # receptor name). The service is fine, so retrying repeats the rejection:
        # abstain immediately (return None), do NOT retry, and never cache it.
        mock_head.return_value = _resp(400)
        cache = _FakeCache()
        assert check_uniprot_existence("grm2_human, grm7_human", cache) is None
        assert mock_head.call_count == 1
        assert "uniprot:grm2_human, grm7_human" not in cache


@patch("gpcr_tools.validator.api_clients.time.sleep", lambda *_: None)
class TestCheckPubchemExistence:
    @patch("gpcr_tools.validator.api_clients.requests.get")
    def test_200_is_true_and_cached(self, mock_get: MagicMock) -> None:
        mock_get.return_value = _resp(200)
        cache = _FakeCache()
        assert check_pubchem_existence("5462471", cache) is True
        assert cache.get("pubchem:5462471") is True

    @patch("gpcr_tools.validator.api_clients.requests.get")
    def test_404_is_false_and_cached(self, mock_get: MagicMock) -> None:
        mock_get.return_value = _resp(404)
        cache = _FakeCache()
        assert check_pubchem_existence("999999999", cache) is False
        assert cache.get("pubchem:999999999") is False

    @patch("gpcr_tools.validator.api_clients.requests.get")
    def test_503_abstains_and_not_cached(self, mock_get: MagicMock) -> None:
        mock_get.return_value = _resp(503)
        cache = _FakeCache()
        assert check_pubchem_existence("5462471", cache) is None
        assert "pubchem:5462471" not in cache

    @patch("gpcr_tools.validator.api_clients.requests.get")
    def test_network_error_abstains_and_not_cached(self, mock_get: MagicMock) -> None:
        mock_get.side_effect = requests.RequestException("boom")
        cache = _FakeCache()
        assert check_pubchem_existence("5462471", cache) is None
        assert "pubchem:5462471" not in cache

    def test_non_numeric_cid_is_false_without_network(self) -> None:
        cache = _FakeCache()
        assert check_pubchem_existence("not-a-cid", cache) is False


@patch("gpcr_tools.validator.api_clients.time.sleep", lambda *_: None)
class TestCheckPubchemExistenceRetry:
    @patch("gpcr_tools.validator.api_clients.requests.get")
    def test_transient_then_success_is_retried(self, mock_get: MagicMock) -> None:
        # A 503 on the first attempt must be retried; the next 200 is the verdict.
        mock_get.side_effect = [_resp(503), _resp(200)]
        cache = _FakeCache()
        assert check_pubchem_existence("5462471", cache) is True
        assert cache.get("pubchem:5462471") is True
        assert mock_get.call_count == 2

    @patch("gpcr_tools.validator.api_clients.requests.get")
    def test_all_transient_abstains_and_not_cached(self, mock_get: MagicMock) -> None:
        mock_get.side_effect = [_resp(503)] * API_MAX_RETRIES
        cache = _FakeCache()
        assert check_pubchem_existence("5462471", cache) is None
        assert "pubchem:5462471" not in cache
        assert mock_get.call_count == API_MAX_RETRIES

    @patch("gpcr_tools.validator.api_clients.requests.get")
    def test_404_is_not_retried(self, mock_get: MagicMock) -> None:
        mock_get.return_value = _resp(404)
        cache = _FakeCache()
        assert check_pubchem_existence("999999999", cache) is False
        assert mock_get.call_count == 1


class _SynonymCache:
    """Minimal list-valued cache mirroring the SynonymCache protocol."""

    def __init__(self) -> None:
        self._d: dict[str, list[str]] = {}

    def has(self, key: str) -> bool:
        return key in self._d

    def get(self, key: str):  # type: ignore[no-untyped-def]
        return self._d.get(key)

    def set(self, key: str, value, *, allow_none: bool = False) -> None:  # type: ignore[no-untyped-def]
        self._d[key] = value


def _synonym_resp(status: int, synonyms: list[str] | None = None) -> MagicMock:
    r = MagicMock()
    r.status_code = status
    if synonyms is not None:
        r.json.return_value = {"InformationList": {"Information": [{"Synonym": synonyms}]}}
    return r


@patch("gpcr_tools.validator.api_clients.time.sleep", lambda *_: None)
class TestCheckPubchemSynonymMatchRetry:
    @patch("gpcr_tools.validator.api_clients.requests.get")
    def test_transient_then_success_is_retried(self, mock_get: MagicMock) -> None:
        # First a network error, then a 200 whose synonyms include the candidate.
        mock_get.side_effect = [
            requests.RequestException("boom"),
            _synonym_resp(200, ["morphine"]),
        ]
        cache = _SynonymCache()
        assert check_pubchem_synonym_match("5288826", ["Morphine"], cache) is True
        assert mock_get.call_count == 2
        assert cache.get("5288826") == ["morphine"]

    @patch("gpcr_tools.validator.api_clients.requests.get")
    def test_all_transient_abstains_and_not_cached(self, mock_get: MagicMock) -> None:
        mock_get.side_effect = [requests.RequestException("boom")] * API_MAX_RETRIES
        cache = _SynonymCache()
        assert check_pubchem_synonym_match("5288826", ["Morphine"], cache) is None
        assert not cache.has("5288826")
        assert mock_get.call_count == API_MAX_RETRIES

    @patch("gpcr_tools.validator.api_clients.requests.get")
    def test_404_is_not_retried_and_cached_empty(self, mock_get: MagicMock) -> None:
        mock_get.return_value = _synonym_resp(404)
        cache = _SynonymCache()
        assert check_pubchem_synonym_match("5288826", ["Morphine"], cache) is False
        assert mock_get.call_count == 1
        assert cache.get("5288826") == []

    @patch("gpcr_tools.validator.api_clients.requests.get")
    def test_400_is_malformed_query_not_retried_not_cached(self, mock_get: MagicMock) -> None:
        # HTTP 400 = malformed query, not a transient outage: abstain immediately
        # (None), do NOT retry, never cache. (This function's status checks live
        # outside its try block, so it's worth covering separately.)
        mock_get.return_value = _synonym_resp(400)
        cache = _SynonymCache()
        assert check_pubchem_synonym_match("5288826", ["Morphine"], cache) is None
        assert mock_get.call_count == 1
        assert not cache.has("5288826")

    @patch("gpcr_tools.validator.api_clients.requests.get")
    def test_parse_error_on_200_is_not_retried(self, mock_get: MagicMock) -> None:
        # A malformed body on a 200 abstains immediately -- no retry, no cache write.
        bad = MagicMock()
        bad.status_code = 200
        bad.json.side_effect = ValueError("bad json")
        mock_get.return_value = bad
        cache = _SynonymCache()
        assert check_pubchem_synonym_match("5288826", ["Morphine"], cache) is None
        assert mock_get.call_count == 1
        assert not cache.has("5288826")


def test_validation_retry_uses_exponential_backoff(monkeypatch: object) -> None:
    """Transient failures back off geometrically (1s, 3s, ...) between retries
    rather than three rapid-fire 1s waits, so a rate-limit/congestion spike gets a
    widening recovery gap. The terminal attempt does not sleep, and nothing is cached.
    """
    sleeps: list[float] = []
    monkeypatch.setattr("gpcr_tools.validator.api_clients.time.sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(
        "gpcr_tools.validator.api_clients.requests.get",
        MagicMock(side_effect=requests.RequestException("boom")),
    )
    cache = _FakeCache()
    assert check_pubchem_existence("5462471", cache) is None
    # One sleep per attempt except the last; each grows by VALIDATION_RETRY_BACKOFF_FACTOR.
    expected = [
        SLEEP_VALIDATION_RETRY * VALIDATION_RETRY_BACKOFF_FACTOR**i
        for i in range(API_MAX_RETRIES - 1)
    ]
    assert sleeps == expected
    assert "pubchem:5462471" not in cache
