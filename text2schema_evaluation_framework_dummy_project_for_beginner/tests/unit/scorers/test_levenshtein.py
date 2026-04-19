"""Unit tests for LevenshteinScorer."""
import pytest
from schemaeval.scorers.levenshtein import LevenshteinScorer


@pytest.fixture
def scorer():
    return LevenshteinScorer()


def test_identical_is_one(scorer):
    d = {"a": 1, "b": "hello"}
    assert scorer.score(d, d) == pytest.approx(1.0)


def test_empty_both_is_one(scorer):
    assert scorer.score({}, {}) == pytest.approx(1.0)


def test_completely_different_is_less_than_one(scorer):
    gen = {"z": 999, "y": "zzz"}
    gold = {"a": 1, "b": "aaa"}
    assert scorer.score(gen, gold) < 1.0


def test_single_char_diff(scorer):
    gen = {"name": "Alice"}
    gold = {"name": "Alicf"}
    score = scorer.score(gen, gold)
    assert score > 0.9  # Very close


def test_score_bounded(scorer):
    gen = {"x": "foo", "y": 1}
    gold = {"x": "bar", "z": 2}
    s = scorer.score(gen, gold)
    assert 0.0 <= s <= 1.0


def test_partial_match(scorer):
    gen = {"name": "Alice", "age": 30}
    gold = {"name": "Alice", "age": 999}
    score = scorer.score(gen, gold)
    assert 0.5 < score < 1.0


def test_extra_keys_lower_score(scorer):
    gen = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
    gold = {"a": 1}
    score = scorer.score(gen, gold)
    assert score < 1.0
