"""Unit tests for JaccardScorer."""
import pytest
from schemaeval.scorers.jaccard import JaccardScorer


@pytest.fixture
def scorer():
    return JaccardScorer()


def test_identical_dicts_score_one(scorer):
    d = {"a": 1, "b": "hello", "c": True}
    assert scorer.score(d, d) == pytest.approx(1.0)


def test_completely_different_dicts(scorer):
    gen = {"x": 1, "y": 2}
    gold = {"a": 10, "b": 20}
    score = scorer.score(gen, gold)
    assert 0.0 <= score < 0.5


def test_partial_overlap(scorer):
    gen = {"name": "Alice", "age": 30, "extra": "foo"}
    gold = {"name": "Alice", "age": 30, "role": "admin"}
    score = scorer.score(gen, gold)
    # "name" and "age" match = 2 out of 4 union keys
    assert 0.3 <= score <= 0.6


def test_empty_both(scorer):
    assert scorer.score({}, {}) == pytest.approx(1.0)


def test_one_empty_other_not(scorer):
    score = scorer.score({"a": 1}, {})
    assert score == pytest.approx(0.0)


def test_nested_identical(scorer):
    d = {"user": {"name": "Bob"}, "active": True}
    assert scorer.score(d, d) == pytest.approx(1.0)


def test_score_bounded(scorer):
    gen = {"k": "v", "n": 42}
    gold = {"k": "v", "x": "other"}
    s = scorer.score(gen, gold)
    assert 0.0 <= s <= 1.0
