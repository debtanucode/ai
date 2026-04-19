"""Unit tests for ROUGEScorer."""
import pytest
from schemaeval.scorers.rouge import ROUGEScorer


@pytest.fixture
def scorer():
    return ROUGEScorer()


def test_identical_is_one(scorer):
    d = {"name": "Alice", "role": "admin", "active": True}
    assert scorer.score(d, d) == pytest.approx(1.0)


def test_empty_both(scorer):
    assert scorer.score({}, {}) == pytest.approx(1.0)


def test_completely_different(scorer):
    gen = {"x": 1, "y": 2}
    gold = {"a": 100, "b": 200}
    score = scorer.score(gen, gold)
    assert 0.0 <= score < 1.0


def test_one_field_different(scorer):
    gen = {"name": "Alice", "age": 30, "city": "NYC"}
    gold = {"name": "Alice", "age": 30, "city": "LA"}
    score = scorer.score(gen, gold)
    assert score > 0.5


def test_score_bounded(scorer):
    gen = {"k": "something", "v": 42}
    gold = {"k": "other", "v": 99}
    s = scorer.score(gen, gold)
    assert 0.0 <= s <= 1.0


def test_nested_identical(scorer):
    d = {"user": {"name": "Bob", "email": "bob@example.com"}}
    assert scorer.score(d, d) == pytest.approx(1.0)
