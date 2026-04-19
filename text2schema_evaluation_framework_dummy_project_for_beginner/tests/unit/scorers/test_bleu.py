"""Unit tests for BLEUScorer."""
import pytest
from schemaeval.scorers.bleu import BLEUScorer


@pytest.fixture
def scorer():
    return BLEUScorer()


def test_identical_dicts_high_score(scorer):
    d = {"product": "headphones", "price": 299}
    score = scorer.score(d, d)
    assert score > 0.9


def test_empty_golden_empty_generated(scorer):
    score = scorer.score({}, {})
    assert score == pytest.approx(1.0)


def test_empty_generated_nonempty_golden(scorer):
    score = scorer.score({}, {"a": 1})
    assert score == pytest.approx(0.0)


def test_partial_overlap_moderate_score(scorer):
    gen = {"name": "Alice", "age": 30, "role": "user"}
    gold = {"name": "Alice", "age": 30, "role": "admin"}
    score = scorer.score(gen, gold)
    assert 0.0 <= score <= 1.0


def test_score_bounded(scorer):
    gen = {"k": "value1", "n": 42}
    gold = {"k": "value2", "n": 99}
    s = scorer.score(gen, gold)
    assert 0.0 <= s <= 1.0


def test_completely_different_low_score(scorer):
    gen = {"z": "zzz", "y": 999}
    gold = {"a": "aaa", "b": 111}
    score = scorer.score(gen, gold)
    assert score < 0.5
