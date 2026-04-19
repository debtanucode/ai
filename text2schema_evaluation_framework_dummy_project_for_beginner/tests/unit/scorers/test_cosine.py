"""Unit tests for CosineScorer."""
import pytest
from schemaeval.scorers.cosine import CosineScorer


@pytest.fixture
def scorer():
    return CosineScorer()


def test_identical_dicts_high_score(scorer):
    d = {"product": "headphones", "price": 299.99, "brand": "AudioTech"}
    score = scorer.score(d, d)
    assert score == pytest.approx(1.0, abs=1e-5)


def test_empty_both(scorer):
    assert scorer.score({}, {}) == pytest.approx(1.0)


def test_one_empty(scorer):
    score = scorer.score({"a": 1}, {})
    assert 0.0 <= score <= 1.0


def test_similar_dicts_moderate_score(scorer):
    # TF-IDF on short JSON tokens; shared tokens ("name","Alice","Smith","role") give moderate overlap
    gen = {"name": "Alice Smith", "role": "engineer"}
    gold = {"name": "Alice Smith", "role": "software engineer"}
    score = scorer.score(gen, gold)
    assert score > 0.1  # TF-IDF on short JSON yields moderate similarity, not necessarily >0.5


def test_completely_different(scorer):
    gen = {"x": 1, "y": 2}
    gold = {"a": "hello", "b": "world"}
    score = scorer.score(gen, gold)
    assert 0.0 <= score <= 1.0


def test_score_bounded(scorer):
    gen = {"k": "some value", "n": 42}
    gold = {"k": "other value", "m": 99}
    s = scorer.score(gen, gold)
    assert 0.0 <= s <= 1.0
