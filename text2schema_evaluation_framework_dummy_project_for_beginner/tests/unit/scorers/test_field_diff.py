"""Unit tests for FieldDiffScorer."""
import pytest
from schemaeval.scorers.field_diff import FieldDiffScorer
from schemaeval.models.diff import FieldStatus


@pytest.fixture
def scorer():
    return FieldDiffScorer()


def test_identical_dicts_all_matched(scorer):
    d = {"name": "Alice", "age": 30, "role": "admin"}
    result = scorer.score_with_details(d, d)
    assert result.matched == 3
    assert result.missing == 0
    assert result.extra == 0
    assert result.mismatched == 0
    assert scorer.score(d, d) == pytest.approx(1.0)


def test_missing_fields(scorer):
    gen = {"name": "Alice"}
    gold = {"name": "Alice", "age": 30, "role": "admin"}
    result = scorer.score_with_details(gen, gold)
    assert result.missing == 2
    assert result.matched == 1


def test_extra_fields(scorer):
    gen = {"name": "Alice", "extra_field": "foo"}
    gold = {"name": "Alice"}
    result = scorer.score_with_details(gen, gold)
    assert result.extra == 1
    assert result.matched == 1


def test_mismatch_fields(scorer):
    gen = {"name": "Alice", "age": 31}
    gold = {"name": "Alice", "age": 30}
    result = scorer.score_with_details(gen, gold)
    assert result.mismatched == 1
    assert result.matched == 1


def test_empty_both(scorer):
    result = scorer.score_with_details({}, {})
    assert result.matched == 0
    assert result.total_fields == 0
    assert scorer.score({}, {}) == pytest.approx(1.0)


def test_nested_structure(scorer):
    gen = {"user": {"name": "Bob", "age": 25}}
    gold = {"user": {"name": "Bob", "age": 25}}
    result = scorer.score_with_details(gen, gold)
    assert result.matched == 2


def test_score_zero_when_all_missing(scorer):
    gen = {}
    gold = {"a": 1, "b": 2, "c": 3}
    score = scorer.score(gen, gold)
    assert score == pytest.approx(0.0)


def test_field_status_types(scorer):
    gen = {"name": "Alice", "extra": "bonus"}
    gold = {"name": "Bob", "missing_key": "val"}
    result = scorer.score_with_details(gen, gold)
    statuses = {f.path: f.status for f in result.fields}
    assert statuses["name"] == FieldStatus.MISMATCH
    assert statuses["extra"] == FieldStatus.EXTRA
    assert statuses["missing_key"] == FieldStatus.MISSING
