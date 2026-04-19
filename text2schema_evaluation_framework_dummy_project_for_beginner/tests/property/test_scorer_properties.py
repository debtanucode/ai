"""Property-based tests using Hypothesis: reflexivity, symmetry, boundedness."""
from __future__ import annotations
import pytest
from hypothesis import given, settings as hyp_settings, HealthCheck
from hypothesis import strategies as st
from schemaeval.scorers.jaccard import JaccardScorer
from schemaeval.scorers.levenshtein import LevenshteinScorer
from schemaeval.scorers.field_diff import FieldDiffScorer

# Strategy for simple flat JSON dicts
json_value = st.one_of(
    st.integers(min_value=-1000, max_value=1000),
    st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    st.text(max_size=20),
    st.booleans(),
)
simple_dict = st.dictionaries(
    keys=st.text(alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")), min_size=1, max_size=10),
    values=json_value,
    max_size=6,
)


# ============================================================
# JaccardScorer properties
# ============================================================

@given(d=simple_dict)
@hyp_settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_jaccard_reflexivity(d):
    """score(d, d) == 1.0 for all d."""
    scorer = JaccardScorer()
    assert scorer.score(d, d) == pytest.approx(1.0)


@given(a=simple_dict, b=simple_dict)
@hyp_settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_jaccard_boundedness(a, b):
    """0.0 <= score(a, b) <= 1.0 for all a, b."""
    scorer = JaccardScorer()
    s = scorer.score(a, b)
    assert 0.0 <= s <= 1.0


@given(a=simple_dict, b=simple_dict)
@hyp_settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_jaccard_symmetry(a, b):
    """score(a, b) == score(b, a)."""
    scorer = JaccardScorer()
    assert scorer.score(a, b) == pytest.approx(scorer.score(b, a))


# ============================================================
# LevenshteinScorer properties
# ============================================================

@given(d=simple_dict)
@hyp_settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_levenshtein_reflexivity(d):
    scorer = LevenshteinScorer()
    assert scorer.score(d, d) == pytest.approx(1.0)


@given(a=simple_dict, b=simple_dict)
@hyp_settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_levenshtein_boundedness(a, b):
    scorer = LevenshteinScorer()
    s = scorer.score(a, b)
    assert 0.0 <= s <= 1.0


@given(a=simple_dict, b=simple_dict)
@hyp_settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_levenshtein_symmetry(a, b):
    scorer = LevenshteinScorer()
    assert scorer.score(a, b) == pytest.approx(scorer.score(b, a))


# ============================================================
# FieldDiffScorer properties
# ============================================================

@given(d=simple_dict)
@hyp_settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_field_diff_reflexivity(d):
    scorer = FieldDiffScorer()
    assert scorer.score(d, d) == pytest.approx(1.0)


@given(a=simple_dict, b=simple_dict)
@hyp_settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_field_diff_boundedness(a, b):
    scorer = FieldDiffScorer()
    s = scorer.score(a, b)
    assert 0.0 <= s <= 1.0


@given(d=simple_dict)
@hyp_settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_field_diff_details_counts_consistent(d):
    """matched + missing + extra + mismatched == total_fields in DiffResult."""
    scorer = FieldDiffScorer()
    result = scorer.score_with_details(d, d)
    # For identical: all matched, none missing/extra/mismatched
    assert result.missing == 0
    assert result.extra == 0
    assert result.mismatched == 0
