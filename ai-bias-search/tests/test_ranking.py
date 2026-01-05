import math

from pytest import approx

from ai_bias_search.evaluation.ranking_similarity import rbo


def test_rbo_identical_lists() -> None:
    left = [{"doi": f"10.0/{idx}"} for idx in range(1, 6)]
    right = [{"doi": f"10.0/{idx}"} for idx in range(1, 6)]
    assert rbo(left, right) == approx(1.0)


def test_rbo_disjoint_lists() -> None:
    left = [{"doi": f"10.0/{idx}"} for idx in range(1, 4)]
    right = [{"doi": f"10.1/{idx}"} for idx in range(1, 4)]
    assert rbo(left, right) == approx(0.0)


def test_rbo_prefers_top_overlap() -> None:
    left = [{"doi": doi} for doi in ["a", "b", "c", "d"]]
    top_overlap = [{"doi": doi} for doi in ["a", "b", "x", "y"]]
    deep_overlap = [{"doi": doi} for doi in ["x", "y", "a", "b"]]

    assert rbo(left, top_overlap) > rbo(left, deep_overlap)


def test_rbo_handles_different_lengths() -> None:
    left = [{"doi": f"10.0/{idx}"} for idx in range(1, 6)]
    right = [{"doi": "10.0/1"}, {"doi": "10.0/4"}, {"doi": "10.0/5"}]

    score = rbo(left, right)
    assert 0.0 <= score <= 1.0
    assert not math.isnan(score)
