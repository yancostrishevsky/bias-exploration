from ai_bias_search.evaluation.overlap import jaccard, overlap_at_k


def test_jaccard_basic() -> None:
    left = {"10.1234/foo", "10.1234/bar"}
    right = {"10.1234/bar", "10.1234/baz"}
    assert jaccard(left, right) == 1 / 3


def test_overlap_at_k() -> None:
    left = [
        {"doi": "10.1234/foo", "rank": 1},
        {"doi": "10.1234/bar", "rank": 2},
    ]
    right = [
        {"doi": "10.1234/bar", "rank": 1},
        {"doi": "10.1234/baz", "rank": 2},
    ]
    assert overlap_at_k(left, right, k=2) == 0.5
