from pytest import approx

from ai_bias_search.evaluation.ranking import kendall_tau, spearman_rho


def test_spearman_perfect_match() -> None:
    left = [{"doi": f"10.0/{idx}", "rank": idx} for idx in range(1, 6)]
    right = [{"doi": f"10.0/{idx}", "rank": idx} for idx in range(1, 6)]
    assert spearman_rho(left, right) == approx(1.0)


def test_kendall_inverse_order() -> None:
    left = [{"doi": f"10.0/{idx}", "rank": idx} for idx in range(1, 6)]
    right = [
        {"doi": f"10.0/{idx}", "rank": rank}
        for idx, rank in zip(range(1, 6), reversed(range(1, 6)))
    ]
    assert kendall_tau(left, right) == approx(-1.0)
