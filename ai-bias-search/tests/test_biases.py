import pandas as pd

from ai_bias_search.evaluation.biases import compute_bias_metrics


def test_bias_metrics_basic() -> None:
    frame = pd.DataFrame(
        {
            "publication_year": [2022, 2023, 2023, 2020],
            "language": ["en", "en", "es", "es"],
            "is_oa": [True, False, True, True],
            "publisher": ["A", "A", "B", "C"],
            "rank": [1, 2, 3, 4],
            "cited_by_count": [10, 5, 20, 0],
        }
    )

    metrics = compute_bias_metrics(frame)
    assert metrics["recency"]["median_year"] == 2022.5
    assert metrics["open_access"]["share_open_access"] == 0.75
    assert set(metrics["language"].keys()) == {"en", "es"}
    assert metrics["publisher_hhi"]["hhi"] > 0
    assert metrics["rank_vs_citations"]["spearman"] is not None
