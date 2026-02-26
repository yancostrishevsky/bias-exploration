from datetime import datetime

import numpy as np
import pandas as pd
from pytest import approx

from ai_bias_search.evaluation.biases import (
    _metadata_completeness,
    _recency_metrics,
    compute_bias_metrics,
)


def test_recency_empty_frame() -> None:
    result = _recency_metrics(pd.DataFrame())
    assert all(value is None for value in result.values())


def test_recency_ignores_implausible_years() -> None:
    frame = pd.DataFrame({"publication_year": [1700, datetime.utcnow().year + 5]})
    result = _recency_metrics(frame)
    assert all(value is None for value in result.values())


def test_recency_known_values() -> None:
    current_year = datetime.utcnow().year
    years = [current_year, current_year - 2, current_year - 5, current_year - 11, current_year - 20]
    frame = pd.DataFrame({"publication_year": years})

    result = _recency_metrics(frame)
    assert result["median_year"] == approx(float(current_year - 5))
    assert result["median_age_years"] == approx(5.0)
    assert result["share_age_le_2"] == approx(2 / 5)
    assert result["share_age_le_5"] == approx(3 / 5)
    assert result["share_age_le_10"] == approx(3 / 5)
    assert result["share_age_gt_10"] == approx(2 / 5)


def test_recency_prefers_publication_year() -> None:
    frame = pd.DataFrame({"publication_year": [2000, 2001], "year": [1990, 1991]})
    result = _recency_metrics(frame)
    assert result["median_year"] == approx(2000.5)
    expected_median_age = datetime.utcnow().year - 2000.5
    assert result["median_age_years"] == approx(expected_median_age)


def test_metadata_completeness_missing_columns() -> None:
    frame = pd.DataFrame({"foo": [1, 2]})
    result = _metadata_completeness(frame)
    assert result == {
        "doi": None,
        "year_coverage": None,
        "language": None,
        "publisher": None,
        "is_oa": None,
        "citations": None,
        "issn_coverage": None,
    }


def test_metadata_completeness_treats_empty_strings_as_missing() -> None:
    frame = pd.DataFrame(
        {
            "doi": ["", None, "10/123"],
            "language": ["en", "", "fr"],
            "publication_year": [2020, None, 2021],
        }
    )
    result = _metadata_completeness(frame)
    assert result["doi"] == approx(1 / 3)
    assert result["language"] == approx(2 / 3)
    assert result["year_coverage"] == approx(2 / 3)
    assert result["publisher"] is None
    assert result["is_oa"] is None
    assert result["citations"] is None
    assert result["issn_coverage"] is None


def test_bias_metrics_basic() -> None:
    frame = pd.DataFrame(
        {
            "doi": ["10.1000/d1", "10.1000/d2", "10.1000/d3", "10.1000/d4"],
            "publication_year": [2022, 2023, 2023, 2020],
            "language": ["en", "en", "es", "es"],
            "is_oa": [True, False, True, True],
            "publisher": ["A", "A", "B", "C"],
            "rank": [1, 2, 3, 4],
            "cited_by_count": [10, 5, 20, 0],
            "platform": ["p1", "p1", "p2", "p2"],
        }
    )

    metrics = compute_bias_metrics(frame)
    recency = metrics["recency"]
    current_year = datetime.utcnow().year
    ages = [current_year - year for year in frame["publication_year"]]
    assert recency["median_year"] == approx(2022.5)
    assert recency["median_age_years"] == approx(float(np.median(ages)))
    assert metrics["open_access"]["share_open_access"] == 0.75
    assert set(metrics["language"].keys()) == {"en", "es"}
    assert metrics["publisher_hhi"]["hhi"] > 0
    assert metrics["rank_vs_citations"]["citations_not_cross_platform_comparable"] is True
    assert metrics["rank_vs_citations"]["spearman"] is None
    assert metrics["completeness"] == {
        "doi": approx(1.0),
        "year_coverage": approx(1.0),
        "language": approx(1.0),
        "publisher": approx(1.0),
        "is_oa": approx(1.0),
        "citations": approx(1.0),
        "issn_coverage": approx(0.0),
    }
    assert set(metrics["by_platform"].keys()) == {"p1", "p2"}
    assert metrics["by_platform"]["p1"]["open_access"]["share_open_access"] == approx(0.5)
    assert metrics["by_platform"]["p2"]["open_access"]["share_open_access"] == approx(1.0)
    assert metrics["open_access_by_platform"] == {
        "p1": {"share_open_access": approx(0.5)},
        "p2": {"share_open_access": approx(1.0)},
    }


def test_top_k_bias_metrics_available() -> None:
    frame = pd.DataFrame(
        {
            "platform": ["p1"] * 6,
            "rank": [1, 2, 3, 4, 5, 6],
            "is_oa": [True, True, False, False, True, False],
            "affiliation_countries": [
                ["PL"],
                ["PL", "DE"],
                ["DE"],
                ["US"],
                ["US"],
                ["PL"],
            ],
            "cited_by_count": [30, 20, 10, 8, 2, 1],
            "doc_type": ["Article", "Article", "Review", "Review", "Article", "Note"],
            "journal_title": ["J1", "J1", "J2", "J3", "J4", "J5"],
            "issn_list": [["12345678"], ["12345678"], ["87654321"], ["11112222"], ["33334444"], []],
        }
    )
    metrics = compute_bias_metrics(frame)
    top_k = metrics["top_k_bias"]
    assert top_k["oa"]["available"] is True
    assert top_k["oa"]["per_k"]["10"]["effective_k"] == 6
    assert top_k["country"]["per_k"]["10"]["js_divergence"] is not None
    assert top_k["citations"]["spearman_rank_vs_citations"] is not None
    assert top_k["doc_type"]["per_k"]["10"]["js_divergence"] is not None
    assert top_k["journal_issn"]["per_k"]["10"]["unique_journal_title_count"] == 5


def test_top_k_bias_metrics_mark_not_available_when_missing() -> None:
    frame = pd.DataFrame({"platform": ["p1", "p1"], "rank": [1, 2]})
    metrics = compute_bias_metrics(frame)
    top_k = metrics["top_k_bias"]
    assert top_k["oa"]["available"] is False
    assert top_k["country"]["available"] is False
    assert top_k["doc_type"]["available"] is False
    assert top_k["citations"]["available"] is False


def test_top_k_citation_reliability_gating_blocks_unstable_delta() -> None:
    frame = pd.DataFrame(
        {
            "platform": ["p1"] * 20,
            "rank": list(range(1, 21)),
            "cited_by_count": [100, 50] + [None] * 8 + [5] * 10,
            "publication_year": [2020] * 20,
        }
    )
    metrics = compute_bias_metrics(frame)
    citations_top10 = metrics["top_k_bias"]["citations"]["per_k"]["10"]
    assert citations_top10["minimum_required"] == 10
    assert citations_top10["available_count"] == 2
    assert citations_top10["available"] is False
    assert citations_top10["delta_median_top_vs_rest"] is None
    assert citations_top10["reliability"] == "low"


def test_multi_platform_citations_are_marked_non_comparable() -> None:
    frame = pd.DataFrame(
        {
            "platform": ["a", "a", "b", "b"],
            "rank": [1, 2, 1, 2],
            "cited_by_count": [10, 5, 100, 50],
            "publication_year": [2020, 2021, 2020, 2021],
        }
    )
    metrics = compute_bias_metrics(frame)
    assert metrics["citations_not_cross_platform_comparable"] is True
    assert metrics["rank_vs_citations"]["citations_not_cross_platform_comparable"] is True
    assert set(metrics["rank_vs_citations"]["per_platform"].keys()) == {"a", "b"}
    assert metrics["top_k_bias"]["citations"]["citations_not_cross_platform_comparable"] is True
