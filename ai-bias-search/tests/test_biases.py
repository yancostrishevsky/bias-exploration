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


def test_bias_metrics_basic() -> None:
    frame = pd.DataFrame(
        {
            "doi": ["d1", "d2", "d3", "d4"],
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
    assert metrics["rank_vs_citations"]["spearman"] is not None
    assert metrics["completeness"] == {
        "doi": approx(1.0),
        "year_coverage": approx(1.0),
        "language": approx(1.0),
        "publisher": approx(1.0),
        "is_oa": approx(1.0),
    }
    assert set(metrics["by_platform"].keys()) == {"p1", "p2"}
    assert metrics["by_platform"]["p1"]["open_access"]["share_open_access"] == approx(0.5)
    assert metrics["by_platform"]["p2"]["open_access"]["share_open_access"] == approx(1.0)
    assert metrics["open_access_by_platform"] == {
        "p1": {"share_open_access": approx(0.5)},
        "p2": {"share_open_access": approx(1.0)},
    }
