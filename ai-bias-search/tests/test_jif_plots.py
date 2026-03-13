from __future__ import annotations

import pandas as pd
import pytest

from ai_bias_search.report.jif_plots import (
    bucket_quartiles,
    build_jif_context,
    cap_values,
    prepare_match_type_counts,
)


def test_prepare_match_type_counts_counts_expected() -> None:
    frame = pd.DataFrame(
        {
            "impact_factor_match": [
                "issn_exact",
                "title_exact",
                None,
                "",
                "title_fuzzy",
                "ambiguous",
                "none",
                "weird",
            ]
        }
    )
    counts = prepare_match_type_counts(frame, match_col="impact_factor_match")
    assert int(counts["issn_exact"]) == 1
    assert int(counts["title_exact"]) == 1
    assert int(counts["title_fuzzy"]) == 1
    assert int(counts["ambiguous"]) == 1
    assert int(counts["none"]) == 4


def test_bucket_quartiles_includes_unknown_bucket() -> None:
    series = pd.Series(["Q1", " q2 ", "", None, "q5", "Q4 "])
    bucketed = bucket_quartiles(series)
    assert bucketed.tolist() == ["Q1", "Q2", "Unknown", "Unknown", "Unknown", "Q4"]


def test_cap_values_clips_extremes_stably() -> None:
    series = pd.Series([1.0, 2.0, 3.0, 100.0])
    capped, cap = cap_values(series, upper_quantile=0.75)
    assert cap == pytest.approx(27.25)
    assert float(capped.iloc[0]) == 1.0
    assert float(capped.iloc[1]) == 2.0
    assert float(capped.iloc[2]) == 3.0
    assert float(capped.iloc[3]) == pytest.approx(27.25)

    capped2, cap2 = cap_values(series, upper_quantile=0.75)
    assert cap2 == pytest.approx(cap)
    assert capped2.equals(capped)


def test_build_jif_context_handles_missing_columns() -> None:
    frame = pd.DataFrame({"title": ["Paper"]})
    context = build_jif_context(frame)
    assert context["jif_enabled"] is False
    assert "not found" in str(context["jif_message"]).lower()
    assert context["jif_distribution_plot_png"] is None
