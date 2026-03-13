from __future__ import annotations

import pandas as pd

from ai_bias_search.utils.io import _sanitize_empty_dict_columns


def test_sanitize_empty_dict_only_columns() -> None:
    frame = pd.DataFrame(
        [
            {"title": "Paper A", "rank": 1, "scopus": {}, "scopus_meta": {}},
            {"title": "Paper B", "rank": 2, "scopus": {}, "scopus_meta": {}},
        ]
    )

    sanitized = _sanitize_empty_dict_columns(frame)
    assert sanitized["scopus"].isna().all()
    assert sanitized["scopus_meta"].isna().all()


def test_sanitize_preserves_mixed_dict_columns() -> None:
    frame = pd.DataFrame(
        [
            {"title": "Paper A", "rank": 1, "scopus": {}},
            {"title": "Paper B", "rank": 2, "scopus": {"abstract": {"scopus_id": "85012345678"}}},
        ]
    )

    sanitized = _sanitize_empty_dict_columns(frame)
    assert sanitized.loc[0, "scopus"] == {}
    assert sanitized.loc[1, "scopus"]["abstract"]["scopus_id"] == "85012345678"
