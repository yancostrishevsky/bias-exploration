from __future__ import annotations

from pathlib import Path

from ai_bias_search.normalization.openalex_enrich import extract_acronym_from_venue_text
from ai_bias_search.utils import core_rankings


def _write_core_csv(path: Path, content: str) -> None:
    path.write_text(content.strip() + "\n", encoding="utf-8")


def test_token_acronyms_require_core_membership(tmp_path: Path) -> None:
    core_path = tmp_path / "core.csv"
    _write_core_csv(
        core_path,
        """
id,title,acronym,source,rank,core
1,ACM SIGIR Conference on Research and Development in Information Retrieval,SIGIR,CORE2023,A*,Yes
""",
    )

    core_rankings.set_core_rankings_path(core_path)
    core_rankings.clear_core_rankings_cache()
    try:
        assert (
            extract_acronym_from_venue_text("WORLD JOURNAL OF CASE REPORTS AND CLINICAL IMAGES")
            is None
        )
        assert extract_acronym_from_venue_text("Proceedings of the SIGIR Conference") == "SIGIR"
    finally:
        core_rankings.set_core_rankings_path(None)
        core_rankings.clear_core_rankings_cache()
