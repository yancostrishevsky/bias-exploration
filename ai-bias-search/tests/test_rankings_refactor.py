from __future__ import annotations

from pathlib import Path

from ai_bias_search.rankings.base import (
    RankingConfig,
    RankingEntry,
    TitleNormalizationConfig,
    normalize_issn,
    normalize_title,
    validate_issn_checksum,
)
from ai_bias_search.rankings.match import MatchingEngine
from ai_bias_search.rankings.registry import get_provider, list_rankings


def test_validate_issn_checksum() -> None:
    assert validate_issn_checksum("1234-5679")
    assert not validate_issn_checksum("1234-5678")

    assert validate_issn_checksum("0000-006X")
    assert not validate_issn_checksum("0000-0060")


def test_normalize_issn_optional_checksum() -> None:
    assert normalize_issn("12345678") == "1234-5678"
    assert normalize_issn("12345678", validate_checksum=True) is None
    assert normalize_issn("12345679", validate_checksum=True) == "1234-5679"


def test_match_ordering_issn_beats_title_fuzzy() -> None:
    cfg = RankingConfig(
        id="test",
        label="Test",
        dataset_path=Path("dummy.csv"),
        format="csv",
        fields={"title": 0, "rank_value": 1},
        normalization=TitleNormalizationConfig(strip_parens=False),
        allow_fuzzy=True,
        fuzzy_threshold=0.80,
        validate_issn_checksum=False,
    )

    entry_fuzzy = RankingEntry(
        venue_key="fuzzy",
        title="Journal of Testing",
        title_norm=normalize_title("Journal of Testing", cfg.normalization),
        issn_print=None,
        issn_online=None,
        issn_l=None,
        rank_value=1.0,
        rank_year=2024,
        source_id=cfg.id,
        extra={},
    )
    entry_issn = RankingEntry(
        venue_key="issn",
        title="Unrelated Journal",
        title_norm=normalize_title("Unrelated Journal", cfg.normalization),
        issn_print="1234-5679",
        issn_online=None,
        issn_l=None,
        rank_value=7.0,
        rank_year=2024,
        source_id=cfg.id,
        extra={},
    )

    engine = MatchingEngine.build(cfg, [entry_fuzzy, entry_issn])
    result = engine.match("Journal of Testin", ["1234-5679"])
    assert result.method == "issn_exact"
    assert result.rank_value == 7.0


def test_registry_loads_yaml_sources() -> None:
    ids = list_rankings(include_disabled=True)
    assert "core" in ids
    assert "jif" in ids
    assert "ministerialny" in ids


def test_core_provider_smoke_match() -> None:
    provider = get_provider("core")
    provider.reset()
    result = provider.match("ACM Conference on Economics and Computation", None)
    assert result.method in {"title_exact", "title_fuzzy"}
    assert result.rank_value == "A*"

