from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from ai_bias_search.normalize.records import (
    canonical_issn_selection,
    extract_openalex_publisher,
    normalize_country_code,
    normalize_record,
    normalize_records,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "fixture_name,expected_issn,expected_publisher",
    [
        (
            "openalex_record_with_issn_publisher.json",
            ["1234-5679", "8765-4321"],
            "ACM",
        ),
        (
            "semanticscholar_record_with_issn_publisher.json",
            ["1476-4687", "2049-3630"],
            "S2 Publisher",
        ),
        (
            "core_record_with_issn_publisher.json",
            ["1749-4885", "1749-4893"],
            "CORE Publisher",
        ),
    ],
)
def test_normalize_record_extracts_issn_and_publisher(
    fixture_name: str, expected_issn: list[str], expected_publisher: str
) -> None:
    raw = _fixture(fixture_name)
    normalized = normalize_record(raw, platform=str(raw["platform"]))

    assert normalized["issn_list"] == expected_issn
    assert normalized["publisher"] == expected_publisher
    assert normalized["issn_source"] is not None
    assert normalized["issn_provenance"] in {"raw", "enriched", "mixed"}
    assert normalized["journal_match"]["confidence"] == 1.0


def test_normalize_scopus_citations_missing_vs_zero_and_wrong_field() -> None:
    missing_raw = _fixture("scopus_record_missing_citations.json")
    zero_raw = _fixture("scopus_record_zero_citations.json")

    missing = normalize_record(missing_raw, platform="scopus")
    zero = normalize_record(zero_raw, platform="scopus")
    wrong_field = normalize_record(
        {
            "platform": "scopus",
            "title": "Wrong Field",
            "rank": 1,
            "citationCount": 11,
            "extra": {"scopus": {"raw": {"publishername": "Elsevier"}}},
        },
        platform="scopus",
    )

    assert missing["citations"] is None
    assert missing["metrics_quality"]["citations"] == "missing"

    assert zero["citations"] == 0
    assert zero["metrics_quality"]["citations"] == "ok"

    assert wrong_field["citations"] is None
    assert wrong_field["metrics_quality"]["citations"] == "missing"


def test_normalize_records_flags_suspicious_zero_citations() -> None:
    records = [
        {
            "platform": "scopus",
            "title": f"paper-{i}",
            "rank": i + 1,
            "year": 2018 + (i % 4),
            "citedby-count": 0,
            "extra": {"scopus": {"raw": {"prism:issn": "1234-5678"}}},
        }
        for i in range(30)
    ]

    normalized = normalize_records(records)
    assert all(item["citations"] == 0 for item in normalized)
    assert all(item["metrics_quality"]["citations"] == "suspicious" for item in normalized)


def test_normalize_records_does_not_flag_recent_zero_citations_as_suspicious() -> None:
    current_year = datetime.utcnow().year
    records = [
        {
            "platform": "scopus",
            "title": f"paper-{i}",
            "rank": i + 1,
            "year": current_year,
            "citedby-count": 0,
            "extra": {"scopus": {"raw": {"prism:issn": "1234-5678"}}},
        }
        for i in range(30)
    ]

    normalized = normalize_records(records)
    assert all(item["metrics_quality"]["citations"] == "ok" for item in normalized)


def test_openalex_publisher_prefers_host_org_name() -> None:
    normalized = normalize_record(
        {
            "platform": "openalex",
            "title": "Host org",
            "rank": 1,
            "extra": {
                "openalex": {
                    "primary_location": {
                        "source": {
                            "host_organization_name": "Elsevier BV",
                            "host_organization_lineage_names": ["Elsevier Group", "Elsevier BV"],
                            "display_name": "Journal Name",
                        }
                    },
                    "host_venue": {"publisher": "Host Publisher"},
                }
            },
        },
        platform="openalex",
    )
    assert normalized["publisher"] == "Elsevier BV"
    assert normalized["extra"]["openalex"]["publisher_provenance"] == "host_org_name"


def test_openalex_publisher_fallback_to_lineage_last() -> None:
    normalized = normalize_record(
        {
            "platform": "openalex",
            "title": "Lineage fallback",
            "rank": 1,
            "extra": {
                "openalex": {
                    "primary_location": {
                        "source": {
                            "host_organization_lineage_names": ["Elsevier Group", "Elsevier BV"],
                            "display_name": "Journal Name",
                        }
                    }
                }
            },
        },
        platform="openalex",
    )
    assert normalized["publisher"] == "Elsevier BV"
    assert normalized["extra"]["openalex"]["publisher_provenance"] == "host_org_lineage_last"


def test_openalex_publisher_missing_returns_none() -> None:
    normalized = normalize_record(
        {
            "platform": "openalex",
            "title": "No publisher",
            "rank": 1,
            "extra": {"openalex": {"primary_location": {"source": {}}, "host_venue": {}}},
        },
        platform="openalex",
    )
    assert normalized["publisher"] is None
    assert normalized["extra"]["openalex"]["publisher_provenance"] == "missing"


def test_extract_openalex_publisher_host_org_priority() -> None:
    publisher, provenance = extract_openalex_publisher(
        {
            "primary_location": {
                "source": {
                    "host_organization_name": "Elsevier BV",
                    "host_organization_lineage_names": ["Elsevier Group", "Elsevier BV"],
                    "display_name": "Journal Name",
                }
            },
            "host_venue": {"publisher": "Host Publisher"},
        }
    )
    assert publisher == "Elsevier BV"
    assert provenance == "host_org_name"


def test_extract_openalex_publisher_lineage_only() -> None:
    publisher, provenance = extract_openalex_publisher(
        {
            "primary_location": {
                "source": {
                    "host_organization_lineage_names": ["Big Group", "Lineage Publisher"],
                }
            }
        }
    )
    assert publisher == "Lineage Publisher"
    assert provenance == "host_org_lineage_last"


def test_extract_openalex_publisher_no_fields() -> None:
    publisher, provenance = extract_openalex_publisher(
        {
            "primary_location": {"source": {}},
            "host_venue": {},
        }
    )
    assert publisher is None
    assert provenance == "missing"


def test_normalize_core_citations_from_raw_citation_count() -> None:
    raw = _fixture("core_record_with_citation_count.json")
    normalized = normalize_record(raw, platform="core")
    assert normalized["citations"] == 7


def test_normalize_core_citations_structurally_unavailable_when_all_zero() -> None:
    current_year = datetime.utcnow().year
    records = [
        {
            "platform": "core",
            "title": f"core-paper-{idx}",
            "rank": idx + 1,
            "citations": 0,
            "year": None,
            "extra": {"core": {"citationCount": 0, "acceptedDate": f"{current_year - 5}-01-01"}},
        }
        for idx in range(30)
    ]
    normalized = normalize_records(records)
    assert all(item["citations"] is None for item in normalized)
    assert all(item["metrics_quality"]["citations"] == "structurally_unavailable" for item in normalized)


def test_semanticscholar_year_resolution_keeps_raw_and_marks_discrepancy() -> None:
    raw = {
        "platform": "semanticscholar",
        "title": "Year discrepancy record",
        "rank": 1,
        "year": 2012,
        "publication_year": 2012,
        "extra": {
            "semanticscholar": {
                "paperId": "s2-1",
                "year": 1897,
                "publicationDate": "1897-03-01",
                "journal": {"name": "Journal"},
            },
            "openalex": {"publication_year": 2012},
        },
    }
    normalized = normalize_record(raw, platform="semanticscholar")
    assert normalized["year"] == 2012
    assert normalized["year_raw"] == 1897
    assert normalized["year_enriched"] == 2012
    assert normalized["year_provenance"] == "mixed"


def test_semanticscholar_journal_title_prefers_publication_venue_name() -> None:
    raw = {
        "platform": "semanticscholar",
        "title": "Venue precedence",
        "rank": 1,
        "extra": {
            "semanticscholar": {
                "paperId": "s2-2",
                "publicationVenue": {"name": "Publication Venue Name"},
                "journal": {"name": "Journal Name"},
                "venue": "Venue String",
            }
        },
    }
    normalized = normalize_record(raw, platform="semanticscholar")
    assert normalized["journal_title"] == "Publication Venue Name"
    assert normalized["extra"]["provenance"]["journal_title"] == "publicationVenue.name"


def test_semanticscholar_issn_preference_is_deduped_and_ordered() -> None:
    raw = {
        "platform": "semanticscholar",
        "title": "ISSN precedence",
        "rank": 1,
        "extra": {
            "semanticscholar": {
                "paperId": "s2-3",
                "publicationVenue": {"issn": ["1234-567X", "1111-1111"]},
                "journal": {"issn": ["1111-1111", "2222-2222"]},
                "externalIds": {"ISSN": ["2222-2222", "3333-3333"]},
            }
        },
    }
    normalized = normalize_record(raw, platform="semanticscholar")
    assert normalized["issn"] == "1234-567X"
    assert normalized["eissn"] is None
    assert normalized["issn_list"] == ["1234-567X", "1111-1111", "2222-2222", "3333-3333"]
    issn_provenance = normalized["extra"]["provenance"]["issn"]
    assert issn_provenance["preferred_source"] == "semanticscholar"
    assert issn_provenance["sources_used"][0] == "semanticscholar"
    assert issn_provenance["count"] == 4


def test_year_selection_raw_only() -> None:
    normalized = normalize_record(
        {"platform": "core", "title": "Raw year", "rank": 1, "year": 2001},
        platform="core",
    )
    assert normalized["year"] == 2001
    assert normalized["year_raw"] == 2001
    assert normalized["year_enriched"] is None
    assert normalized["year_provenance"] == "raw"


def test_year_selection_enriched_only() -> None:
    normalized = normalize_record(
        {
            "platform": "semanticscholar",
            "title": "Enriched year",
            "rank": 1,
            "extra": {"openalex_enrich": {"publication_year": 2020}},
        },
        platform="semanticscholar",
    )
    assert normalized["year"] == 2020
    assert normalized["year_raw"] is None
    assert normalized["year_enriched"] == 2020
    assert normalized["year_provenance"] == "enriched"


def test_year_selection_both_same_is_mixed() -> None:
    normalized = normalize_record(
        {
            "platform": "semanticscholar",
            "title": "Same year",
            "rank": 1,
            "year": 2018,
            "extra": {"openalex_enrich": {"publication_year": 2018}},
        },
        platform="semanticscholar",
    )
    assert normalized["year"] == 2018
    assert normalized["year_raw"] == 2018
    assert normalized["year_enriched"] == 2018
    assert normalized["year_provenance"] == "mixed"


def test_year_selection_prefers_higher_trust_enriched_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("YEAR_TRUST_PRIORITY", "openalex,scopus,other,raw")
    normalized = normalize_record(
        {
            "platform": "semanticscholar",
            "title": "Trust order",
            "rank": 1,
            "year": 1999,
            "extra": {"openalex_enrich": {"publication_year": 2015}},
        },
        platform="semanticscholar",
    )
    assert normalized["year"] == 2015
    assert normalized["year_provenance"] == "mixed"


def test_canonical_issn_selection_honors_priority(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ISSN_SOURCE_PRIORITY", "openalex,scopus,raw")
    selected = canonical_issn_selection(
        {
            "platform": "semanticscholar",
            "issn": "1111-1111",
            "extra": {
                "openalex_enrich": {
                    "primary_location": {"source": {"issn": ["2222-2222"]}},
                    "host_venue": {},
                },
                "scopus": {"raw": {"prism:issn": "3333-3333"}},
            },
        },
        platform="semanticscholar",
    )
    assert selected["issn_list"][:3] == ["2222-2222", "3333-3333", "1111-1111"]
    assert selected["issn_source"] == "openalex"


def test_openalex_geo_extraction_from_authorship_institutions() -> None:
    raw = {
        "platform": "openalex",
        "title": "Geo from OpenAlex",
        "rank": 1,
        "extra": {
            "openalex": {
                "authorships": [
                    {"institutions": [{"country_code": "PL"}, {"country_code": "DE"}]},
                    {"institutions": [{"country_code": "PL"}]},
                ]
            }
        },
    }
    normalized = normalize_record(raw, platform="openalex")
    assert normalized["countries"] == ["DE", "PL"]
    assert normalized["country_primary"] == "MULTI"
    assert normalized["affiliation_countries"] == ["DE", "PL"]
    assert normalized["country_count"] == 2
    assert normalized["country_dominant"] == "MULTI"
    assert normalized["country_provenance"] == "openalex.authorships.institutions.country_code"


def test_scopus_geo_extraction_from_affiliation_country() -> None:
    raw = {
        "platform": "scopus",
        "title": "Geo from Scopus",
        "rank": 1,
        "scopus": {
            "abstract": {
                "response": {
                    "affiliation": [
                        {"affiliation-country": "Poland"},
                        {"affiliation-country": "Germany"},
                    ]
                }
            }
        },
    }
    normalized = normalize_record(raw, platform="scopus")
    assert normalized["countries"] == ["DE", "PL"]
    assert normalized["country_primary"] == "MULTI"
    assert normalized["affiliation_countries"] == ["DE", "PL"]
    assert normalized["country_count"] == 2
    assert normalized["country_dominant"] == "MULTI"
    assert normalized["country_provenance"] == "scopus.abstract.affiliation-country"


def test_scopus_geo_parses_stringified_array_without_combo_strings() -> None:
    raw = {
        "platform": "scopus",
        "title": "Stringified country list",
        "rank": 1,
        "scopus": {
            "abstract": {
                "affiliation_countries": "['Australia' 'China']",
            }
        },
    }
    normalized = normalize_record(raw, platform="scopus")
    assert normalized["countries"] == ["AU", "CN"]
    assert normalized["country_primary"] == "MULTI"
    assert normalized["country_count"] == 2
    assert all(code in {"AU", "CN"} for code in normalized["countries"])


def test_normalize_country_code_maps_common_names() -> None:
    assert normalize_country_code("Australia") == "AU"
    assert normalize_country_code("  united states of america  ") == "US"
    assert normalize_country_code("PL") == "PL"
    assert normalize_country_code("unknown-country") is None
