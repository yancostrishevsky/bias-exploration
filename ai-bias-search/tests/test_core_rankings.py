from __future__ import annotations

from pathlib import Path

from ai_bias_search.utils import core_rankings


def _write_core_csv(path: Path, content: str) -> None:
    path.write_text(content.strip() + "\n", encoding="utf-8")


def test_core_rank_lookup_with_header(tmp_path: Path) -> None:
    csv_content = (
        "id,title,acronym,source,rank,core,field,field2\n"
        "1,ACM Conference on Research and Development in Information Retrieval (SIGIR),"
        "SIGIR,CORE2023,A*,Yes,4605,,\n"
        "2,ACM International Conference on Knowledge Discovery and Data Mining,"
        "KDD,CORE2023,A*,Yes,4605,,\n"
        "3,ACM Conference on Computer and Communications Security,CCS,CORE2023,A*,Yes,4604,,\n"
        "4,Some Unranked Conference,SUC,CORE2023,Unranked,Yes,0000,,\n"
    )
    core_path = tmp_path / "core.csv"
    _write_core_csv(core_path, csv_content)

    core_rankings.set_core_rankings_path(core_path)
    core_rankings.clear_core_rankings_cache()
    try:
        assert (
            core_rankings.lookup_core_rank(
                "ACM Conference on Research and Development in Information Retrieval"
            )
            == "A*"
        )
        assert core_rankings.lookup_core_rank(None, "SIGIR") == "A*"
        assert (
            core_rankings.lookup_core_rank(
                "ACM International Conference on Knowledge Discovery and Data Mining"
            )
            == "A*"
        )
        assert core_rankings.lookup_core_rank("Some Unranked Conference") is None
    finally:
        core_rankings.set_core_rankings_path(None)
        core_rankings.clear_core_rankings_cache()


def test_core_rank_lookup_without_header(tmp_path: Path) -> None:
    csv_content = (
        "1,ACM Conference on Computer and Communications Security,CCS,CORE2023,A*,Yes,4604,,\n"
        "2,International Conference on Information Retrieval,IR,CORE2023,B,Yes,4605,,\n"
    )
    core_path = tmp_path / "core_no_header.csv"
    _write_core_csv(core_path, csv_content)

    core_rankings.set_core_rankings_path(core_path)
    core_rankings.clear_core_rankings_cache()
    try:
        assert core_rankings.lookup_core_rank(None, "ccs") == "A*"
        assert (
            core_rankings.lookup_core_rank("International Conference on Information Retrieval")
            == "B"
        )
    finally:
        core_rankings.set_core_rankings_path(None)
        core_rankings.clear_core_rankings_cache()
