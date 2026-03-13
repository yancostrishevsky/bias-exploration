"""Journal Impact Factor (JIF/JCR) plots for the HTML report."""

from __future__ import annotations

import base64
import io
from typing import Any, Dict, Mapping, Tuple

import pandas as pd

from ai_bias_search.utils.logging import configure_logging

LOGGER = configure_logging()

MATCH_TYPE_ORDER: Tuple[str, ...] = (
    "issn_exact",
    "title_exact",
    "title_fuzzy",
    "ambiguous",
    "none",
)

QUARTILE_ORDER: Tuple[str, ...] = ("Q1", "Q2", "Q3", "Q4", "Unknown")


def build_jif_context(frame: pd.DataFrame) -> Dict[str, object]:
    """Build template context for the Journal Impact Factor (JIF/JCR) section.

    The returned dictionary is safe to unpack into the main Jinja context.
    """

    total_records = int(len(frame))
    if total_records == 0:
        return _disabled_context("No records available.")

    if "impact_factor" not in frame.columns:
        return _disabled_context("JIF/JCR columns not found in the enriched dataset.")

    impact_factor = coerce_positive_numeric(frame["impact_factor"])
    matched_mask = impact_factor.notna()
    matched_count = int(matched_mask.sum())
    coverage_pct = float(matched_count / total_records * 100.0) if total_records else 0.0

    if matched_count == 0:
        return _disabled_context(
            "No impact factor values available (enrichment disabled or no matches).",
            coverage_pct=coverage_pct,
        )

    jif_match_plot_png = None
    match_col = _choose_match_type_column(frame)
    if match_col:
        counts = prepare_match_type_counts(frame, match_col=match_col)
        if not counts.empty:
            jif_match_plot_png = render_match_quality_plot(frame, counts, match_col=match_col)

    jif_distribution_plot_png = render_impact_factor_distribution_plot(frame, impact_factor)

    jif_quartile_plot_png = None
    if "jcr_quartile" in frame.columns:
        raw_quartiles = clean_text(frame["jcr_quartile"])
        if not raw_quartiles.dropna().empty:
            quartiles = bucket_quartiles(raw_quartiles)
            jif_quartile_plot_png = render_quartile_distribution_plot(frame, quartiles)

    jif_publisher_plot_png = None
    jif_publisher_hhi_text = None
    if "jcr_publisher" in frame.columns:
        pubs = clean_text(frame["jcr_publisher"])
        pubs = pubs.where(matched_mask)
        pubs = pubs.dropna()
        if not pubs.empty:
            top_counts, hhi, hhi_by_platform = prepare_publisher_concentration(pubs, frame)
            if not top_counts.empty:
                jif_publisher_plot_png = render_top_publishers_plot(top_counts)
            if hhi is not None:
                jif_publisher_hhi_text = _format_hhi_text(hhi, hhi_by_platform)

    jif_rank_plot_png = None
    jif_rank_corr_text = None
    if "rank" in frame.columns:
        rank_png, corr_text = maybe_render_rank_vs_impact_factor(
            frame, impact_factor=impact_factor, min_points=30
        )
        jif_rank_plot_png = rank_png
        jif_rank_corr_text = corr_text

    return {
        "jif_enabled": True,
        "jif_message": None,
        "jif_coverage_pct": coverage_pct,
        "jif_match_plot_png": jif_match_plot_png,
        "jif_distribution_plot_png": jif_distribution_plot_png,
        "jif_quartile_plot_png": jif_quartile_plot_png,
        "jif_publisher_plot_png": jif_publisher_plot_png,
        "jif_publisher_hhi_text": jif_publisher_hhi_text,
        "jif_rank_plot_png": jif_rank_plot_png,
        "jif_rank_corr_text": jif_rank_corr_text,
    }


def coerce_positive_numeric(series: pd.Series) -> pd.Series:
    """Return a float series with non-positive values coerced to NA."""

    values = pd.to_numeric(series, errors="coerce").astype(float)
    values = values.where(values > 0)
    return values


def cap_values(series: pd.Series, *, upper_quantile: float = 0.99) -> tuple[pd.Series, float | None]:
    """Cap a numeric series at a robust upper percentile."""

    values = pd.to_numeric(series, errors="coerce").astype(float).dropna()
    if values.empty:
        return series, None
    cap = float(values.quantile(upper_quantile))
    capped = pd.to_numeric(series, errors="coerce").astype(float).clip(upper=cap)
    return capped, cap


def clean_text(series: pd.Series) -> pd.Series:
    """Normalize a string-like series, turning empty strings into NA."""

    text = series.astype("string")
    text = text.str.replace("\u00a0", " ", regex=False).str.strip()
    text = text.replace("", pd.NA)
    return text


def bucket_quartiles(series: pd.Series) -> pd.Series:
    """Map raw quartile values to Q1..Q4 and Unknown."""

    text = clean_text(series).str.upper()
    mapped = text.where(text.isin(["Q1", "Q2", "Q3", "Q4"]), other="Unknown")
    mapped = mapped.fillna("Unknown")
    return mapped


def prepare_match_type_counts(frame: pd.DataFrame, *, match_col: str) -> pd.Series:
    """Return counts of match types (including missing as 'none')."""

    raw = clean_text(frame[match_col]).str.lower()
    match = raw.fillna("none")
    match = match.where(match.isin(MATCH_TYPE_ORDER), other="none")
    return match.value_counts().reindex(list(MATCH_TYPE_ORDER), fill_value=0)


def prepare_publisher_concentration(
    publishers: pd.Series,
    frame: pd.DataFrame,
    *,
    platform_col: str = "platform",
) -> tuple[pd.Series, float | None, dict[str, float] | None]:
    """Compute top publisher counts and HHI for the provided publisher series."""

    counts = publishers.value_counts()
    top_counts = counts.head(10)
    total = int(counts.sum())
    if total <= 0:
        return top_counts, None, None
    hhi = float(((counts / total) ** 2).sum())

    hhi_by_platform: dict[str, float] | None = None
    if platform_col in frame.columns and publishers.index.isin(frame.index).all():
        joined = (
            pd.DataFrame(
                {
                    "publisher": publishers,
                    "platform": clean_text(frame.loc[publishers.index, platform_col]),
                }
            )
            .dropna()
            .astype({"publisher": "string", "platform": "string"})
        )
        if not joined.empty:
            hhi_by_platform = {}
            for platform, subset in joined.groupby("platform")["publisher"]:
                subset_counts = subset.value_counts()
                subset_total = int(subset_counts.sum())
                if subset_total <= 0:
                    continue
                hhi_by_platform[str(platform)] = float(((subset_counts / subset_total) ** 2).sum())

    return top_counts, hhi, hhi_by_platform


def maybe_render_rank_vs_impact_factor(
    frame: pd.DataFrame,
    *,
    impact_factor: pd.Series,
    min_points: int = 30,
    platform_col: str = "platform",
) -> tuple[str | None, str | None]:
    """Optionally render rank vs impact factor plot and Spearman correlation text."""

    ranks = pd.to_numeric(frame["rank"], errors="coerce").astype(float)
    data = pd.DataFrame({"rank": ranks, "impact_factor": impact_factor}).dropna()
    if data.shape[0] < min_points:
        return None, None

    rho_overall = spearman_rho(data["rank"], data["impact_factor"])
    rho_text_parts = []
    if rho_overall is not None:
        rho_text_parts.append(f"overall={rho_overall:+.3f}")

    if platform_col in frame.columns:
        platforms = clean_text(frame.loc[data.index, platform_col])
        data = data.assign(platform=platforms).dropna(subset=["platform"])
        by_platform_parts = []
        for platform, subset in data.groupby("platform"):
            if subset.shape[0] < min_points:
                continue
            rho = spearman_rho(subset["rank"], subset["impact_factor"])
            if rho is None:
                continue
            by_platform_parts.append(f"{platform}={rho:+.3f}")
        if by_platform_parts:
            rho_text_parts.append("by platform: " + ", ".join(by_platform_parts))

    corr_text = None
    if rho_text_parts:
        corr_text = "Spearman ρ(rank, impact_factor): " + " | ".join(rho_text_parts)

    plot_png = render_rank_vs_impact_factor_plot(frame.loc[data.index], impact_factor.loc[data.index])
    return plot_png, corr_text


def spearman_rho(x: pd.Series, y: pd.Series) -> float | None:
    """Compute Spearman correlation without SciPy."""

    if x.empty or y.empty:
        return None
    xr = x.rank(method="average")
    yr = y.rank(method="average")
    rho = xr.corr(yr, method="pearson")
    if rho is None or pd.isna(rho):
        return None
    return float(rho)


def render_match_quality_plot(frame: pd.DataFrame, counts: pd.Series, *, match_col: str) -> str:
    """Render match type shares, faceted by platform when possible."""

    platform_col = "platform"
    by_platform = platform_col in frame.columns and frame[platform_col].dropna().nunique() > 1
    match = clean_text(frame[match_col]).str.lower().fillna("none")
    match = match.where(match.isin(MATCH_TYPE_ORDER), other="none")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    if by_platform:
        subset = pd.DataFrame({platform_col: clean_text(frame[platform_col]), "match": match}).dropna(
            subset=[platform_col]
        )
        ctab = pd.crosstab(subset[platform_col], subset["match"]).reindex(
            columns=list(MATCH_TYPE_ORDER), fill_value=0
        )
        platform_order = ctab.sum(axis=1).sort_values(ascending=False).index
        shares = ctab.div(ctab.sum(axis=1).replace(0, pd.NA), axis=0).fillna(0.0)
        ordered_cols = [col for col in list(MATCH_TYPE_ORDER) if shares[col].sum() > 0]
        shares = shares.reindex(columns=ordered_cols)
        shares = shares.loc[platform_order]
        shares.plot(kind="bar", stacked=True, ax=ax)
        ax.set_xlabel("Platform")
        ax.set_ylabel("Share")
        ax.set_title("Impact factor match quality by platform")
        ax.legend(title="Match type", bbox_to_anchor=(1.04, 1), loc="upper left")
        fig.tight_layout()
        return _fig_to_base64_png(fig)

    share = counts / float(counts.sum()) if counts.sum() else counts
    share = share[share > 0]
    share.plot(kind="bar", ax=ax)
    ax.set_xlabel("Match type")
    ax.set_ylabel("Share")
    ax.set_ylim(0, 1)
    ax.set_title("Impact factor match quality")
    fig.tight_layout()
    return _fig_to_base64_png(fig)


def render_impact_factor_distribution_plot(frame: pd.DataFrame, impact_factor: pd.Series) -> str:
    """Render impact factor distribution, by platform when possible."""

    platform_col = "platform"
    by_platform = platform_col in frame.columns and frame[platform_col].dropna().nunique() > 1

    capped, cap = cap_values(impact_factor, upper_quantile=0.99)
    capped = capped.dropna()

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    cap_label = f"(capped at p99={cap:.2f})" if cap is not None else ""

    if by_platform:
        platforms = clean_text(frame.loc[capped.index, platform_col])
        data = pd.DataFrame({"platform": platforms, "impact_factor": capped}).dropna(subset=["platform"])
        platform_order = data.groupby("platform")["impact_factor"].size().sort_values(ascending=False).index
        groups = []
        labels = []
        for platform in platform_order:
            subset = data.loc[data["platform"] == platform, "impact_factor"]
            if subset.empty:
                continue
            groups.append(subset.values)
            labels.append(str(platform))
        if groups:
            ax.boxplot(groups, labels=labels, showfliers=False)  # type: ignore[call-arg]
            ax.set_xlabel("Platform")
            ax.set_ylabel(f"Impact factor {cap_label}".strip())
            ax.set_title("Impact factor distribution by platform")
            fig.tight_layout()
            return _fig_to_base64_png(fig)

    ax.hist(capped.values, bins=min(30, max(10, int(capped.nunique()))), color="#4c72b0", alpha=0.85)
    ax.set_xlabel(f"Impact factor {cap_label}".strip())
    ax.set_ylabel("Count")
    ax.set_title("Impact factor distribution")
    fig.tight_layout()
    return _fig_to_base64_png(fig)


def render_quartile_distribution_plot(frame: pd.DataFrame, quartiles: pd.Series) -> str:
    platform_col = "platform"
    by_platform = platform_col in frame.columns and frame[platform_col].dropna().nunique() > 1

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))

    if by_platform:
        platforms = clean_text(frame.loc[quartiles.index, platform_col])
        subset = (
            pd.DataFrame({"platform": platforms, "quartile": quartiles})
            .dropna(subset=["platform"])
            .astype({"platform": "string", "quartile": "string"})
        )
        ctab = pd.crosstab(subset["platform"], subset["quartile"]).reindex(
            columns=list(QUARTILE_ORDER), fill_value=0
        )
        platform_order = ctab.sum(axis=1).sort_values(ascending=False).index
        shares = ctab.div(ctab.sum(axis=1).replace(0, pd.NA), axis=0).fillna(0.0)
        shares = shares.loc[platform_order]
        shares.plot(kind="bar", stacked=True, ax=ax)
        ax.set_xlabel("Platform")
        ax.set_ylabel("Share")
        ax.set_title("JCR quartile distribution by platform")
        ax.legend(title="Quartile", bbox_to_anchor=(1.04, 1), loc="upper left")
        fig.tight_layout()
        return _fig_to_base64_png(fig)

    counts = quartiles.value_counts().reindex(list(QUARTILE_ORDER), fill_value=0)
    shares = counts / float(counts.sum()) if counts.sum() else counts
    shares.plot(kind="bar", ax=ax)
    ax.set_xlabel("Quartile")
    ax.set_ylabel("Share")
    ax.set_ylim(0, 1)
    ax.set_title("JCR quartile distribution")
    fig.tight_layout()
    return _fig_to_base64_png(fig)


def render_top_publishers_plot(counts: pd.Series) -> str:
    import matplotlib.pyplot as plt

    counts = counts.sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh([str(x) for x in counts.index], counts.values, color="#55a868", alpha=0.9)
    ax.set_xlabel("Count")
    ax.set_ylabel("Publisher")
    ax.set_title("Top JCR publishers (JIF-matched subset)")
    fig.tight_layout()
    return _fig_to_base64_png(fig)


def render_rank_vs_impact_factor_plot(frame: pd.DataFrame, impact_factor: pd.Series) -> str:
    import matplotlib.pyplot as plt

    capped, cap = cap_values(impact_factor, upper_quantile=0.99)
    cap_label = f"(capped at p99={cap:.2f})" if cap is not None else ""

    fig, ax = plt.subplots(figsize=(6, 4))
    if "platform" in frame.columns:
        platforms = clean_text(frame["platform"]).fillna("Unknown")
        df = pd.DataFrame({"rank": frame["rank"], "impact_factor": capped, "platform": platforms}).dropna()
        for platform, subset in df.groupby("platform"):
            ax.scatter(
                subset["rank"],
                subset["impact_factor"],
                alpha=0.55,
                s=14,
                label=str(platform),
            )
        ax.legend(title="Platform", bbox_to_anchor=(1.04, 1), loc="upper left")
    else:
        ax.scatter(frame["rank"], capped, alpha=0.55, s=14)
    ax.set_xlabel("Rank (lower is better)")
    ax.set_ylabel(f"Impact factor {cap_label}".strip())
    ax.set_title("Rank vs impact factor")
    fig.tight_layout()
    return _fig_to_base64_png(fig)


def _choose_match_type_column(frame: pd.DataFrame) -> str | None:
    if "impact_factor_match" in frame.columns:
        return "impact_factor_match"
    if "jcr_match_type" in frame.columns:
        return "jcr_match_type"
    return None


def _disabled_context(message: str, *, coverage_pct: float = 0.0) -> Dict[str, object]:
    return {
        "jif_enabled": False,
        "jif_message": message,
        "jif_coverage_pct": float(coverage_pct),
        "jif_match_plot_png": None,
        "jif_distribution_plot_png": None,
        "jif_quartile_plot_png": None,
        "jif_publisher_plot_png": None,
        "jif_publisher_hhi_text": None,
        "jif_rank_plot_png": None,
        "jif_rank_corr_text": None,
    }


def _fig_to_base64_png(fig: Any) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    try:
        import matplotlib.pyplot as plt

        plt.close(fig)
    except Exception:  # pragma: no cover - best-effort cleanup
        pass
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _format_hhi_text(hhi: float, by_platform: Mapping[str, float] | None) -> str:
    parts = [f"Publisher HHI (JIF-matched subset): {hhi:.3f}"]
    if by_platform:
        ordered = ", ".join(f"{k}={v:.3f}" for k, v in sorted(by_platform.items()))
        parts.append(f"by platform: {ordered}")
    return " | ".join(parts)
