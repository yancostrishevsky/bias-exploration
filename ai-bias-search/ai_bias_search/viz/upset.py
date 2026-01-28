"""Optional UpSet plot helper."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

from ai_bias_search.utils.logging import configure_logging

LOGGER = configure_logging()


def plot_upset(records: Iterable[Mapping[str, object]], output: Path) -> None:
    """Generate an UpSet plot showing overlaps between platforms."""

    try:  # pragma: no cover - optional dependency
        from upsetplot import UpSet  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency
        LOGGER.warning("upsetplot not installed; skipping UpSet visualisation")
        return

    frame = pd.DataFrame(records)
    if {"query_id", "platform", "doi"} - set(frame.columns):
        LOGGER.warning("Insufficient data to plot UpSet")
        return

    pivot = frame.pivot_table(
        index="doi", columns="platform", values="query_id", aggfunc="count", fill_value=0
    )
    binary = pivot.astype(bool)
    upset = UpSet(binary, sort_by="cardinality")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig = upset.plot()
    fig.figure.savefig(output)
    LOGGER.info("Saved UpSet plot to %s", output)
