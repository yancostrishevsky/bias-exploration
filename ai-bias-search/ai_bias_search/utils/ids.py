"""Identifier normalisation helpers."""

from __future__ import annotations

import re
from typing import Mapping, Optional

DOI_REGEX = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)


def normalise_doi(value: Optional[str]) -> Optional[str]:
    """Return a normalised DOI or *None* if the input is invalid."""

    if not value:
        return None
    doi = value.strip().lower()
    if doi.startswith("doi:"):
        doi = doi[4:]
    if DOI_REGEX.fullmatch(doi):
        return doi
    return None


def doi_from_url(url: Optional[str]) -> Optional[str]:
    """Extract a DOI from a DOI resolver URL."""

    if not url:
        return None
    lowered = url.lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "http://dx.doi.org/", "https://dx.doi.org/"):
        if lowered.startswith(prefix):
            return normalise_doi(url[len(prefix) :])
    return None


def best_identifier(record: Mapping[str, object]) -> Optional[str]:
    """Return the best available identifier for a record."""

    doi = normalise_doi(record.get("doi"))  # type: ignore[arg-type]
    if doi:
        return doi
    doi = doi_from_url(record.get("url"))  # type: ignore[arg-type]
    if doi:
        return doi
    raw_id = record.get("raw_id")
    if isinstance(raw_id, str) and raw_id:
        return raw_id
    return None
