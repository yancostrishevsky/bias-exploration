"""Identifier normalisation helpers."""

from __future__ import annotations

import re
from urllib.parse import unquote
from typing import Mapping, Optional

DOI_REGEX = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
_DOI_PREFIX_RE = re.compile(r"^(?:doi:|https?://(?:dx\.)?doi\.org/)", re.IGNORECASE)


def normalise_doi(value: Optional[str]) -> Optional[str]:
    """Return a normalised DOI or *None* if the input is invalid."""

    if value is None:
        return None
    doi = unquote(str(value)).strip().lower()
    if not doi:
        return None
    doi = _DOI_PREFIX_RE.sub("", doi)
    doi = re.sub(r"\s+", "", doi)
    doi = doi.strip(" \t\r\n.;,")
    match = DOI_REGEX.search(doi)
    if not match:
        return None
    candidate = match.group(0).rstrip(".,;")
    if DOI_REGEX.fullmatch(candidate):
        return candidate
    return None


def doi_from_url(url: Optional[str]) -> Optional[str]:
    """Extract a DOI from a DOI resolver URL."""

    if not url:
        return None
    lowered = url.lower()
    for prefix in (
        "https://doi.org/",
        "http://doi.org/",
        "http://dx.doi.org/",
        "https://dx.doi.org/",
    ):
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
