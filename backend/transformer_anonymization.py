"""Deterministic, non-reversible transformer aliases for external outputs."""

from __future__ import annotations

import hashlib
import re
from typing import Iterable


def build_transformer_aliases(transformer_ids: Iterable[object]) -> dict[str, str]:
    """Return stable aliases without exposing the source asset identifier."""
    unique_ids = sorted({str(value) for value in transformer_ids if value is not None})
    aliases: dict[str, str] = {}
    used: set[str] = set()
    for source_id in unique_ids:
        if re.fullmatch(r"TR-[0-9A-F]{10,64}", source_id, flags=re.IGNORECASE):
            aliases[source_id] = source_id.upper()
            used.add(source_id.upper())
            continue
        digest = hashlib.sha256(source_id.encode("utf-8")).hexdigest().upper()
        length = 10
        alias = f"TR-{digest[:length]}"
        while alias in used:
            length += 2
            alias = f"TR-{digest[:length]}"
        aliases[source_id] = alias
        used.add(alias)
    return aliases


def anonymize_transformer_column(frame, column: str = "transformer_id"):
    """Copy a dataframe and replace one transformer identifier column with aliases."""
    aliases = build_transformer_aliases(frame[column].dropna().tolist())
    result = frame.copy()
    result[column] = result[column].map(lambda value: aliases.get(str(value), value))
    return result