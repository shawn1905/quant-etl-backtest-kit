"""Generic futures contract parsing and dominant-contract selection helpers."""

from __future__ import annotations

import math
import re
from typing import Any, Iterable, Mapping

_CODE = re.compile(r"^([A-Za-z]+)\.([A-Za-z]+)(\d+)$")


def split_code(code: Any) -> tuple[str | None, str | None]:
    """Extract (exchange, lowercase_product) from 'EXCH.prod2610' style codes."""
    match = _CODE.match(str(code))
    if not match:
        return None, None
    return match.group(1), match.group(2).lower()


def _num(value: Any) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def dominant_by_day(rows: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    """Select dominant contract per day by max open interest, tie-breaking by vol then code."""
    best: dict[str, tuple[tuple[float, float, str], str]] = {}
    for row in rows:
        oi, vol = _num(row.get("oi")), _num(row.get("vol"))
        if oi is None:
            continue
        code = str(row["code"])
        day = str(row["date"])
        key = (oi, vol or 0.0, code)
        if day not in best or key > best[day][0]:
            best[day] = (key, code)
    return {d: v[1] for d, v in best.items()}
