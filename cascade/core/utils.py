"""
Utility helpers shared across cascade experimental scripts.

These functions centralise common parsing, normalisation, and slugification
logic that previously lived in multiple CLI runners and analysis tools.
"""

from __future__ import annotations

import re
from typing import Iterable, Mapping, MutableMapping, Optional, Sequence, Union

_LABELS = ("A", "B", "C", "D", "E")
_DEFAULT_LEVELS = {"high", "medium", "low"}
_DEFAULT_LEVELS_WITH_V2 = {"high", "medium", "low", "high_v2", "medium_v2", "low_v2"}
_DEFAULT_LEVELS_WITH_V3 = {
    "high", "medium", "low",
    "high_v2", "medium_v2", "low_v2",
    "high_v3", "medium_v3", "low_v3"
}
_SLUG_RE = re.compile(r"[^a-z0-9_-]+", re.I)


def slugify(value: object) -> str:
    """Return a filesystem-safe slug."""
    text = str(value).strip().lower().replace(" ", "-")
    text = _SLUG_RE.sub("-", text)
    return re.sub(r"-{2,}", "-", text).strip("-")


def uniform_belief(labels: Sequence[str] = _LABELS) -> dict[str, float]:
    """Return a uniform categorical belief distribution."""
    n = len(labels)
    value = 1.0 / n if n else 0.0
    return {label: value for label in labels}


def normalize_belief(
    belief: Mapping[str, float] | MutableMapping[str, float],
    labels: Sequence[str] = _LABELS,
) -> dict[str, float]:
    """
    Normalize a belief dictionary ensuring values sum to 1.0 and are non-negative.

    Falls back to a uniform distribution if the supplied belief is empty or sums
    to zero/negative values.
    """
    vec = [float(belief.get(label, 0.0)) for label in labels]
    total = sum(max(v, 0.0) for v in vec)
    if total <= 0:
        return uniform_belief(labels)
    return {label: float(max(v, 0.0) / total) for label, v in zip(labels, vec)}


def _as_iterable(raw: Union[str, Sequence, None]) -> Optional[list[str]]:
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        return [str(item).strip() for item in raw if str(item).strip() != ""]
    text = str(raw).strip()
    if not text:
        return None
    return [part.strip() for part in text.split(",") if part.strip()]


def parse_float_list(raw: Union[str, Sequence[float], None]) -> Optional[list[float]]:
    """Parse a comma separated string or iterable into a list of floats."""
    items = _as_iterable(raw)
    if items is None:
        return None
    try:
        return [float(item) for item in items]
    except Exception as exc:  # pragma: no cover - defensive guard
        raise SystemExit(
            "ERROR: Failed to parse float list (check commas and numbers)."
        ) from exc


def parse_int_list(raw: Union[str, Sequence[int], None]) -> Optional[list[int]]:
    """Parse a comma separated string or iterable into a list of integers."""
    items = _as_iterable(raw)
    if items is None:
        return None
    try:
        return [int(item) for item in items]
    except Exception as exc:  # pragma: no cover - defensive guard
        raise SystemExit(
            "ERROR: Failed to parse int list (check commas and integers)."
        ) from exc


def parse_level_list(
    raw: Union[str, Sequence[str], None],
    valid_levels: Iterable[str] = _DEFAULT_LEVELS_WITH_V3,
) -> Optional[list[str]]:
    """
    Parse comma separated trait levels (high/medium/low by default, plus _v2 and _v3 variants).

    Returns lowercase strings; raises SystemExit on invalid tokens to mirror
    argparse error semantics.

    Supports v1 levels (high, medium, low), v2 levels (high_v2, medium_v2, low_v2),
    and v3 levels (high_v3, medium_v3, low_v3).
    """
    items = _as_iterable(raw)
    if items is None:
        return None
    levels = [item.lower() for item in items]
    missing = set(levels) - set(valid_levels)
    if missing:
        raise SystemExit(
            f"ERROR: Invalid levels {sorted(missing)}. "
            f"Valid options: {sorted(valid_levels)}."
        )
    return levels
