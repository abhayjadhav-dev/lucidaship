"""Type coercion helpers for preprocessing scoring data."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

NULL_LIKE = {"", "nan", "none", "null", "n/a", "na"}
POSITIVE_TOKENS = {"1", "yes", "true", "y", "t", "on", "won", "converted", "success"}
NEGATIVE_TOKENS = {"0", "no", "false", "n", "f", "off", "lost"}


def _normalize_binary_token(value: Any) -> int | float | None:
    if pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return 1 if bool(value) else 0
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return None
        rounded = round(float(value))
        if abs(float(value) - rounded) < 1e-9:
            return int(rounded)
        return None
    token = str(value).strip().lower()
    if token in NULL_LIKE:
        return None
    if token in POSITIVE_TOKENS:
        return 1
    if token in NEGATIVE_TOKENS:
        return 0
    return None


def _stringify_or_none(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(value).strip()


def coerce_series_to_expected_type(
    series: pd.Series,
    expected_type: str,
) -> Tuple[pd.Series, Dict[str, Any]]:
    """Coerce a series into an expected semantic type.

    Returns the coerced series plus a small diagnostics dict.
    """
    original_non_null = int(series.notna().sum())

    if expected_type in {"numeric"}:
        coerced = pd.to_numeric(series, errors="coerce")
    elif expected_type in {"temporal"}:
        coerced = pd.to_datetime(series, errors="coerce", format="mixed")
    elif expected_type in {"binary"}:
        coerced = series.map(_normalize_binary_token)
    else:
        coerced = series.map(_stringify_or_none)

    coerced_non_null = int(pd.Series(coerced).notna().sum())
    report = {
        "expected_type": expected_type,
        "original_non_null": original_non_null,
        "coerced_non_null": coerced_non_null,
        "null_after": int(len(series) - coerced_non_null),
        "coercion_ratio": float(coerced_non_null / max(1, original_non_null)),
    }

    return pd.Series(coerced, index=series.index), report
