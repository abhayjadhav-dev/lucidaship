"""Imputation helpers for schema validation and scoring."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

import pandas as pd
import numpy as np


def _safe_to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", format="mixed")


def extract_imputation_stats(
    df: pd.DataFrame,
    column_types: Dict[str, str],
    target_col: str | None = None,
) -> Dict[str, Dict[str, Any]]:
    stats: Dict[str, Dict[str, Any]] = {}
    for col, col_type in column_types.items():
        if col == target_col:
            continue
        if col_type in {"ignore", "id"}:
            continue

        col_data = df[col] if col in df.columns else pd.Series(dtype=object)

        if col_type == "numeric":
            median_value = float(col_data.median()) if col_data.notna().any() else 0.0
            mean_value = float(col_data.mean()) if col_data.notna().any() else 0.0
            stats[col] = {
                "type": "numeric",
                "median": median_value,
                "mean": mean_value,
                "min": float(col_data.min()) if col_data.notna().any() else 0.0,
                "max": float(col_data.max()) if col_data.notna().any() else 0.0,
                "default": median_value,
            }
        elif col_type in {"categorical", "text"}:
            mode_value = col_data.dropna().mode().iloc[0] if not col_data.dropna().empty else "Unknown"
            stats[col] = {
                "type": col_type,
                "mode": str(mode_value),
                "default": str(mode_value),
            }
        elif col_type == "binary":
            mode_value = col_data.dropna().mode().iloc[0] if not col_data.dropna().empty else 0
            stats[col] = {
                "type": "binary",
                "mode": int(mode_value) if str(mode_value).isdigit() else 0,
                "default": int(mode_value) if str(mode_value).isdigit() else 0,
            }
        elif col_type == "temporal":
            parsed = _safe_to_datetime(col_data)
            days_ago = (pd.Timestamp.today() - parsed).dt.days
            median_days = float(days_ago.dropna().median()) if days_ago.notna().any() else 30.0
            stats[col] = {
                "type": "temporal",
                "median_days": median_days,
                "default": None,
            }

    return stats


def impute_missing_columns(
    df: pd.DataFrame,
    expected_columns: Iterable[str],
    imputation_stats: Dict[str, Dict[str, Any]],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    missing_columns = [col for col in expected_columns if col not in df.columns]
    imputed_columns: Dict[str, Dict[str, Any]] = {}

    for col in missing_columns:
        stats = imputation_stats.get(col, {})
        col_type = stats.get("type")
        default_value = stats.get("default")

        if col_type == "numeric":
            default_value = float(default_value) if default_value is not None else 0.0
        elif col_type in {"categorical", "text"}:
            default_value = str(default_value) if default_value is not None else "Unknown"
        elif col_type == "binary":
            default_value = int(default_value) if default_value is not None else 0
        elif col_type == "temporal":
            default_value = pd.NaT
        else:
            default_value = None

        df[col] = default_value
        imputed_columns[col] = {
            "type": col_type or "unknown",
            "default": default_value if default_value is not pd.NaT else None,
        }

    report = {
        "missing_columns": missing_columns,
        "imputed_columns": imputed_columns,
        "imputed_count": int(len(imputed_columns)),
    }

    return df, report
