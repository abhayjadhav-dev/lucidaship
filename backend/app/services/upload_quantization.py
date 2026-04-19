"""
Upload-time numeric compression utilities.

The first rollout is intentionally conservative:
- only eligible numeric columns are quantized,
- structural columns stay lossless,
- shadow mode preserves the current full-precision execution path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


ID_LIKE_KEYWORDS = ("id", "email", "key", "uuid", "contact", "prospect", "phone", "account")


@dataclass
class IngestedDatasetAsset:
    name: str
    raw_df: pd.DataFrame
    protected_df: pd.DataFrame
    dequantized_df: pd.DataFrame
    numeric_block: Optional[np.ndarray]
    quantized_block: Optional[np.ndarray]
    quantizer_metadata: Dict[str, object] = field(default_factory=dict)
    mode: str = "full"
    diagnostics: Dict[str, object] = field(default_factory=dict)


def _is_datetime_like(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if series.dtype != "object":
        return False
    non_null = series.dropna()
    if non_null.empty:
        return False
    sample = non_null.astype(str).head(100)
    parsed = pd.to_datetime(sample, errors="coerce")
    return bool(parsed.notna().mean() >= 0.8)


def _is_id_like(column: str, series: pd.Series) -> bool:
    normalized = column.lower()
    if any(keyword in normalized for keyword in ID_LIKE_KEYWORDS):
        return True

    non_null = series.dropna()
    if non_null.empty:
        return False

    uniqueness_ratio = float(non_null.nunique(dropna=True) / max(len(non_null), 1))
    if not pd.api.types.is_numeric_dtype(series):
        return bool(uniqueness_ratio >= 0.95)

    return bool(uniqueness_ratio >= 0.98 and pd.api.types.is_integer_dtype(series))


def _is_binary_numeric(series: pd.Series) -> bool:
    non_null = series.dropna()
    if non_null.empty:
        return False
    return bool(non_null.nunique(dropna=True) <= 2)


def _protected_columns(df: pd.DataFrame, target_column: Optional[str] = None) -> List[str]:
    protected: List[str] = []
    for column in df.columns:
        series = df[column]
        if target_column and column == target_column:
            protected.append(column)
            continue
        if not pd.api.types.is_numeric_dtype(series):
            protected.append(column)
            continue
        if _is_binary_numeric(series):
            protected.append(column)
            continue
        if _is_datetime_like(series):
            protected.append(column)
            continue
        if _is_id_like(column, series):
            protected.append(column)
    return protected


def _quantizable_numeric_columns(
    df: pd.DataFrame,
    protected_columns: List[str],
    numeric_only: bool = True,
) -> List[str]:
    if not numeric_only:
        return []

    protected = set(protected_columns)
    columns: List[str] = []
    for column in df.columns:
        if column in protected:
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            columns.append(column)
    return columns


def _sample_inner_product_error(original: np.ndarray, reconstructed: np.ndarray) -> float:
    if original.shape[0] < 2:
        return 0.0

    sample_size = min(original.shape[0], 64)
    left = original[:sample_size]
    right = np.flipud(original[:sample_size])
    left_reconstructed = reconstructed[:sample_size]
    right_reconstructed = np.flipud(reconstructed[:sample_size])

    denom = np.maximum(np.abs(np.sum(left * right, axis=1)), 1e-8)
    error = np.abs(
        np.sum(left_reconstructed * right_reconstructed, axis=1)
        - np.sum(left * right, axis=1)
    ) / denom
    return float(np.mean(error))


def _compress_numeric_block(block_df: pd.DataFrame) -> Dict[str, object]:
    matrix = block_df.to_numpy(dtype=np.float32, copy=True)
    null_mask = np.isnan(matrix)
    fill_values = np.nanmedian(matrix, axis=0)
    fill_values = np.where(np.isnan(fill_values), 0.0, fill_values).astype(np.float32)
    filled = np.where(null_mask, fill_values, matrix)

    seed = 42
    rng = np.random.default_rng(seed)
    dim = filled.shape[1]

    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=dim).astype(np.float32)
    permutation = rng.permutation(dim)
    rotated = filled[:, permutation] * signs

    mins = rotated.min(axis=0)
    maxs = rotated.max(axis=0)
    scales = maxs - mins
    safe_scales = np.where(scales < 1e-8, 1.0, scales)

    normalized = (rotated - mins) / safe_scales
    quantized = np.clip(np.round(normalized * 255.0), 0, 255).astype(np.uint8)
    reconstructed_rotated = (quantized.astype(np.float32) / 255.0) * safe_scales + mins

    inverse_permutation = np.argsort(permutation)
    reconstructed = reconstructed_rotated[:, inverse_permutation] * signs[inverse_permutation]
    reconstructed = reconstructed.astype(np.float32)
    reconstructed[null_mask] = np.nan

    mse = float(np.nanmean((matrix - reconstructed) ** 2))
    ip_error = _sample_inner_product_error(filled, np.where(np.isnan(reconstructed), fill_values, reconstructed))

    raw_bytes = int(matrix.nbytes)
    compressed_bytes = int(quantized.nbytes + mins.nbytes + safe_scales.nbytes + signs.nbytes + permutation.nbytes)

    return {
        "numeric_block": matrix,
        "quantized_block": quantized,
        "dequantized_block": reconstructed,
        "metadata": {
            "seed": seed,
            "rotation": "random_sign_permutation",
            "bits_per_coordinate": 8,
            "columns": list(block_df.columns),
            "permutation": permutation.tolist(),
            "signs": signs.tolist(),
            "mins": mins.tolist(),
            "scales": safe_scales.tolist(),
            "fill_values": fill_values.tolist(),
        },
        "distortion_metrics": {
            "mse": mse,
            "inner_product_error": ip_error,
        },
        "estimated_memory_saved_mb": max(0.0, (raw_bytes - compressed_bytes) / (1024 * 1024)),
    }


def ingest_uploaded_dataset(
    name: str,
    df: pd.DataFrame,
    *,
    enabled: bool,
    mode: str,
    numeric_only: bool,
    min_rows: int,
    max_allowed_mse: float,
    max_allowed_ip_error: float,
    target_column: Optional[str] = None,
) -> IngestedDatasetAsset:
    start = perf_counter()
    protected_columns = _protected_columns(df, target_column=target_column)
    eligible_numeric_columns = _quantizable_numeric_columns(df, protected_columns, numeric_only=numeric_only)
    diagnostics: Dict[str, object] = {
        "enabled": bool(enabled),
        "mode": mode if enabled else "disabled",
        "protected_columns": protected_columns,
        "eligible_numeric_columns": eligible_numeric_columns,
        "compressed_numeric_columns": [],
        "bypass_reason": None,
        "distortion_metrics": None,
        "estimated_memory_saved_mb": 0.0,
        "latency_ms": 0.0,
        "used_compressed_execution": False,
    }

    protected_df = df[protected_columns].copy() if protected_columns else pd.DataFrame(index=df.index)
    fallback_asset = IngestedDatasetAsset(
        name=name,
        raw_df=df,
        protected_df=protected_df,
        dequantized_df=df,
        numeric_block=None,
        quantized_block=None,
        quantizer_metadata={},
        mode="full",
        diagnostics=diagnostics,
    )

    if not enabled:
        diagnostics["bypass_reason"] = "upload_compression_disabled"
    elif len(df) < min_rows:
        diagnostics["bypass_reason"] = "below_min_rows"
    elif len(eligible_numeric_columns) < 2:
        diagnostics["bypass_reason"] = "insufficient_eligible_numeric_columns"
    else:
        numeric_df = df[eligible_numeric_columns]
        compressed = _compress_numeric_block(numeric_df)
        diagnostics["compressed_numeric_columns"] = list(eligible_numeric_columns)
        diagnostics["distortion_metrics"] = compressed["distortion_metrics"]
        diagnostics["estimated_memory_saved_mb"] = compressed["estimated_memory_saved_mb"]

        within_threshold = (
            compressed["distortion_metrics"]["mse"] <= max_allowed_mse
            and compressed["distortion_metrics"]["inner_product_error"] <= max_allowed_ip_error
        )

        if not within_threshold:
            diagnostics["bypass_reason"] = "distortion_threshold_exceeded"
        else:
            used_compressed_execution = mode == "safe_default_on"
            diagnostics["used_compressed_execution"] = used_compressed_execution
            dequantized_df = df

            if used_compressed_execution:
                dequantized_df = df.copy()
                dequantized_values = compressed["dequantized_block"]
                for index, column in enumerate(eligible_numeric_columns):
                    dequantized_df[column] = dequantized_values[:, index]

            fallback_asset = IngestedDatasetAsset(
                name=name,
                raw_df=df,
                protected_df=protected_df,
                dequantized_df=dequantized_df,
                numeric_block=compressed["numeric_block"],
                quantized_block=compressed["quantized_block"],
                quantizer_metadata=compressed["metadata"],
                mode="compressed" if used_compressed_execution else "shadow",
                diagnostics=diagnostics,
            )

            if not used_compressed_execution:
                diagnostics["bypass_reason"] = "shadow_mode_full_precision_authoritative"

    diagnostics["latency_ms"] = round((perf_counter() - start) * 1000, 3)
    return fallback_asset
