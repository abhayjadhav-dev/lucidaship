"""Background training task for non-blocking async model training."""

from __future__ import annotations

import io
import json
import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.core.config import get_settings
from app.database import get_db
from app.services.dataset_relationships import DatasetAsset, execute_merge_plan, prepare_combined_dataset
from app.services.upload_quantization import IngestedDatasetAsset, ingest_uploaded_dataset
from app.services import model_storage
from adaptive_scorer import UniversalAdaptiveScorer

logger = logging.getLogger(__name__)
settings = get_settings()


def _ingest_file(filename: str, file_content: bytes, target_column: Optional[str]) -> IngestedDatasetAsset:
    df = pd.read_csv(io.BytesIO(file_content), low_memory=True)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df = df.dropna(axis=1, how="all").dropna(how="all")

    if len(df) > settings.MAX_ROWS_PER_FILE:
        raise ValueError(f"File '{filename}' has {len(df)} rows (max {settings.MAX_ROWS_PER_FILE})")
    if len(df.columns) > settings.MAX_COLUMNS_PER_FILE:
        raise ValueError(
            f"File '{filename}' has {len(df.columns)} columns (max {settings.MAX_COLUMNS_PER_FILE})"
        )

    return ingest_uploaded_dataset(
        filename,
        df,
        enabled=settings.UPLOAD_COMPRESSION_ENABLED,
        mode=settings.UPLOAD_COMPRESSION_MODE,
        numeric_only=settings.UPLOAD_COMPRESSION_NUMERIC_ONLY,
        min_rows=settings.UPLOAD_COMPRESSION_MIN_ROWS,
        max_allowed_mse=settings.UPLOAD_COMPRESSION_MAX_ALLOWED_MSE,
        max_allowed_ip_error=settings.UPLOAD_COMPRESSION_MAX_ALLOWED_IP_ERROR,
        target_column=target_column,
    )


def _prepare_assets(ingested_items: List[Tuple[str, IngestedDatasetAsset]]) -> List[DatasetAsset]:
    return [
        DatasetAsset(
            name=name,
            df=ingested.raw_df,
            raw_df=ingested.raw_df,
            protected_df=ingested.protected_df,
            dequantized_df=ingested.dequantized_df,
            compression=ingested.diagnostics,
            execution_mode=ingested.mode,
        )
        for name, ingested in ingested_items
    ]


def _compression_summary(assets: List[DatasetAsset]) -> Dict[str, Any]:
    total_memory_saved = 0.0
    used_compressed_execution = False
    datasets = []
    for asset in assets:
        compression = asset.compression or {}
        total_memory_saved += float(compression.get("estimated_memory_saved_mb") or 0.0)
        used_compressed_execution = used_compressed_execution or bool(compression.get("used_compressed_execution"))
        datasets.append({
            "dataset": asset.name,
            "mode": compression.get("mode", asset.execution_mode),
            "compressed_numeric_columns": compression.get("compressed_numeric_columns", []),
            "bypass_reason": compression.get("bypass_reason"),
            "estimated_memory_saved_mb": compression.get("estimated_memory_saved_mb", 0.0),
        })
    return {
        "enabled": settings.UPLOAD_COMPRESSION_ENABLED,
        "mode": settings.UPLOAD_COMPRESSION_MODE,
        "used_compressed_execution": used_compressed_execution,
        "estimated_memory_saved_mb": round(total_memory_saved, 4),
        "datasets": datasets,
    }


def _resolve_combined_dataset(assets: List[DatasetAsset], compression: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if compression.get("used_compressed_execution"):
        execution_assets = [asset.for_execution() for asset in assets]
        _, merge_plan = prepare_combined_dataset(assets)
        return execute_merge_plan(execution_assets, merge_plan)
    return prepare_combined_dataset(assets)


def execute_training_task(
    job_id: str,
    files_data: List[tuple],  # List of (filename, file_content_bytes)
    target_column: Optional[str],
    mode: str,
    model_name: str,
    tenant_id: str,
    progress_callback,
    **kwargs,
) -> Dict[str, Any]:
    """Execute async model training with safe multi-CSV merging."""
    try:
        from app.api.scoring import _enforce_model_creation_quota, _set_model

        quota_violation = _enforce_model_creation_quota(tenant_id, model_name)
        if quota_violation:
            raise ValueError(quota_violation["message"])

        progress_callback(10, "Loading and profiling CSV files...")
        if not files_data:
            raise ValueError("No valid CSV files provided")

        ingested_items: List[Tuple[str, IngestedDatasetAsset]] = []
        for filename, file_content in files_data:
            ingested = _ingest_file(filename, file_content, target_column=target_column)
            ingested_items.append((filename, ingested))
            logger.info("Loaded %s: %d rows × %d cols", filename, len(ingested.raw_df), len(ingested.raw_df.columns))

        assets = _prepare_assets(ingested_items)
        compression = _compression_summary(assets)

        progress_callback(30, f"Merging {len(assets)} dataset(s) safely...")
        merged_df, merge_plan = _resolve_combined_dataset(assets, compression)

        if len(assets) > 1 and not merge_plan.get("executed_steps") and merge_plan.get("warnings"):
            raise ValueError(
                "No safe dataset relationship was found. Upload datasets with stronger bridge keys "
                "or inspect /merge-plan before training."
            )

        if merged_df.shape[0] < 10:
            raise ValueError(f"Need at least 10 rows after merge, got {merged_df.shape[0]}")
        if merged_df.shape[1] < 2:
            raise ValueError(f"Need at least 2 columns after merge, got {merged_df.shape[1]}")

        progress_callback(55, "Training lead ranking model...")
        scorer = UniversalAdaptiveScorer()

        training_target = target_column
        if mode == "unsupervised":
            merged_df = merged_df.copy()
            merged_df["__synthetic_target__"] = (np.arange(len(merged_df)) % 2).astype(int)
            training_target = "__synthetic_target__"

        train_result = scorer.train(
            merged_df,
            target_col=training_target,
            client_id=model_name,
        )

        if mode == "supervised" and not target_column:
            recommendation = train_result["analysis"]["target_diagnostics"].get("recommendation")
            if recommendation == "manual_review_recommended":
                raise ValueError(
                    "Automatic target detection is not reliable for this dataset. "
                    "Please provide target_column explicitly."
                )

        if mode == "unsupervised" and scorer.scorer and hasattr(scorer.scorer, "metadata"):
            scorer.scorer.metadata["training_mode"] = "unsupervised"
            scorer.scorer.metadata["original_columns"] = list(merged_df.columns)

        progress_callback(80, "Saving trained model...")
        artifact_path = model_storage.save_model(scorer, tenant_id, model_name)

        # Update in-memory cache for immediate scoring availability.
        _set_model(tenant_id, model_name, scorer)

        progress_callback(90, "Persisting training metadata...")
        run_id = str(uuid.uuid4())
        metrics_payload = train_result["metrics"]
        conn = get_db()
        conn.execute(
            """INSERT INTO training_runs (id, tenant_id, model_name, artifact_path, metrics, row_count, created_at)
               VALUES (?, ?, ?, ?, ?, ?, datetime('now'))""",
            [run_id, tenant_id, model_name, artifact_path, json.dumps(metrics_payload, default=str), int(len(merged_df))],
        )

        progress_callback(98, "Finalizing...")
        return {
            "success": True,
            "job_id": job_id,
            "model_name": model_name,
            "target_column": train_result["analysis"].get("target_column"),
            "mode": mode,
            "dataset": {
                "rows": int(len(merged_df)),
                "columns": int(len(merged_df.columns)),
            },
            "merge_summary": merge_plan,
            "compression": compression,
            "analysis": train_result["analysis"],
            "metrics": train_result["metrics"],
            "timestamp": pd.Timestamp.now().isoformat(),
        }
    except Exception as e:
        logger.exception("Training task failed: %s", e)
        raise
