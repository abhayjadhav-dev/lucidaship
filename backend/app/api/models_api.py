"""
Model management API routes — list, info, delete trained models.
Uses raw SQL against Turso/libSQL. Protected by Clerk auth.
"""

import json
import logging
from fastapi import APIRouter, Depends
import pandas as pd

from app.database import get_db
from app.core.auth import get_current_user, require_role
from app.core.responses import success_response, error_response
from app.services import model_storage
from app.api.scoring import _delete_model_from_cache, _get_model

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/models", tags=["Models"])


def _feedback_summary(conn, tenant_id: str, model_name: str):
    """Summarize persisted feedback events for a model."""
    total_rows = conn.execute(
        """SELECT COUNT(*)
           FROM feedback_events
           WHERE tenant_id = ? AND model_name = ?""",
        [tenant_id, model_name],
    ).rows
    total_feedback = int(total_rows[0][0]) if total_rows else 0

    recent_rows = conn.execute(
        """SELECT actual_outcome, predicted_score, feedback_at
           FROM feedback_events
           WHERE tenant_id = ? AND model_name = ?
           ORDER BY feedback_at DESC
           LIMIT 10""",
        [tenant_id, model_name],
    ).rows

    last_feedback_at = recent_rows[0][2] if recent_rows else None
    avg_recent_score = (
        round(sum(float(row[1]) for row in recent_rows if row[1] is not None) / len(recent_rows), 2)
        if recent_rows
        else None
    )
    recent_positive_rate = (
        round(sum(int(row[0]) for row in recent_rows) / len(recent_rows), 4)
        if recent_rows
        else None
    )

    readiness = "insufficient_feedback"
    if total_feedback >= 25:
        readiness = "ready_for_feedback_retrain"
    elif total_feedback >= 10:
        readiness = "collecting_signal"

    return {
        "total_feedback_events": total_feedback,
        "last_feedback_at": last_feedback_at,
        "avg_recent_predicted_score": avg_recent_score,
        "recent_positive_rate": recent_positive_rate,
        "retrain_readiness": readiness,
    }


def _segment_feedback_insights(conn, tenant_id: str, model_name: str):
    """Find segment-level drift hotspots from stored feedback events."""
    rows = conn.execute(
        """SELECT lead_data, actual_outcome, predicted_score
           FROM feedback_events
           WHERE tenant_id = ? AND model_name = ?
           ORDER BY feedback_at DESC
           LIMIT 500""",
        [tenant_id, model_name],
    ).rows

    records = []
    for lead_data_json, actual_outcome, predicted_score in rows:
        try:
            lead_data = json.loads(lead_data_json) if lead_data_json else {}
        except json.JSONDecodeError:
            continue
        lead_data["_actual_outcome"] = int(actual_outcome)
        lead_data["_predicted_score"] = float(predicted_score) if predicted_score is not None else 0.0
        records.append(lead_data)

    if not records:
        return []

    df = pd.DataFrame(records)
    candidate_columns = []
    for col in df.columns:
        if col.startswith("_"):
            continue
        non_null = df[col].dropna()
        if non_null.empty:
            continue
        unique_count = non_null.nunique()
        if 2 <= unique_count <= 12:
            candidate_columns.append(col)

    insights = []
    for col in candidate_columns[:4]:
        grouped = (
            df.dropna(subset=[col])
            .groupby(col)
            .agg(
                sample_count=("_actual_outcome", "count"),
                actual_win_rate=("_actual_outcome", "mean"),
                avg_predicted_score=("_predicted_score", "mean"),
            )
            .reset_index()
        )

        for _, row in grouped.iterrows():
            sample_count = int(row["sample_count"])
            if sample_count < 5:
                continue
            actual_win_rate = float(row["actual_win_rate"])
            avg_predicted_score = float(row["avg_predicted_score"])
            expected_win_rate = avg_predicted_score / 100
            drift_gap = round(avg_predicted_score - (actual_win_rate * 100), 2)
            segment_readiness = "stable"
            if sample_count >= 10 and abs(drift_gap) >= 20:
                segment_readiness = "segment_retrain_candidate"
            elif sample_count >= 5 and abs(drift_gap) >= 12:
                segment_readiness = "watch_segment"

            insights.append({
                "dimension": str(col),
                "segment": str(row[col]),
                "sample_count": sample_count,
                "actual_win_rate": round(actual_win_rate, 4),
                "avg_predicted_score": round(avg_predicted_score, 2),
                "drift_gap": drift_gap,
                "segment_readiness": segment_readiness,
            })

    insights.sort(key=lambda item: (item["segment_readiness"] == "segment_retrain_candidate", abs(item["drift_gap"]), item["sample_count"]), reverse=True)
    return insights[:12]


@router.get("")
def list_models(
    user: dict = Depends(get_current_user),
):
    """List all trained models for the current tenant."""
    tenant_id = user["tenant_id"]

    # Get model names from disk
    model_names = model_storage.list_models(tenant_id)

    # Enrich with DB metadata
    conn = get_db()
    models = []
    for name in model_names:
        result = conn.execute(
            """SELECT id, model_name, artifact_path, metrics, row_count, created_at
               FROM training_runs
               WHERE tenant_id = ? AND model_name = ?
               ORDER BY created_at DESC LIMIT 1""",
            [tenant_id, name],
        )

        info = {"model_name": name}
        model = _get_model(tenant_id, name)
        if result.rows:
            row = result.rows[0]
            info["trained_at"] = row[5]
            info["n_rows"] = row[4]
            if row[3]:
                try:
                    metrics = json.loads(row[3])
                    info["accuracy"] = metrics.get("accuracy")
                    info["roc_auc"] = metrics.get("roc_auc")
                    info["precision"] = metrics.get("precision")
                    info["recall"] = metrics.get("recall")
                    info["ranking_version"] = metrics.get("ranking_version")
                except json.JSONDecodeError:
                    pass
        if model and model.analyzer:
            info["target_column"] = model.analyzer.target_col
            info["target_recommendation"] = model.analyzer.get_target_diagnostics().get("recommendation")
        info["feedback_summary"] = _feedback_summary(conn, tenant_id, name)
        info["segment_hotspots"] = _segment_feedback_insights(conn, tenant_id, name)[:4]
        models.append(info)

    return success_response(data={
        "count": len(models),
        "models": models,
    })


@router.get("/{model_name}")
def get_model_info(
    model_name: str,
    user: dict = Depends(get_current_user),
):
    """Get detailed info about a specific model."""
    tenant_id = user["tenant_id"]
    scorer = _get_model(tenant_id, model_name)

    if not scorer:
        return error_response("MODEL_NOT_FOUND", f"No model '{model_name}' found", 404)

    conn = get_db()
    result = conn.execute(
        """SELECT id, model_name, artifact_path, metrics, row_count, created_at
           FROM training_runs
           WHERE tenant_id = ? AND model_name = ?
           ORDER BY created_at DESC LIMIT 1""",
        [tenant_id, model_name],
    )

    info = {
        "model_name": model_name,
        "feature_names": scorer.scorer.feature_names if scorer.scorer else [],
        "target_column": scorer.analyzer.target_col if scorer.analyzer else None,
        "target_diagnostics": scorer.analyzer.get_target_diagnostics() if scorer.analyzer else {},
        "feature_blueprint": scorer.engineer.summarize_feature_blueprint() if scorer.engineer else {},
        "ranking_version": scorer.scorer.metadata.get("ranking_version") if scorer.scorer else None,
        "rationale_version": scorer.scorer.metadata.get("rationale_version") if scorer.scorer else None,
    }

    if result.rows:
        row = result.rows[0]
        info["trained_at"] = row[5]
        info["n_rows"] = row[4]
        if row[3]:
            try:
                info["metrics"] = json.loads(row[3])
            except json.JSONDecodeError:
                pass

    info["feedback_summary"] = _feedback_summary(conn, tenant_id, model_name)
    info["segment_hotspots"] = _segment_feedback_insights(conn, tenant_id, model_name)

    timeline_rows = conn.execute(
        """SELECT feedback_at, actual_outcome, predicted_score, rank_at_score_time
           FROM feedback_events
           WHERE tenant_id = ? AND model_name = ?
           ORDER BY feedback_at DESC
           LIMIT 20""",
        [tenant_id, model_name],
    ).rows
    info["feedback_timeline"] = [
        {
            "feedback_at": row[0],
            "actual_outcome": int(row[1]),
            "predicted_score": float(row[2]) if row[2] is not None else None,
            "rank_at_score_time": int(row[3]) if row[3] is not None else None,
        }
        for row in timeline_rows
    ]

    return success_response(data=info)


@router.delete("/{model_name}")
def delete_model(
    model_name: str,
    user: dict = Depends(require_role(["admin"])),
):
    """Delete a model. Admin only."""
    tenant_id = user["tenant_id"]

    # Delete from disk
    deleted = model_storage.delete_model(tenant_id, model_name)
    if not deleted:
        return error_response("MODEL_NOT_FOUND", f"No model '{model_name}' found", 404)

    # Remove from cache
    _delete_model_from_cache(tenant_id, model_name)

    # Delete DB records
    conn = get_db()
    conn.execute(
        "DELETE FROM training_runs WHERE tenant_id = ? AND model_name = ?",
        [tenant_id, model_name],
    )

    logger.info("Model deleted: tenant=%s model=%s", tenant_id, model_name)
    return success_response(data={"message": f"Model '{model_name}' deleted"})
