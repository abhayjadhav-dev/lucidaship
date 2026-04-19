"""
Model storage service — save/load/list/delete trained .joblib model artifacts.
Scoped by tenant_id for isolation. Uses filesystem, not database.
"""

import logging
from pathlib import Path
from typing import Dict, List
from datetime import datetime, timezone
import shutil

import joblib

from app.core.config import get_settings
from adaptive_scorer import UniversalAdaptiveScorer

logger = logging.getLogger(__name__)
settings = get_settings()


def _tenant_dir(tenant_id: str) -> Path:
    """Get or create the model directory for a tenant."""
    path = Path(settings.MODEL_ARTIFACTS_DIR) / tenant_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _versioned_candidates(directory: Path, model_name: str) -> List[Path]:
    return sorted(
        directory.glob(f"{model_name}__*.joblib"),
        key=lambda candidate: candidate.stat().st_mtime,
        reverse=True,
    )


def _load_first_compatible(paths: List[Path]) -> UniversalAdaptiveScorer:
    errors = []
    for path in paths:
        try:
            return joblib.load(path)
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")
    raise RuntimeError("; ".join(errors) if errors else "No compatible model artifacts found")


def save_model(model: UniversalAdaptiveScorer, tenant_id: str, model_name: str) -> str:
    """
    Save a trained model to disk.
    Returns the artifact path.
    """
    directory = _tenant_dir(tenant_id)
    version_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    versioned_filepath = directory / f"{model_name}__{version_id}.joblib"
    latest_filepath = directory / f"{model_name}.joblib"

    joblib.dump(model, versioned_filepath)
    shutil.copy2(versioned_filepath, latest_filepath)

    logger.info(
        "Model saved: tenant=%s model=%s version=%s path=%s",
        tenant_id,
        model_name,
        version_id,
        versioned_filepath,
    )
    return str(versioned_filepath)


def load_model(tenant_id: str, model_name: str) -> UniversalAdaptiveScorer:
    """
    Load a model from disk. Raises FileNotFoundError if not found.
    """
    directory = _tenant_dir(tenant_id)
    filepath = directory / f"{model_name}.joblib"
    candidate_paths = [filepath] if filepath.exists() else []
    candidate_paths.extend(_versioned_candidates(directory, model_name))
    if not candidate_paths:
        raise FileNotFoundError(f"No model '{model_name}' found for tenant '{tenant_id}'")
    model = _load_first_compatible(candidate_paths)
    logger.info("Model loaded: tenant=%s model=%s", tenant_id, model_name)
    return model


def load_model_from_path(artifact_path: str) -> UniversalAdaptiveScorer:
    """Load a model from a specific artifact path."""
    filepath = Path(artifact_path)
    if not filepath.exists():
        raise FileNotFoundError(f"Model artifact missing: {artifact_path}")
    model = joblib.load(filepath)
    logger.info("Model loaded from artifact path=%s", filepath)
    return model


def list_models(tenant_id: str) -> List[str]:
    """Return list of model names for a tenant."""
    directory = _tenant_dir(tenant_id)
    return [
        f.stem  # filename without extension
        for f in directory.glob("*.joblib")
        if "__" not in f.stem
    ]


def delete_model(tenant_id: str, model_name: str) -> bool:
    """Delete a model file. Returns True if deleted, False if not found."""
    directory = _tenant_dir(tenant_id)
    filepath = directory / f"{model_name}.joblib"
    deleted_any = False

    if filepath.exists():
        filepath.unlink()
        deleted_any = True

    for versioned_file in directory.glob(f"{model_name}__*.joblib"):
        versioned_file.unlink()
        deleted_any = True

    if deleted_any:
        logger.info("Model deleted: tenant=%s model=%s", tenant_id, model_name)
    return deleted_any


def load_all_models() -> Dict[str, Dict[str, UniversalAdaptiveScorer]]:
    """
    Scan model_artifacts/ and load ALL models into memory.
    Called on app startup so models survive restarts.

    Returns: {tenant_id: {model_name: scorer_obj}}
    """
    all_models: Dict[str, Dict[str, UniversalAdaptiveScorer]] = {}
    artifacts_dir = Path(settings.MODEL_ARTIFACTS_DIR)

    if not artifacts_dir.exists():
        logger.info("No model_artifacts directory found, starting fresh")
        return all_models

    for tenant_dir in artifacts_dir.iterdir():
        if not tenant_dir.is_dir():
            continue
        tenant_id = tenant_dir.name
        all_models[tenant_id] = {}

        for model_file in tenant_dir.glob("*.joblib"):
            model_name = model_file.stem
            if "__" in model_name:
                continue
            try:
                model = _load_first_compatible([model_file, *_versioned_candidates(tenant_dir, model_name)])
                all_models[tenant_id][model_name] = model
                logger.info("Loaded model on startup: tenant=%s model=%s", tenant_id, model_name)
            except Exception as e:
                logger.error("Failed to load model %s/%s: %s", tenant_id, model_name, e)

    total = sum(len(models) for models in all_models.values())
    logger.info("Startup model reload complete: %d models across %d tenants", total, len(all_models))
    return all_models
