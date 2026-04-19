"""
Standardised API response helpers.
Every endpoint returns {success, data, error} for consistency.
"""

import math
from typing import Any, Optional
from fastapi.responses import JSONResponse
import numpy as np


def _sanitize(obj: Any) -> Any:
    """Recursively convert numpy/pandas types to native Python types."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(obj, np.ndarray):
        return [_sanitize(v) for v in obj.tolist()]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
    return obj


def success_response(data: Any = None, status_code: int = 200) -> JSONResponse:
    """Return a standardised success response."""
    return JSONResponse(
        status_code=status_code,
        content=_sanitize({
            "success": True,
            "data": data,
            "error": None,
        }),
    )


def error_response(code: str, message: str, status_code: int = 400) -> JSONResponse:
    """Return a standardised error response."""
    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "data": None,
            "error": {
                "code": code,
                "message": message,
            },
        },
    )

