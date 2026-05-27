from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .config import RISK_LEVELS


def normalize_column_name(name: object) -> str:
    """Normalize a column name for tolerant matching."""
    text = str(name).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def find_column_by_candidates(
    columns: Iterable[str],
    candidates: Iterable[str],
    *,
    prefer_numeric: Optional[pd.DataFrame] = None,
) -> Optional[str]:
    """Return the first column whose normalized name matches a candidate."""
    normalized = {col: normalize_column_name(col) for col in columns}
    candidate_norm = [normalize_column_name(candidate) for candidate in candidates]

    for candidate in candidate_norm:
        for col, norm in normalized.items():
            if norm == candidate:
                if prefer_numeric is None or pd.api.types.is_numeric_dtype(prefer_numeric[col]):
                    return col

    for candidate in candidate_norm:
        for col, norm in normalized.items():
            if candidate in norm:
                if prefer_numeric is None or pd.api.types.is_numeric_dtype(prefer_numeric[col]):
                    return col

    return None


def safe_numeric_columns(df: pd.DataFrame) -> list[str]:
    """Return columns that can reasonably be treated as numeric model features."""
    numeric_cols: list[str] = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().mean() >= 0.8:
            numeric_cols.append(col)
    return numeric_cols


def read_json(path: Path, default: object | None = None) -> object:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, default=str)


def classify_wbt(value: float | int | None) -> str:
    """Classify wet-bulb temperature in Celsius.

    Thresholds are intentionally centralized and can be adjusted if the
    challenge organizers or domain experts provide region-specific cutoffs.
    """
    if value is None or not np.isfinite(value):
        return "Unknown"
    value = float(value)
    for level in RISK_LEVELS:
        if level["min"] <= value < level["max"]:
            return str(level["name"])
    return "Unknown"


def risk_color(risk_level: str) -> str:
    for level in RISK_LEVELS:
        if level["name"].lower() == str(risk_level).lower():
            return str(level["color"])
    return "#94A3B8"


def risk_score(value: float | int | None) -> float:
    if value is None or not np.isfinite(value):
        return 0.0
    value = float(value)
    return float(np.clip((value - 18.0) / (34.0 - 18.0) * 100.0, 0.0, 100.0))


def approximate_wet_bulb_celsius(temperature_c: float, relative_humidity: float) -> float:
    """Approximate wet-bulb temperature from air temperature and RH.

    This Stull-style approximation is useful for manual demo inputs only; real
    training uses the dataset target column when present.
    """
    t = float(temperature_c)
    rh = float(np.clip(relative_humidity, 1.0, 100.0))
    return float(
        t * math.atan(0.151977 * math.sqrt(rh + 8.313659))
        + math.atan(t + rh)
        - math.atan(rh - 1.676331)
        + 0.00391838 * rh ** 1.5 * math.atan(0.023101 * rh)
        - 4.686035
    )


def confidence_from_rmse(rmse: float | None, *, is_demo: bool = False) -> float:
    if rmse is None or not np.isfinite(rmse):
        base = 64.0
    else:
        base = 100.0 - min(55.0, float(rmse) * 8.0)
    if is_demo:
        base -= 8.0
    return float(np.clip(base, 35.0, 96.0))


def reliability_label(confidence_pct: float) -> str:
    if confidence_pct >= 82:
        return "High"
    if confidence_pct >= 65:
        return "Medium"
    return "Exploratory"


def humanize_feature_name(name: str) -> str:
    cleaned = normalize_column_name(name).replace("_", " ")
    replacements = {
        "wbt": "wet-bulb temperature",
        "wet bulb": "wet-bulb temperature",
        "rh": "relative humidity",
        "temp": "temperature",
        "tmax": "maximum temperature",
        "tmin": "minimum temperature",
        "lag": "previous",
        "rolling": "recent",
        "std": "variability",
    }
    for old, new in replacements.items():
        cleaned = re.sub(rf"\b{re.escape(old)}\b", new, cleaned)
    return cleaned.strip()

