from __future__ import annotations

from typing import Optional

import pandas as pd

from .config import DATE_CANDIDATES, LOCATION_CANDIDATES, TARGET_CANDIDATES
from .utils import find_column_by_candidates


def augment_competition_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add Kaggle competition helper columns derived from row_id when present."""
    output = df.copy()
    if "row_id" not in output.columns:
        return output

    parsed = output["row_id"].astype(str).str.extract(r"^(?P<location_id>[^_]+)_D(?P<row_day_index>\d+)$")
    if "location_id" not in output.columns and parsed["location_id"].notna().any():
        output["location_id"] = parsed["location_id"]
    if "row_day_index" not in output.columns and parsed["row_day_index"].notna().any():
        output["row_day_index"] = pd.to_numeric(parsed["row_day_index"], errors="coerce")
    return output


def detect_target_column(df: pd.DataFrame) -> Optional[str]:
    return find_column_by_candidates(df.columns, TARGET_CANDIDATES, prefer_numeric=df)


def detect_date_column(df: pd.DataFrame) -> Optional[str]:
    detected = find_column_by_candidates(df.columns, DATE_CANDIDATES)
    if detected:
        parsed = pd.to_datetime(df[detected], errors="coerce")
        if parsed.notna().mean() >= 0.6:
            return detected

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        parsed = pd.to_datetime(df[col], errors="coerce")
        if parsed.notna().mean() >= 0.8:
            return col
    return None


def detect_location_column(df: pd.DataFrame) -> Optional[str]:
    candidate = find_column_by_candidates(df.columns, LOCATION_CANDIDATES)
    if candidate and df[candidate].nunique(dropna=True) > 1:
        return candidate
    return None


def coerce_numeric_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    output = df.copy()
    for col in columns:
        if col not in output.columns:
            output[col] = pd.NA
        output[col] = pd.to_numeric(output[col], errors="coerce")
    return output[columns]
