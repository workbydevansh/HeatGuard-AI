from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .config import LATITUDE_CANDIDATES, LONGITUDE_CANDIDATES
from .preprocessing import augment_competition_columns, detect_date_column, detect_location_column, detect_target_column
from .utils import find_column_by_candidates, safe_numeric_columns


@dataclass
class FeatureBuildResult:
    frame: pd.DataFrame
    feature_names: list[str]
    target_name: Optional[str]
    date_column: Optional[str]
    location_column: Optional[str]


def _sort_for_time(df: pd.DataFrame, date_col: Optional[str], location_col: Optional[str]) -> pd.DataFrame:
    df = df.copy()
    sort_cols = []
    if location_col and location_col in df.columns:
        sort_cols.append(location_col)
    if date_col and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        sort_cols.append(date_col)
    for order_col in ("row_day_index", "day_index"):
        if order_col in df.columns and order_col not in sort_cols:
            sort_cols.append(order_col)
    if sort_cols:
        return df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    return df.reset_index(drop=True)


def _add_time_features(df: pd.DataFrame, date_col: Optional[str]) -> pd.DataFrame:
    if not date_col or date_col not in df.columns:
        return df

    output = df.copy()
    dates = pd.to_datetime(output[date_col], errors="coerce")
    output["day"] = dates.dt.day
    output["month"] = dates.dt.month
    output["year"] = dates.dt.year
    output["dayofyear"] = dates.dt.dayofyear
    output["weekofyear"] = dates.dt.isocalendar().week.astype("float")
    output["sin_dayofyear"] = np.sin(2 * np.pi * output["dayofyear"] / 365.25)
    output["cos_dayofyear"] = np.cos(2 * np.pi * output["dayofyear"] / 365.25)
    output["sin_month"] = np.sin(2 * np.pi * output["month"] / 12)
    output["cos_month"] = np.cos(2 * np.pi * output["month"] / 12)
    return output


def _add_location_code(df: pd.DataFrame, location_col: Optional[str]) -> pd.DataFrame:
    if not location_col or location_col not in df.columns:
        return df
    output = df.copy()
    output[f"{location_col}_code"] = pd.Categorical(output[location_col].astype(str)).codes
    return output


def _grouped_shift(df: pd.DataFrame, column: str, lag: int, location_col: Optional[str]) -> pd.Series:
    if location_col and location_col in df.columns:
        return df.groupby(location_col, observed=False)[column].shift(lag)
    return df[column].shift(lag)


def _grouped_rolling(
    df: pd.DataFrame,
    column: str,
    window: int,
    agg: str,
    location_col: Optional[str],
) -> pd.Series:
    def rolling_series(series: pd.Series) -> pd.Series:
        shifted = series.shift(1)
        roller = shifted.rolling(window=window, min_periods=1)
        if agg == "mean":
            return roller.mean()
        if agg == "std":
            return roller.std()
        if agg == "max":
            return roller.max()
        raise ValueError(f"Unsupported rolling aggregation: {agg}")

    if location_col and location_col in df.columns:
        return df.groupby(location_col, observed=False)[column].transform(rolling_series)
    return rolling_series(df[column])


def build_feature_frame(
    df: pd.DataFrame,
    *,
    target_col: Optional[str] = None,
    date_col: Optional[str] = None,
    location_col: Optional[str] = None,
    feature_names: Optional[list[str]] = None,
) -> FeatureBuildResult:
    """Create model features with only current and past information."""
    if df.empty:
        raise ValueError("The input dataframe is empty.")

    output = augment_competition_columns(df)
    target_col = target_col or detect_target_column(output)
    date_col = date_col or detect_date_column(output)
    location_col = location_col or detect_location_column(output)

    output = _sort_for_time(output, date_col, location_col)
    output = _add_time_features(output, date_col)
    output = _add_location_code(output, location_col)

    numeric_cols = [col for col in safe_numeric_columns(output) if col != date_col]
    for col in numeric_cols:
        output[col] = pd.to_numeric(output[col], errors="coerce")

    lat_col = find_column_by_candidates(output.columns, LATITUDE_CANDIDATES)
    lon_col = find_column_by_candidates(output.columns, LONGITUDE_CANDIDATES)
    static_cols = {col for col in [lat_col, lon_col] if col}

    lag_source_cols = [
        col
        for col in numeric_cols
        if col not in static_cols
        and not col.endswith("_code")
        and col not in {"day", "month", "year", "dayofyear", "weekofyear"}
    ]

    engineered_columns: dict[str, pd.Series] = {}
    for col in lag_source_cols:
        for lag in (1, 2, 3, 7, 10):
            engineered_columns[f"{col}_lag_{lag}"] = _grouped_shift(output, col, lag, location_col)
        engineered_columns[f"{col}_rolling_mean_3"] = _grouped_rolling(output, col, 3, "mean", location_col)
        engineered_columns[f"{col}_rolling_mean_7"] = _grouped_rolling(output, col, 7, "mean", location_col)
        engineered_columns[f"{col}_rolling_std_7"] = _grouped_rolling(output, col, 7, "std", location_col)
        engineered_columns[f"{col}_rolling_max_7"] = _grouped_rolling(output, col, 7, "max", location_col)

    if engineered_columns:
        output = pd.concat([output, pd.DataFrame(engineered_columns, index=output.index)], axis=1)

    engineered_numeric = safe_numeric_columns(output)
    feature_cols = [
        col
        for col in engineered_numeric
        if col != "__future_target__"
        and col != date_col
        and col != target_col
        and not str(col).startswith("__")
    ]

    if feature_names:
        for feature in feature_names:
            if feature not in output.columns:
                output[feature] = np.nan
        feature_cols = feature_names

    return FeatureBuildResult(
        frame=output,
        feature_names=feature_cols,
        target_name=target_col,
        date_column=date_col,
        location_column=location_col,
    )


def build_supervised_frame(
    df: pd.DataFrame,
    *,
    target_col: Optional[str] = None,
    date_col: Optional[str] = None,
    location_col: Optional[str] = None,
    horizon: int = 10,
    feature_names: Optional[list[str]] = None,
) -> FeatureBuildResult:
    """Build a direct lead-time forecasting dataset.

    The label is shifted backward by `horizon`, while lag and rolling features
    are shifted from previous observations. This avoids future target leakage.
    """
    result = build_feature_frame(
        df,
        target_col=target_col,
        date_col=date_col,
        location_col=location_col,
        feature_names=feature_names,
    )

    if not result.target_name or result.target_name not in result.frame.columns:
        raise ValueError(
            "Could not detect the wet-bulb target column. Pass --target-col or select it in the app."
        )

    output = result.frame.copy()
    if result.location_column and result.location_column in output.columns:
        output["__future_target__"] = output.groupby(result.location_column, observed=False)[
            result.target_name
        ].shift(-horizon)
    else:
        output["__future_target__"] = output[result.target_name].shift(-horizon)

    output = output.dropna(subset=["__future_target__"]).reset_index(drop=True)

    return FeatureBuildResult(
        frame=output,
        feature_names=result.feature_names,
        target_name=result.target_name,
        date_column=result.date_column,
        location_column=result.location_column,
    )
