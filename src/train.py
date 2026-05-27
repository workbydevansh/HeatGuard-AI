from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    DEFAULT_FORECAST_HORIZON,
    FEATURES_PATH,
    METRICS_PATH,
    MODEL_PATH,
    RANDOM_STATE,
    ensure_directories,
)
from .data_loader import load_available_dataset
from .feature_engineering import build_supervised_frame
from .model import refit_model, save_model_artifact, train_and_select_model
from .preprocessing import (
    augment_competition_columns,
    coerce_numeric_frame,
    detect_date_column,
    detect_location_column,
    detect_target_column,
)
from .utils import write_json


@dataclass
class TrainResult:
    artifact: dict
    metrics: dict
    model_path: Path
    metrics_path: Path
    features_path: Path


def _split_supervised_frame(
    frame: pd.DataFrame,
    *,
    feature_names: list[str],
    date_col: Optional[str],
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, str]:
    valid = frame.dropna(subset=["__future_target__"]).copy()
    X = coerce_numeric_frame(valid, feature_names)
    y = pd.to_numeric(valid["__future_target__"], errors="coerce")
    mask = y.notna()
    X = X.loc[mask]
    y = y.loc[mask]

    if len(X) < 40:
        raise ValueError("Not enough rows after feature engineering. Provide more history or reduce horizon.")

    if date_col and date_col in valid.columns:
        ordered_idx = pd.to_datetime(valid.loc[mask, date_col], errors="coerce").sort_values().index
        X = X.loc[ordered_idx]
        y = y.loc[ordered_idx]
        split_idx = max(1, int(len(X) * 0.8))
        return X.iloc[:split_idx], X.iloc[split_idx:], y.iloc[:split_idx], y.iloc[split_idx:], "chronological_80_20"

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
    )
    return X_train, X_val, y_train, y_val, "random_80_20_no_date_column"


def _limit_training_rows(
    frame: pd.DataFrame,
    max_rows: int | None,
    *,
    date_col: Optional[str] = None,
) -> tuple[pd.DataFrame, dict]:
    if not max_rows or len(frame) <= max_rows:
        return frame, {"enabled": False, "rows_before": int(len(frame)), "rows_after": int(len(frame))}
    if date_col and date_col in frame.columns:
        ordered = frame.assign(__sort_date__=pd.to_datetime(frame[date_col], errors="coerce"))
        limited = ordered.sort_values("__sort_date__", kind="mergesort").tail(max_rows)
        limited = limited.drop(columns=["__sort_date__"]).reset_index(drop=True)
        strategy = "latest_rows_by_date"
    else:
        limited = frame.tail(max_rows).reset_index(drop=True)
        strategy = "latest_rows_after_feature_sort"
    return limited, {
        "enabled": True,
        "strategy": strategy,
        "rows_before": int(len(frame)),
        "rows_after": int(len(limited)),
    }


def _evaluation_payload(evaluations) -> list[dict]:
    return [
        {"model": item.name, "mae": item.mae, "rmse": item.rmse, "r2": item.r2}
        for item in evaluations
    ]


def train_heatguard_model(
    dataframe: Optional[pd.DataFrame] = None,
    *,
    csv_path: str | Path | None = None,
    target_col: Optional[str] = None,
    date_col: Optional[str] = None,
    location_col: Optional[str] = None,
    forecast_horizon: int = DEFAULT_FORECAST_HORIZON,
    model_path: Path = MODEL_PATH,
    metrics_path: Path = METRICS_PATH,
    features_path: Path = FEATURES_PATH,
    random_state: int = RANDOM_STATE,
    max_training_rows: int | None = 250_000,
) -> TrainResult:
    ensure_directories()

    if dataframe is None:
        df, data_meta = load_available_dataset(csv_path)
    else:
        df = augment_competition_columns(dataframe)
        demo_marker = "data_mode" in df.columns and df["data_mode"].astype(str).str.contains("demo", case=False).any()
        data_meta = {
            "is_demo": bool(demo_marker),
            "source_name": "Synthetic demo dataframe" if demo_marker else "Uploaded dataframe",
            "source_path": None,
        }

    target_col = target_col or detect_target_column(df)
    date_col = date_col or detect_date_column(df)
    location_col = location_col or detect_location_column(df)

    supervised = build_supervised_frame(
        df,
        target_col=target_col,
        date_col=date_col,
        location_col=location_col,
        horizon=forecast_horizon,
    )
    supervised_frame, sampling_meta = _limit_training_rows(
        supervised.frame,
        max_training_rows,
        date_col=supervised.date_column,
    )

    X_train, X_val, y_train, y_val, split_type = _split_supervised_frame(
        supervised_frame,
        feature_names=supervised.feature_names,
        date_col=supervised.date_column,
        random_state=random_state,
    )

    best_eval, evaluations = train_and_select_model(
        X_train,
        y_train,
        X_val,
        y_val,
        random_state=random_state,
    )

    full_X = coerce_numeric_frame(supervised_frame, supervised.feature_names)
    full_y = pd.to_numeric(supervised_frame["__future_target__"], errors="coerce")
    full_mask = full_y.notna()
    primary_model = refit_model(best_eval.model, full_X.loc[full_mask], full_y.loc[full_mask])

    horizon_models: dict[str, object] = {}
    horizon_metrics: dict[str, dict] = {}
    for lead_time in range(1, int(forecast_horizon) + 1):
        horizon_frame = build_supervised_frame(
            df,
            target_col=supervised.target_name,
            date_col=supervised.date_column,
            location_col=supervised.location_column,
            horizon=lead_time,
            feature_names=supervised.feature_names,
        )
        lead_frame, _ = _limit_training_rows(
            horizon_frame.frame,
            max_training_rows,
            date_col=supervised.date_column,
        )
        lead_X = coerce_numeric_frame(lead_frame, supervised.feature_names)
        lead_y = pd.to_numeric(lead_frame["__future_target__"], errors="coerce")
        lead_mask = lead_y.notna()
        horizon_models[str(lead_time)] = refit_model(
            best_eval.model,
            lead_X.loc[lead_mask],
            lead_y.loc[lead_mask],
        )
        horizon_metrics[str(lead_time)] = {"training_rows": int(lead_mask.sum())}

    metrics = {
        "selected_model": best_eval.name,
        "forecast_horizon_days": int(forecast_horizon),
        "primary_horizon_metrics": {
            "mae": best_eval.mae,
            "rmse": best_eval.rmse,
            "r2": best_eval.r2,
        },
        "candidate_metrics": _evaluation_payload(evaluations),
        "validation_approach": split_type,
        "rows_used": int(len(supervised_frame)),
        "rows_available_after_feature_engineering": int(len(supervised.frame)),
        "training_sampling": sampling_meta,
        "features_used": int(len(supervised.feature_names)),
        "target_column": supervised.target_name,
        "date_column": supervised.date_column,
        "location_column": supervised.location_column,
        "data_source": data_meta,
        "leakage_note": (
            "Direct lead-time labels are shifted into the future; lag and rolling features are "
            "computed from current or previous observations only."
        ),
        "horizon_models": horizon_metrics,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    artifact = {
        "model_type": best_eval.name,
        "primary_model": primary_model,
        "horizon_models": horizon_models,
        "feature_names": supervised.feature_names,
        "target_col": supervised.target_name,
        "date_col": supervised.date_column,
        "location_col": supervised.location_column,
        "forecast_horizon": int(forecast_horizon),
        "metrics": metrics,
        "data_is_demo": bool(data_meta.get("is_demo", False)),
        "version": "1.0.0",
    }

    save_model_artifact(artifact, model_path)
    write_json(metrics_path, metrics)
    write_json(
        features_path,
        {
            "feature_names": supervised.feature_names,
            "target_col": supervised.target_name,
            "date_col": supervised.date_column,
            "location_col": supervised.location_column,
            "forecast_horizon": int(forecast_horizon),
        },
    )

    return TrainResult(
        artifact=artifact,
        metrics=metrics,
        model_path=model_path,
        metrics_path=metrics_path,
        features_path=features_path,
    )
