from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import DEFAULT_FORECAST_HORIZON, MODEL_PATH
from .explainability import get_feature_importance
from .feature_engineering import build_feature_frame
from .model import load_model_artifact
from .preprocessing import coerce_numeric_frame
from .utils import (
    approximate_wet_bulb_celsius,
    classify_wbt,
    confidence_from_rmse,
    reliability_label,
    risk_score,
)


def load_or_none(model_path: Path = MODEL_PATH) -> dict | None:
    return load_model_artifact(model_path)


def make_manual_climate_frame(
    *,
    temperature_c: float,
    relative_humidity: float,
    pressure_hpa: float = 1006.0,
    wind_speed_ms: float = 2.4,
    rainfall_mm: float = 0.0,
    target_col: str = "wet_bulb_c",
    days: int = 35,
) -> pd.DataFrame:
    today = pd.Timestamp(date.today())
    rng = np.random.default_rng(7)
    records = []
    base_wbt = approximate_wet_bulb_celsius(temperature_c, relative_humidity)
    for offset in range(days, 0, -1):
        drift = (days - offset) / max(days, 1)
        temp = temperature_c - 1.2 + 1.2 * drift + rng.normal(0, 0.25)
        rh = np.clip(relative_humidity - 2.5 + 2.5 * drift + rng.normal(0, 1.5), 5, 100)
        wbt = approximate_wet_bulb_celsius(temp, rh)
        records.append(
            {
                "date": (today - pd.Timedelta(days=offset)).date().isoformat(),
                "station": "Manual entry",
                "temperature_c": round(float(temp), 2),
                "relative_humidity": round(float(rh), 2),
                "pressure_hpa": round(float(pressure_hpa + rng.normal(0, 0.5)), 2),
                "wind_speed_ms": round(float(max(0.1, wind_speed_ms + rng.normal(0, 0.12))), 2),
                "rainfall_mm": round(float(max(0, rainfall_mm + rng.normal(0, 0.2))), 2),
                "solar_radiation_wm2": 610.0,
                target_col: round(float(wbt), 2),
            }
        )

    records.append(
        {
            "date": today.date().isoformat(),
            "station": "Manual entry",
            "temperature_c": float(temperature_c),
            "relative_humidity": float(relative_humidity),
            "pressure_hpa": float(pressure_hpa),
            "wind_speed_ms": float(wind_speed_ms),
            "rainfall_mm": float(rainfall_mm),
            "solar_radiation_wm2": 625.0,
            target_col: round(float(base_wbt), 2),
        }
    )
    return pd.DataFrame(records)


def predict_wbt(
    input_df: pd.DataFrame,
    *,
    artifact: Optional[dict] = None,
    model_path: Path = MODEL_PATH,
    horizon: int = DEFAULT_FORECAST_HORIZON,
) -> dict:
    artifact = artifact or load_or_none(model_path)
    if artifact is None:
        raise FileNotFoundError(
            "No trained HeatGuard model found. Train the model from the dashboard or run scripts/train_model.py."
        )

    feature_names = list(artifact["feature_names"])
    feature_result = build_feature_frame(
        input_df,
        target_col=artifact.get("target_col"),
        date_col=artifact.get("date_col"),
        location_col=artifact.get("location_col"),
        feature_names=feature_names,
    )

    if feature_result.frame.empty:
        raise ValueError("No rows available for prediction after feature engineering.")

    latest = feature_result.frame.tail(1).copy()
    X_latest = coerce_numeric_frame(latest, feature_names)

    metrics = artifact.get("metrics", {}).get("primary_horizon_metrics", {})
    rmse = metrics.get("rmse")
    confidence = confidence_from_rmse(rmse, is_demo=bool(artifact.get("data_is_demo", False)))
    reliability = reliability_label(confidence)

    raw_latest_date = None
    if artifact.get("date_col") and artifact.get("date_col") in latest.columns:
        raw_latest_date = pd.to_datetime(latest[artifact["date_col"]].iloc[0], errors="coerce")
    if raw_latest_date is None or pd.isna(raw_latest_date):
        raw_latest_date = pd.Timestamp(date.today())

    forecast_rows = []
    max_horizon = min(int(horizon), int(artifact.get("forecast_horizon", horizon)))
    for day_ahead in range(1, max_horizon + 1):
        model = artifact.get("horizon_models", {}).get(str(day_ahead)) or artifact.get("primary_model")
        prediction = float(model.predict(X_latest)[0])
        forecast_date = raw_latest_date + pd.Timedelta(days=day_ahead)
        forecast_rows.append(
            {
                "day": day_ahead,
                "date": forecast_date.date().isoformat(),
                "predicted_wbt": prediction,
                "risk_category": classify_wbt(prediction),
                "risk_score": risk_score(prediction),
                "confidence_pct": confidence,
                "reliability": reliability,
                "lower_bound": prediction - float(rmse or 1.5),
                "upper_bound": prediction + float(rmse or 1.5),
            }
        )

    forecast = pd.DataFrame(forecast_rows)
    peak_idx = forecast["predicted_wbt"].idxmax()
    peak = forecast.loc[peak_idx].to_dict()
    importance = get_feature_importance(artifact, top_n=10)

    return {
        "forecast": forecast,
        "peak": peak,
        "confidence_pct": confidence,
        "reliability": reliability,
        "feature_importance": importance,
        "latest_features": X_latest,
    }

