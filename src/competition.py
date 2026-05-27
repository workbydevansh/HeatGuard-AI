from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DATA_DIR
from .data_loader import load_csv
from .feature_engineering import build_feature_frame
from .preprocessing import coerce_numeric_frame
from .utils import classify_wbt, risk_score


def find_competition_dir(data_dir: str | Path | None = None) -> Path | None:
    root = Path(data_dir) if data_dir else DATA_DIR
    if (root / "train.csv").exists() and (root / "test.csv").exists():
        return root
    candidates = []
    for train_path in DATA_DIR.rglob("train.csv"):
        folder = train_path.parent
        if (folder / "test.csv").exists() and (folder / "sample_submission.csv").exists():
            candidates.append(folder)
    return sorted(candidates, key=lambda path: len(str(path)))[0] if candidates else None


def competition_file_status(competition_dir: Path | None) -> dict:
    if competition_dir is None:
        return {"available": False}
    files = {}
    for name in ("train.csv", "test.csv", "context.csv", "sample_submission.csv"):
        path = competition_dir / name
        files[name] = {
            "exists": path.exists(),
            "size_mb": round(path.stat().st_size / 1024 / 1024, 2) if path.exists() else 0,
        }
    return {"available": True, "folder": str(competition_dir), "files": files}


def load_context_and_test(competition_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    context = load_csv(competition_dir / "context.csv") if (competition_dir / "context.csv").exists() else pd.DataFrame()
    test = load_csv(competition_dir / "test.csv")
    return context, test


def make_location_forecast(
    artifact: dict,
    context_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    location_id: str,
    start_day_index: int = 0,
    horizon: int = 10,
) -> dict:
    selected_test = test_df[test_df["location_id"].astype(str) == str(location_id)].copy()
    selected_context = context_df[context_df["location_id"].astype(str) == str(location_id)].copy()
    if selected_test.empty:
        raise ValueError(f"No test rows found for location {location_id}.")

    selected_test["__is_test__"] = True
    if selected_context.empty:
        combined = selected_test.copy()
    else:
        selected_context["__is_test__"] = False
        combined = pd.concat([selected_context, selected_test], ignore_index=True, sort=False)

    features = build_feature_frame(
        combined,
        target_col=artifact.get("target_col"),
        date_col=artifact.get("date_col"),
        location_col=artifact.get("location_col"),
        feature_names=list(artifact["feature_names"]),
    )
    feature_frame = features.frame[features.frame["__is_test__"].fillna(False)].copy()
    feature_frame = feature_frame.sort_values("day_index" if "day_index" in feature_frame.columns else "row_day_index")

    if "day_index" in feature_frame.columns:
        available_days = pd.to_numeric(feature_frame["day_index"], errors="coerce")
        if start_day_index not in set(available_days.dropna().astype(int)):
            start_day_index = int(available_days.dropna().astype(int).min())
        anchor = feature_frame[available_days.astype("Int64") == int(start_day_index)].head(1)
    else:
        anchor = feature_frame.head(1)

    if anchor.empty:
        anchor = feature_frame.head(1)

    X_anchor = coerce_numeric_frame(anchor, list(artifact["feature_names"]))
    rmse = artifact.get("metrics", {}).get("primary_horizon_metrics", {}).get("rmse") or 1.5

    last_context_date = None
    if not selected_context.empty and "date" in selected_context.columns:
        last_context_date = pd.to_datetime(selected_context["date"], errors="coerce").max()

    rows = []
    for day in range(1, min(horizon, int(artifact.get("forecast_horizon", horizon))) + 1):
        model = artifact.get("horizon_models", {}).get(str(day)) or artifact.get("primary_model")
        predicted = float(model.predict(X_anchor)[0])
        if pd.notna(last_context_date):
            forecast_date = (last_context_date + pd.Timedelta(days=int(start_day_index) + day)).date().isoformat()
        else:
            forecast_date = f"target_day_{day}"
        rows.append(
            {
                "day": day,
                "date": forecast_date,
                "predicted_wbt": predicted,
                "risk_category": classify_wbt(predicted),
                "risk_score": risk_score(predicted),
                "lower_bound": predicted - float(rmse),
                "upper_bound": predicted + float(rmse),
                "row_id": str(anchor["row_id"].iloc[0]) if "row_id" in anchor.columns else "",
            }
        )

    forecast = pd.DataFrame(rows)
    peak = forecast.loc[forecast["predicted_wbt"].idxmax()].to_dict()
    return {
        "forecast": forecast,
        "peak": peak,
        "anchor_row_id": str(anchor["row_id"].iloc[0]) if "row_id" in anchor.columns else "",
        "history": selected_context.sort_values("date") if "date" in selected_context.columns else selected_context,
        "available_day_indices": sorted(pd.to_numeric(selected_test.get("day_index", pd.Series(dtype=int)), errors="coerce").dropna().astype(int).unique().tolist()),
    }

