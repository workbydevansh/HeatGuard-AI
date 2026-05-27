from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DATA_DIR, DEFAULT_FORECAST_HORIZON
from src.data_loader import load_csv
from src.feature_engineering import build_feature_frame
from src.model import load_model_artifact
from src.preprocessing import augment_competition_columns, coerce_numeric_frame
from src.train import train_heatguard_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a Kaggle submission for HeatGuard AI.")
    parser.add_argument("--data-dir", type=str, default=None, help="Folder containing train.csv, test.csv, context.csv, and sample_submission.csv.")
    parser.add_argument("--output", type=str, default="submission.csv", help="Output CSV path.")
    parser.add_argument("--horizon", type=int, default=DEFAULT_FORECAST_HORIZON, help="Forecast horizon in days.")
    parser.add_argument("--target-col", type=str, default="WBT", help="Wet-bulb target column.")
    parser.add_argument("--max-training-rows", type=int, default=250000, help="Training row cap used if a model must be trained.")
    parser.add_argument("--force-train", action="store_true", help="Retrain even if a saved model exists.")
    return parser.parse_args()


def find_competition_dir(data_dir: str | None = None) -> Path:
    root = Path(data_dir) if data_dir else DATA_DIR
    if (root / "train.csv").exists() and (root / "test.csv").exists():
        return root

    candidates = []
    for train_path in DATA_DIR.rglob("train.csv"):
        folder = train_path.parent
        if (folder / "test.csv").exists() and (folder / "sample_submission.csv").exists():
            candidates.append(folder)
    if candidates:
        return sorted(candidates, key=lambda path: len(str(path)))[0]
    raise FileNotFoundError("Could not find a folder containing train.csv, test.csv, and sample_submission.csv.")


def ensure_artifact(args: argparse.Namespace, competition_dir: Path) -> dict:
    artifact = None if args.force_train else load_model_artifact()
    if artifact and artifact.get("target_col") == args.target_col and artifact.get("forecast_horizon", 0) >= args.horizon:
        return artifact

    print("Training HeatGuard model on competition train.csv...")
    result = train_heatguard_model(
        csv_path=competition_dir / "train.csv",
        target_col=args.target_col,
        date_col="date",
        location_col="location_id",
        forecast_horizon=args.horizon,
        max_training_rows=args.max_training_rows or None,
    )
    primary = result.metrics["primary_horizon_metrics"]
    print(f"Selected model: {result.metrics['selected_model']}")
    print(f"Validation RMSE: {primary['rmse']:.4f}")
    return result.artifact


def build_submission_features(artifact: dict, competition_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    test_df = load_csv(competition_dir / "test.csv")
    test_df["__is_test__"] = True

    context_path = competition_dir / "context.csv"
    if context_path.exists():
        context_df = load_csv(context_path)
        context_df["__is_test__"] = False
        combined = pd.concat([context_df, test_df], ignore_index=True, sort=False)
    else:
        combined = test_df.copy()

    combined = augment_competition_columns(combined)
    feature_result = build_feature_frame(
        combined,
        target_col=artifact.get("target_col"),
        date_col=artifact.get("date_col"),
        location_col=artifact.get("location_col"),
        feature_names=list(artifact["feature_names"]),
    )
    feature_frame = feature_result.frame[feature_result.frame["__is_test__"].fillna(False)].copy()
    feature_frame = feature_frame.set_index("row_id").loc[test_df["row_id"]].reset_index()
    return test_df, feature_frame


def create_kaggle_submission(args: argparse.Namespace) -> Path:
    competition_dir = find_competition_dir(args.data_dir)
    print(f"Using competition data folder: {competition_dir}")
    artifact = ensure_artifact(args, competition_dir)

    test_df, feature_frame = build_submission_features(artifact, competition_dir)
    X_test = coerce_numeric_frame(feature_frame, list(artifact["feature_names"]))

    submission = pd.DataFrame({"row_id": test_df["row_id"].astype(str)})
    for day in range(1, args.horizon + 1):
        model = artifact.get("horizon_models", {}).get(str(day)) or artifact.get("primary_model")
        if model is None:
            raise RuntimeError(f"No model available for horizon day {day}.")
        submission[f"target_day_{day}"] = model.predict(X_test)

    sample_path = competition_dir / "sample_submission.csv"
    if sample_path.exists():
        sample_cols = list(pd.read_csv(sample_path, nrows=1).columns)
        for col in sample_cols:
            if col not in submission.columns and col != "row_id":
                submission[col] = submission[f"target_day_{args.horizon}"]
        submission = submission[sample_cols]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    args = parse_args()
    output_path = create_kaggle_submission(args)
    print(f"Kaggle submission saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
