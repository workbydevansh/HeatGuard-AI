from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train import train_heatguard_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the HeatGuard AI wet-bulb forecasting model.")
    parser.add_argument("--csv", type=str, default=None, help="Optional path to a CSV file. Defaults to data/train.csv or demo fallback.")
    parser.add_argument("--target-col", type=str, default=None, help="Wet-bulb target column if auto-detection fails.")
    parser.add_argument("--date-col", type=str, default=None, help="Date column for chronological validation.")
    parser.add_argument("--location-col", type=str, default=None, help="Station/location column for grouped lags.")
    parser.add_argument("--horizon", type=int, default=10, help="Forecast horizon in days.")
    parser.add_argument(
        "--max-training-rows",
        type=int,
        default=250000,
        help="Resource-aware cap for large datasets. Use 0 to train on all engineered rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_heatguard_model(
        csv_path=args.csv,
        target_col=args.target_col,
        date_col=args.date_col,
        location_col=args.location_col,
        forecast_horizon=args.horizon,
        max_training_rows=args.max_training_rows or None,
    )
    metrics = result.metrics
    primary = metrics["primary_horizon_metrics"]
    print("HeatGuard AI training complete")
    print(f"Selected model: {metrics['selected_model']}")
    print(f"Validation: {metrics['validation_approach']}")
    print(f"MAE: {primary['mae']:.4f}")
    print(f"RMSE: {primary['rmse']:.4f}")
    print(f"R2: {primary['r2']:.4f}")
    print(f"Model saved to: {result.model_path}")
    print(f"Metrics saved to: {result.metrics_path}")
    print(f"Features saved to: {result.features_path}")


if __name__ == "__main__":
    main()
