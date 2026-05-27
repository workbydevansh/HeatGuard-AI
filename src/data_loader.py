from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import DATA_DIR, RANDOM_STATE, ensure_directories
from .preprocessing import augment_competition_columns
from .utils import approximate_wet_bulb_celsius


def list_csv_files(data_dir: Path = DATA_DIR) -> list[Path]:
    ensure_directories()
    return sorted(path for path in data_dir.rglob("*.csv") if path.is_file())


def default_csv_path(data_dir: Path = DATA_DIR) -> Optional[Path]:
    csv_files = list_csv_files(data_dir)
    if not csv_files:
        return None
    for path in csv_files:
        if path.name.lower() == "train.csv":
            return path
    return csv_files[0]


def load_csv(path: str | Path) -> pd.DataFrame:
    return augment_competition_columns(pd.read_csv(path))


def load_available_dataset(path: str | Path | None = None) -> tuple[pd.DataFrame, dict]:
    """Load real data if available; otherwise return a labeled demo dataset."""
    ensure_directories()
    selected_path = Path(path) if path else default_csv_path()
    if selected_path and selected_path.exists():
        df = load_csv(selected_path)
        return df, {
            "is_demo": False,
            "source_name": selected_path.name,
            "source_path": str(selected_path),
        }

    df = generate_synthetic_demo_data()
    return df, {
        "is_demo": True,
        "source_name": "Synthetic demo climate data",
        "source_path": None,
    }


def generate_synthetic_demo_data(
    *,
    start_date: str = "2024-01-01",
    periods: int = 850,
    seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Create a small, realistic-looking climate dataset for UI testing only."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start_date, periods=periods, freq="D")
    stations = [
        ("Lucknow Urban", 26.8467, 80.9462, 1.2),
        ("Lucknow Airport", 26.7606, 80.8893, 0.6),
        ("Kanpur Belt", 26.4499, 80.3319, 1.0),
        ("Gomti Floodplain", 26.8858, 81.0261, 1.5),
    ]

    records: list[dict] = []
    for station, lat, lon, heat_bias in stations:
        station_noise = rng.normal(0, 0.45, len(dates))
        for i, date in enumerate(dates):
            day = int(date.dayofyear)
            summer_wave = np.sin(2 * np.pi * (day - 108) / 365.25)
            monsoon_wave = np.sin(2 * np.pi * (day - 175) / 365.25)

            temperature = (
                29.0
                + 8.2 * summer_wave
                + heat_bias
                + station_noise[i]
                + rng.normal(0, 1.15)
            )
            humidity = (
                56.0
                + 20.0 * monsoon_wave
                - 7.0 * summer_wave
                + rng.normal(0, 5.0)
            )
            humidity = float(np.clip(humidity, 25, 96))
            pressure = 1006.0 - 3.5 * summer_wave + rng.normal(0, 2.1)
            wind = float(np.clip(2.3 + rng.normal(0, 0.85) - 0.35 * summer_wave, 0.2, 8.0))
            rainfall = float(max(0, rng.gamma(1.2, 6.0) * max(0, monsoon_wave)))
            solar = float(np.clip(560 + 175 * summer_wave - 70 * monsoon_wave + rng.normal(0, 35), 120, 880))
            wet_bulb = approximate_wet_bulb_celsius(temperature, humidity) + rng.normal(0, 0.35)

            records.append(
                {
                    "date": date.date().isoformat(),
                    "station": station,
                    "latitude": lat,
                    "longitude": lon,
                    "temperature_c": round(float(temperature), 2),
                    "relative_humidity": round(float(humidity), 2),
                    "pressure_hpa": round(float(pressure), 2),
                    "wind_speed_ms": round(float(wind), 2),
                    "rainfall_mm": round(rainfall, 2),
                    "solar_radiation_wm2": round(solar, 2),
                    "wet_bulb_c": round(float(wet_bulb), 2),
                    "data_mode": "demo_fallback",
                }
            )

    return pd.DataFrame.from_records(records)
