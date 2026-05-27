from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
ASSETS_DIR = ROOT_DIR / "assets"

MODEL_PATH = MODELS_DIR / "heatguard_model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
FEATURES_PATH = MODELS_DIR / "features.json"

DEFAULT_FORECAST_HORIZON = 10
RANDOM_STATE = 42

TARGET_CANDIDATES = [
    "wet_bulb",
    "wetbulb",
    "wet bulb",
    "wbt",
    "wbgt",
    "target",
    "y",
]

DATE_CANDIDATES = [
    "date",
    "datetime",
    "timestamp",
    "time",
    "observed_at",
    "observation_date",
]

LOCATION_CANDIDATES = [
    "location_id",
    "station",
    "station_id",
    "location",
    "city",
    "district",
    "site",
    "region",
]

LATITUDE_CANDIDATES = ["lat", "latitude"]
LONGITUDE_CANDIDATES = ["lon", "lng", "long", "longitude"]

RISK_LEVELS = [
    {"name": "Low", "min": float("-inf"), "max": 24.0, "color": "#20D998"},
    {"name": "Moderate", "min": 24.0, "max": 27.0, "color": "#67F3C2"},
    {"name": "High", "min": 27.0, "max": 30.0, "color": "#F4B547"},
    {"name": "Extreme", "min": 30.0, "max": float("inf"), "color": "#F36B6B"},
]

AUDIENCES = [
    "Citizens",
    "Hospitals",
    "Schools/colleges",
    "Outdoor workers",
    "Local authorities",
]


def ensure_directories() -> None:
    """Create runtime directories used by the app and training scripts."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
