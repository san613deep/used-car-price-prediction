"""Shared utilities for the used car price prediction project."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

RAW_DATA_PATH = RAW_DATA_DIR / "car_data.csv"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "car_data_processed.csv"
BEST_MODEL_PATH = MODEL_DIR / "best_model.joblib"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.joblib"
METRICS_PATH = MODEL_DIR / "model_metrics.json"
FEATURE_IMPORTANCE_PATH = MODEL_DIR / "feature_importance.csv"

TARGET_COLUMN = "selling_price"
RANDOM_STATE = 42
CURRENT_YEAR = 2026

COLUMN_RENAME_MAP = {
    "Car_Name": "car_name",
    "Year": "year",
    "Selling_Price": "selling_price",
    "Present_Price": "present_price",
    "Driven_kms": "driven_km",
    "Fuel_Type": "fuel_type",
    "Selling_type": "seller_type",
    "Seller_Type": "seller_type",
    "Transmission": "transmission",
    "Owner": "owner",
}

REQUIRED_RAW_COLUMNS = {
    "year",
    "selling_price",
    "present_price",
    "driven_km",
    "fuel_type",
    "seller_type",
    "transmission",
    "owner",
}

MODEL_INPUT_COLUMNS = [
    "year",
    "present_price",
    "driven_km",
    "fuel_type",
    "seller_type",
    "transmission",
    "owner",
]

LUXURY_BRANDS = {
    "audi",
    "bmw",
    "mercedes",
    "mercedes-benz",
    "volvo",
    "jaguar",
    "land rover",
    "lexus",
    "porsche",
}


def configure_logging(level: int = logging.INFO) -> None:
    """Configure project-wide logging with a compact production-friendly format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def ensure_directories() -> None:
    """Create required project directories when they do not already exist."""
    for directory in (RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def normalize_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the dataframe with consistent snake_case column names."""
    renamed = data.rename(columns=COLUMN_RENAME_MAP).copy()
    renamed.columns = [
        column.strip().lower().replace(" ", "_") for column in renamed.columns
    ]
    return renamed


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw used car dataset and normalize its column names."""
    if not path.exists():
        fallback_path = PROJECT_ROOT.parent / "car data.csv"
        if fallback_path.exists():
            path = fallback_path
        else:
            raise FileNotFoundError(
                f"Raw dataset not found at {path}. Place car_data.csv in data/raw/."
            )

    data = pd.read_csv(path)
    data = normalize_columns(data)
    validate_columns(data, REQUIRED_RAW_COLUMNS)
    return data


def validate_columns(data: pd.DataFrame, required_columns: set[str]) -> None:
    """Raise a clear error when required columns are missing."""
    missing_columns = sorted(required_columns.difference(data.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def save_joblib_artifact(artifact: Any, path: Path) -> None:
    """Persist a Python object as a joblib artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)


def load_joblib_artifact(path: Path) -> Any:
    """Load a joblib artifact with a helpful error message."""
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    return joblib.load(path)


def save_json(payload: dict[str, Any], path: Path) -> None:
    """Save a dictionary as a readable JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file as a dictionary."""
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)
