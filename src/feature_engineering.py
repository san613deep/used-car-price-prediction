"""Feature engineering utilities for used car price prediction."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.utils import CURRENT_YEAR, LUXURY_BRANDS, TARGET_COLUMN, normalize_columns


logger = logging.getLogger(__name__)


def add_car_age(data: pd.DataFrame, current_year: int = CURRENT_YEAR) -> pd.DataFrame:
    """Create vehicle age from manufacturing year."""
    transformed = data.copy()
    transformed["car_age"] = (current_year - transformed["year"]).clip(lower=0)
    return transformed


def add_km_per_year(data: pd.DataFrame) -> pd.DataFrame:
    """Create annual usage intensity from driven kilometers and vehicle age."""
    transformed = data.copy()
    age_denominator = transformed["car_age"].replace(0, 1)
    transformed["km_per_year"] = transformed["driven_km"] / age_denominator
    return transformed


def add_depreciation_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create absolute and percentage depreciation features."""
    transformed = data.copy()

    if TARGET_COLUMN in transformed.columns:
        transformed["depreciation"] = (
            transformed["present_price"] - transformed[TARGET_COLUMN]
        ).clip(lower=0)
        transformed["depreciation_rate"] = np.where(
            transformed["present_price"] > 0,
            transformed["depreciation"] / transformed["present_price"],
            0,
        )
    else:
        transformed["depreciation"] = np.nan
        transformed["depreciation_rate"] = np.nan

    return transformed


def _extract_brand(car_name: object) -> str:
    """Extract a coarse brand token from the raw car name."""
    if pd.isna(car_name):
        return "unknown"

    normalized_name = str(car_name).strip().lower()
    if not normalized_name:
        return "unknown"

    for luxury_brand in LUXURY_BRANDS:
        if luxury_brand in normalized_name:
            return luxury_brand

    return normalized_name.split()[0]


def add_brand_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create brand and luxury-brand indicators when car names are available."""
    transformed = data.copy()

    if "car_name" not in transformed.columns:
        transformed["brand"] = "unknown"
    else:
        transformed["brand"] = transformed["car_name"].apply(_extract_brand)

    transformed["is_luxury_brand"] = transformed["brand"].isin(LUXURY_BRANDS).astype(int)
    return transformed


def add_vehicle_age_bucket(data: pd.DataFrame) -> pd.DataFrame:
    """Bucket vehicles into interpretable age groups."""
    transformed = data.copy()
    transformed["vehicle_age_bucket"] = pd.cut(
        transformed["car_age"],
        bins=[-1, 2, 5, 8, 12, np.inf],
        labels=["0-2 years", "3-5 years", "6-8 years", "9-12 years", "12+ years"],
    ).astype(str)
    return transformed


def add_price_segment(data: pd.DataFrame) -> pd.DataFrame:
    """Create business-friendly price segments from present market price."""
    transformed = data.copy()
    transformed["price_segment"] = pd.cut(
        transformed["present_price"],
        bins=[-np.inf, 4, 8, 15, np.inf],
        labels=["budget", "mid_market", "premium", "luxury"],
    ).astype(str)
    return transformed


def add_log_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create log-transformed numerical features to reduce skew."""
    transformed = data.copy()
    transformed["log_present_price"] = np.log1p(transformed["present_price"])
    transformed["log_driven_km"] = np.log1p(transformed["driven_km"])

    if TARGET_COLUMN in transformed.columns:
        transformed["log_selling_price"] = np.log1p(transformed[TARGET_COLUMN])

    return transformed


def engineer_features(
    data: pd.DataFrame,
    current_year: int = CURRENT_YEAR,
    include_target_features: bool = True,
) -> pd.DataFrame:
    """Apply the complete feature engineering workflow.

    Target-derived features are useful for EDA and offline analysis, but they are
    excluded from model inputs to prevent leakage during training and inference.
    """
    logger.info("Starting feature engineering for %s rows", len(data))
    transformed = normalize_columns(data)
    transformed = add_car_age(transformed, current_year=current_year)
    transformed = add_km_per_year(transformed)
    transformed = add_brand_features(transformed)
    transformed = add_vehicle_age_bucket(transformed)
    transformed = add_price_segment(transformed)
    transformed = add_log_features(transformed)

    if include_target_features:
        transformed = add_depreciation_features(transformed)

    logger.info("Completed feature engineering with %s columns", transformed.shape[1])
    return transformed


def get_model_feature_columns(data: pd.DataFrame) -> list[str]:
    """Return leakage-safe feature columns for model training and prediction."""
    excluded_columns = {
        TARGET_COLUMN,
        "log_selling_price",
        "depreciation",
        "depreciation_rate",
        "car_name",
    }
    return [column for column in data.columns if column not in excluded_columns]
