"""Data cleaning and preprocessing pipeline builders."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.feature_engineering import engineer_features, get_model_feature_columns
from src.utils import PROCESSED_DATA_PATH, TARGET_COLUMN, load_raw_data


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureGroups:
    """Container for numerical and categorical feature names."""

    numerical_features: list[str]
    categorical_features: list[str]


def remove_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows while preserving the first occurrence."""
    duplicate_count = int(data.duplicated().sum())
    if duplicate_count:
        logger.info("Removing %s duplicate rows", duplicate_count)
    return data.drop_duplicates().reset_index(drop=True)


def standardize_categorical_values(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize categorical values for consistent downstream encoding."""
    transformed = data.copy()
    categorical_columns = transformed.select_dtypes(include=["object", "category"]).columns

    for column in categorical_columns:
        transformed[column] = (
            transformed[column]
            .astype(str)
            .str.strip()
            .str.replace("_", " ", regex=False)
            .str.title()
        )

    return transformed


def cap_outliers_iqr(
    data: pd.DataFrame,
    columns: list[str],
    multiplier: float = 1.5,
) -> pd.DataFrame:
    """Cap numerical outliers using the interquartile range rule."""
    transformed = data.copy()

    for column in columns:
        if column not in transformed.columns:
            continue

        q1 = transformed[column].quantile(0.25)
        q3 = transformed[column].quantile(0.75)
        iqr = q3 - q1

        if pd.isna(iqr) or iqr == 0:
            continue

        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
        transformed[column] = transformed[column].clip(lower_bound, upper_bound)

    return transformed


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic cleaning rules before model preprocessing."""
    cleaned = remove_duplicates(data)
    cleaned = standardize_categorical_values(cleaned)

    numeric_columns = [
        "year",
        "present_price",
        "selling_price",
        "driven_km",
        "owner",
    ]
    for column in numeric_columns:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned = cleaned.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
    cleaned = cleaned[cleaned["present_price"] > 0]
    cleaned = cleaned[cleaned["driven_km"] >= 0]
    cleaned = cleaned[cleaned["year"].between(1990, 2026)]

    outlier_columns = ["present_price", "selling_price", "driven_km"]
    cleaned = cap_outliers_iqr(cleaned, outlier_columns, multiplier=1.5)
    return cleaned.reset_index(drop=True)


def identify_feature_groups(data: pd.DataFrame, feature_columns: list[str]) -> FeatureGroups:
    """Split model features into numerical and categorical groups."""
    numerical_features = [
        column
        for column in feature_columns
        if pd.api.types.is_numeric_dtype(data[column])
    ]
    categorical_features = [
        column
        for column in feature_columns
        if column not in numerical_features
    ]
    return FeatureGroups(
        numerical_features=numerical_features,
        categorical_features=categorical_features,
    )


def build_preprocessor(feature_groups: FeatureGroups) -> ColumnTransformer:
    """Create a production-ready preprocessing transformer."""
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, feature_groups.numerical_features),
            ("categorical", categorical_pipeline, feature_groups.categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def prepare_model_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
    """Clean data, engineer features, and build the preprocessing object."""
    cleaned = clean_data(data)
    featured = engineer_features(cleaned, include_target_features=True)
    feature_columns = get_model_feature_columns(featured)
    feature_groups = identify_feature_groups(featured, feature_columns)
    preprocessor = build_preprocessor(feature_groups)

    features = featured[feature_columns]
    target = featured[TARGET_COLUMN]
    return features, target, preprocessor


def create_processed_dataset(output_path: str | None = None) -> pd.DataFrame:
    """Create and save the cleaned feature-engineered dataset."""
    raw_data = load_raw_data()
    cleaned = clean_data(raw_data)
    processed = engineer_features(cleaned, include_target_features=True)

    destination = PROCESSED_DATA_PATH if output_path is None else output_path
    processed.to_csv(destination, index=False)
    logger.info("Saved processed dataset to %s", destination)
    return processed


def get_prediction_input_schema() -> dict[str, type]:
    """Return expected API and Streamlit input fields."""
    return {
        "year": int,
        "present_price": float,
        "driven_km": int,
        "fuel_type": str,
        "seller_type": str,
        "transmission": str,
        "owner": int,
    }


def prepare_single_prediction_input(payload: dict[str, object]) -> pd.DataFrame:
    """Convert a user payload into a model-ready dataframe before preprocessing."""
    frame = pd.DataFrame([payload])
    frame = engineer_features(frame, include_target_features=False)
    frame = frame.replace({np.inf: np.nan, -np.inf: np.nan})
    return frame
