"""Unit tests for used-car feature engineering and data cleaning."""
import pandas as pd

from src.data_preprocessing import clean_data
from src.feature_engineering import engineer_features, get_model_feature_columns


def test_engineering_adds_age_usage_and_depreciation_features():
    frame = pd.DataFrame(
        {
            "Year": [2020],
            "Present_Price": [10.0],
            "Selling_Price": [5.0],
            "Driven_kms": [20_000],
            "Fuel_Type": ["Petrol"],
            "Selling_type": ["Dealer"],
            "Transmission": ["Manual"],
            "Owner": [0],
            "Car_Name": ["Honda City"],
        }
    )

    result = engineer_features(frame, current_year=2026, include_target_features=True)

    assert result.loc[0, "car_age"] == 6
    assert result.loc[0, "km_per_year"] == 20_000 / 6
    assert result.loc[0, "depreciation"] == 5.0
    assert result.loc[0, "brand"] == "honda"


def test_inference_features_do_not_require_target_column():
    frame = pd.DataFrame(
        {
            "year": [2021],
            "present_price": [8.0],
            "driven_km": [30_000],
            "fuel_type": ["Diesel"],
            "seller_type": ["Individual"],
            "transmission": ["Manual"],
            "owner": [1],
        }
    )

    result = engineer_features(frame, current_year=2026, include_target_features=False)
    model_columns = get_model_feature_columns(result)

    assert "car_age" in model_columns
    assert "km_per_year" in model_columns
    assert "selling_price" not in model_columns
    assert "depreciation" not in model_columns
    assert "depreciation_rate" not in model_columns
    assert "log_selling_price" not in model_columns


def test_model_feature_list_excludes_target_derived_features():
    frame = pd.DataFrame(
        {
            "year": [2020],
            "present_price": [10.0],
            "selling_price": [5.0],
            "driven_km": [20_000],
            "fuel_type": ["Petrol"],
            "seller_type": ["Dealer"],
            "transmission": ["Manual"],
            "owner": [0],
        }
    )

    engineered = engineer_features(frame, current_year=2026, include_target_features=True)
    columns = get_model_feature_columns(engineered)

    assert "selling_price" not in columns
    assert "log_selling_price" not in columns
    assert "depreciation" not in columns
    assert "depreciation_rate" not in columns


def test_clean_data_removes_duplicates_and_invalid_rows():
    frame = pd.DataFrame(
        {
            "year": [2020, 2020, 2019, 2021],
            "present_price": [10.0, 10.0, -2.0, 8.0],
            "selling_price": [5.0, 5.0, 1.0, None],
            "driven_km": [20_000, 20_000, 10_000, 5_000],
            "fuel_type": ["Petrol", "Petrol", "Diesel", "Petrol"],
            "seller_type": ["Dealer", "Dealer", "Individual", "Dealer"],
            "transmission": ["Manual", "Manual", "Manual", "Automatic"],
            "owner": [0, 0, 1, 0],
        }
    )

    cleaned = clean_data(frame)

    assert len(cleaned) == 1
    assert cleaned["selling_price"].notna().all()
    assert (cleaned["present_price"] > 0).all()
    assert (cleaned["driven_km"] >= 0).all()
