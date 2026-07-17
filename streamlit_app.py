"""Streamlit application for used car price prediction."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data_preprocessing import prepare_single_prediction_input  # noqa: E402
from src.predict import PredictionService  # noqa: E402
from src.utils import FEATURE_IMPORTANCE_PATH, METRICS_PATH, load_json  # noqa: E402


st.set_page_config(
    page_title="Used Car Price Predictor",
    page_icon="car",
    layout="wide",
)


@st.cache_resource
def load_prediction_service() -> PredictionService | None:
    """Load prediction artifacts once per Streamlit session."""
    try:
        return PredictionService.from_artifacts()
    except FileNotFoundError:
        return None


@st.cache_data
def load_feature_importance() -> pd.DataFrame:
    """Load saved feature importance if available."""
    if FEATURE_IMPORTANCE_PATH.exists():
        return pd.read_csv(FEATURE_IMPORTANCE_PATH)
    return pd.DataFrame(columns=["feature", "importance"])


@st.cache_data
def load_model_metrics() -> dict[str, Any]:
    """Load model metrics if available."""
    if METRICS_PATH.exists():
        return load_json(METRICS_PATH)
    return {}


def build_sidebar_payload() -> dict[str, object]:
    """Collect vehicle inputs from the sidebar."""
    st.sidebar.header("Vehicle Details")
    return {
        "year": st.sidebar.slider("Manufacturing Year", 1990, 2026, 2017),
        "present_price": st.sidebar.number_input(
            "Current Showroom Price (INR lakhs)",
            min_value=0.1,
            max_value=100.0,
            value=8.5,
            step=0.1,
        ),
        "driven_km": st.sidebar.number_input(
            "Kilometers Driven",
            min_value=0,
            max_value=500_000,
            value=35_000,
            step=1_000,
        ),
        "fuel_type": st.sidebar.selectbox(
            "Fuel Type",
            options=["Petrol", "Diesel", "Cng"],
            index=0,
        ),
        "seller_type": st.sidebar.selectbox(
            "Seller Type",
            options=["Dealer", "Individual"],
            index=0,
        ),
        "transmission": st.sidebar.selectbox(
            "Transmission",
            options=["Manual", "Automatic"],
            index=0,
        ),
        "owner": st.sidebar.selectbox(
            "Previous Owners",
            options=[0, 1, 2, 3],
            index=0,
        ),
    }


def render_prediction_panel(
    prediction_service: PredictionService | None,
    payload: dict[str, object],
) -> float | None:
    """Render the prediction result panel."""
    st.subheader("Price Prediction")

    if prediction_service is None:
        st.warning("Model artifacts are missing. Run `python src/train.py` first.")
        return None

    if not st.sidebar.button("Predict Price", type="primary", use_container_width=True):
        st.info("Adjust the sidebar inputs and click Predict Price.")
        return None

    predicted_price = prediction_service.predict(payload)
    st.metric(
        label="Estimated Selling Price",
        value=f"INR {predicted_price:,.2f} lakhs",
        delta="Model-based fair price estimate",
    )

    col_a, col_b, col_c = st.columns(3)
    car_age = 2026 - int(payload["year"])
    km_per_year = float(payload["driven_km"]) / max(car_age, 1)
    depreciation_estimate = float(payload["present_price"]) - predicted_price

    col_a.metric("Vehicle Age", f"{car_age} years")
    col_b.metric("KM per Year", f"{km_per_year:,.0f}")
    col_c.metric("Estimated Depreciation", f"INR {depreciation_estimate:,.2f} lakhs")
    return predicted_price


def render_feature_importance(feature_importance: pd.DataFrame) -> None:
    """Render model feature importance."""
    st.subheader("Feature Importance")

    if feature_importance.empty:
        st.info("Feature importance will appear after model training.")
        return

    top_features = feature_importance.head(15).sort_values("importance")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top_features["feature"], top_features["importance"], color="#2f6f73")
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    ax.set_title("Top Drivers of Predicted Price")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)


def render_model_performance(metrics: dict[str, Any]) -> None:
    """Render model comparison and best-model metrics."""
    st.subheader("Model Performance")

    if not metrics:
        st.info("Model performance metrics will appear after training.")
        return

    best_model_metrics = metrics.get("best_model_metrics", {})
    best_model = metrics.get("best_model", "Unavailable")
    comparison = pd.DataFrame(metrics.get("model_comparison", []))

    st.caption(f"Best model: {best_model}")
    metric_columns = st.columns(4)
    metric_columns[0].metric("R2", f"{best_model_metrics.get('r2', 0):.3f}")
    metric_columns[1].metric("MAE", f"{best_model_metrics.get('mae', 0):.3f}")
    metric_columns[2].metric("RMSE", f"{best_model_metrics.get('rmse', 0):.3f}")
    metric_columns[3].metric("MAPE", f"{best_model_metrics.get('mape', 0):.2f}%")

    if not comparison.empty:
        st.dataframe(comparison, use_container_width=True, hide_index=True)


def render_shap_explanation(
    prediction_service: PredictionService | None,
    payload: dict[str, object],
    predicted_price: float | None,
) -> None:
    """Render a local SHAP explanation for the latest prediction."""
    st.subheader("Prediction Explanation")

    if prediction_service is None or predicted_price is None:
        st.info("A SHAP explanation will appear after a successful prediction.")
        return

    try:
        import shap
        engineered_input = prepare_single_prediction_input(payload)
        transformed_input = prediction_service.preprocessor.transform(engineered_input)
        feature_names = prediction_service.preprocessor.get_feature_names_out()
        transformed_frame = pd.DataFrame(transformed_input, columns=feature_names)

        explainer = shap.Explainer(prediction_service.model, transformed_frame)
        shap_values = explainer(transformed_frame)

        fig = plt.figure(figsize=(9, 4))
        shap.plots.waterfall(shap_values[0], show=False, max_display=12)
        st.pyplot(fig, use_container_width=True)
    except Exception as exc:
        st.info(f"SHAP explanation is unavailable for this model instance: {exc}")


def main() -> None:
    """Run the Streamlit app."""
    prediction_service = load_prediction_service()
    feature_importance = load_feature_importance()
    metrics = load_model_metrics()
    payload = build_sidebar_payload()

    st.title("Used Car Price Prediction")
    st.caption("Production-style ML app for estimating resale value in INR lakhs.")

    left_column, right_column = st.columns([1.1, 0.9])

    with left_column:
        predicted_price = render_prediction_panel(prediction_service, payload)
        render_model_performance(metrics)

    with right_column:
        render_feature_importance(feature_importance)
        render_shap_explanation(prediction_service, payload, predicted_price)


if __name__ == "__main__":
    main()
