"""Model evaluation, visualization, and explainability utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import learning_curve, validation_curve

from src.utils import RANDOM_STATE


logger = logging.getLogger(__name__)


def _get_shap():
    """Import SHAP only when explainability functions are called."""
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "SHAP is required for explainability plots. Install requirements.txt first."
        ) from exc
    return shap


def mean_absolute_percentage_error_safe(
    actual: np.ndarray | pd.Series,
    predicted: np.ndarray | pd.Series,
) -> float:
    """Calculate MAPE while safely handling zero actual values."""
    actual_array = np.asarray(actual)
    predicted_array = np.asarray(predicted)
    non_zero_mask = actual_array != 0

    if not np.any(non_zero_mask):
        return float("nan")

    return float(
        np.mean(
            np.abs(
                (actual_array[non_zero_mask] - predicted_array[non_zero_mask])
                / actual_array[non_zero_mask]
            )
        )
        * 100
    )


def calculate_regression_metrics(
    actual: np.ndarray | pd.Series,
    predicted: np.ndarray | pd.Series,
) -> dict[str, float]:
    """Return the core regression metrics used to compare models."""
    return {
        "r2": float(r2_score(actual, predicted)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mape": mean_absolute_percentage_error_safe(actual, predicted),
        "median_absolute_error": float(median_absolute_error(actual, predicted)),
        "explained_variance": float(explained_variance_score(actual, predicted)),
    }


def build_model_comparison_table(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Create a sorted model comparison table from evaluation results."""
    comparison = pd.DataFrame(results)
    sort_columns = ["r2", "rmse"]
    ascending = [False, True]
    available_columns = [column for column in sort_columns if column in comparison.columns]

    if available_columns:
        comparison = comparison.sort_values(
            by=available_columns,
            ascending=ascending[: len(available_columns)],
        )

    return comparison.reset_index(drop=True)


def plot_actual_vs_predicted(
    actual: np.ndarray | pd.Series,
    predicted: np.ndarray | pd.Series,
    output_path: Path | None = None,
) -> plt.Figure:
    """Plot actual prices against predicted prices."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=actual, y=predicted, ax=ax, alpha=0.75)
    min_value = min(np.min(actual), np.min(predicted))
    max_value = max(np.max(actual), np.max(predicted))
    ax.plot([min_value, max_value], [min_value, max_value], color="red", linewidth=2)
    ax.set_title("Actual vs Predicted Selling Price")
    ax.set_xlabel("Actual Selling Price")
    ax.set_ylabel("Predicted Selling Price")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _save_figure(fig, output_path)
    return fig


def plot_residuals(
    actual: np.ndarray | pd.Series,
    predicted: np.ndarray | pd.Series,
    output_path: Path | None = None,
) -> plt.Figure:
    """Plot residuals to inspect systematic prediction errors."""
    residuals = np.asarray(actual) - np.asarray(predicted)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=predicted, y=residuals, ax=ax, alpha=0.75)
    ax.axhline(0, color="red", linestyle="--", linewidth=2)
    ax.set_title("Residual Plot")
    ax.set_xlabel("Predicted Selling Price")
    ax.set_ylabel("Residual")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _save_figure(fig, output_path)
    return fig


def plot_prediction_error_distribution(
    actual: np.ndarray | pd.Series,
    predicted: np.ndarray | pd.Series,
    output_path: Path | None = None,
) -> plt.Figure:
    """Plot the prediction error distribution."""
    errors = np.asarray(actual) - np.asarray(predicted)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(errors, kde=True, ax=ax, color="#2a9d8f")
    ax.axvline(0, color="red", linestyle="--", linewidth=2)
    ax.set_title("Prediction Error Distribution")
    ax.set_xlabel("Actual - Predicted")
    ax.set_ylabel("Count")
    fig.tight_layout()
    _save_figure(fig, output_path)
    return fig


def plot_learning_curve(
    estimator: Any,
    features: pd.DataFrame,
    target: pd.Series,
    output_path: Path | None = None,
    cv: int = 5,
) -> plt.Figure:
    """Plot train and validation learning curves."""
    train_sizes, train_scores, validation_scores = learning_curve(
        estimator,
        features,
        target,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        train_sizes=np.linspace(0.2, 1.0, 5),
        n_jobs=1,
    )
    train_rmse = -train_scores.mean(axis=1)
    validation_rmse = -validation_scores.mean(axis=1)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(train_sizes, train_rmse, marker="o", label="Training RMSE")
    ax.plot(train_sizes, validation_rmse, marker="o", label="Validation RMSE")
    ax.set_title("Learning Curve")
    ax.set_xlabel("Training Examples")
    ax.set_ylabel("RMSE")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _save_figure(fig, output_path)
    return fig


def plot_validation_curve(
    estimator: Any,
    features: pd.DataFrame,
    target: pd.Series,
    parameter_name: str,
    parameter_range: list[int] | list[float],
    output_path: Path | None = None,
    cv: int = 5,
) -> plt.Figure:
    """Plot validation curve for one hyperparameter."""
    train_scores, validation_scores = validation_curve(
        estimator,
        features,
        target,
        param_name=parameter_name,
        param_range=parameter_range,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=1,
    )
    train_rmse = -train_scores.mean(axis=1)
    validation_rmse = -validation_scores.mean(axis=1)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(parameter_range, train_rmse, marker="o", label="Training RMSE")
    ax.plot(parameter_range, validation_rmse, marker="o", label="Validation RMSE")
    ax.set_title(f"Validation Curve: {parameter_name}")
    ax.set_xlabel(parameter_name)
    ax.set_ylabel("RMSE")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _save_figure(fig, output_path)
    return fig


def get_feature_names_from_preprocessor(preprocessor: Any) -> list[str]:
    """Extract transformed feature names from a fitted ColumnTransformer."""
    if hasattr(preprocessor, "get_feature_names_out"):
        return list(preprocessor.get_feature_names_out())
    return []


def extract_feature_importance(
    model: Any,
    feature_names: list[str],
    features: pd.DataFrame | np.ndarray | None = None,
    target: pd.Series | np.ndarray | None = None,
) -> pd.DataFrame:
    """Extract model-native or permutation-based feature importance."""
    if hasattr(model, "feature_importances_"):
        importance_values = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance_values = np.abs(np.ravel(model.coef_))
    elif features is not None and target is not None:
        logger.info("Using permutation importance for model without native importance")
        permutation_result = permutation_importance(
            model,
            features,
            target,
            n_repeats=20,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        importance_values = permutation_result.importances_mean
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    return (
        pd.DataFrame({"feature": feature_names, "importance": importance_values})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def plot_feature_importance(
    feature_importance: pd.DataFrame,
    top_n: int = 20,
    output_path: Path | None = None,
) -> plt.Figure:
    """Plot the top feature importances."""
    top_features = feature_importance.head(top_n).sort_values("importance")
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.barplot(data=top_features, x="importance", y="feature", ax=ax, color="#457b9d")
    ax.set_title(f"Top {top_n} Feature Importances")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    _save_figure(fig, output_path)
    return fig


def create_shap_explainer(model: Any, transformed_features: pd.DataFrame) -> Any:
    """Create a SHAP explainer for the fitted model."""
    shap = _get_shap()
    try:
        return shap.TreeExplainer(model)
    except Exception:
        logger.info("Falling back to generic SHAP explainer")
        return shap.Explainer(model, transformed_features)


def calculate_shap_values(
    model: Any,
    transformed_features: pd.DataFrame,
    max_rows: int = 200,
) -> tuple[Any, pd.DataFrame]:
    """Calculate SHAP values on a sample for efficient explainability."""
    sample = transformed_features.sample(
        min(max_rows, len(transformed_features)),
        random_state=RANDOM_STATE,
    )
    explainer = create_shap_explainer(model, sample)
    shap_values = explainer(sample)
    return shap_values, sample


def plot_shap_summary(
    shap_values: Any,
    sample: pd.DataFrame,
    output_path: Path | None = None,
) -> plt.Figure:
    """Generate a SHAP summary plot."""
    shap = _get_shap()
    shap.summary_plot(shap_values, sample, show=False)
    fig = plt.gcf()
    fig.tight_layout()
    _save_figure(fig, output_path)
    return fig


def plot_shap_waterfall(
    shap_values: Any,
    row_index: int = 0,
    output_path: Path | None = None,
) -> plt.Figure:
    """Generate a SHAP waterfall plot for one prediction."""
    shap = _get_shap()
    shap.plots.waterfall(shap_values[row_index], show=False)
    fig = plt.gcf()
    fig.tight_layout()
    _save_figure(fig, output_path)
    return fig


def _save_figure(fig: plt.Figure, output_path: Path | None) -> None:
    """Save a matplotlib figure when an output path is provided."""
    if output_path is None:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")


