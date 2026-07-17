"""Train and select the best used car price prediction model."""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RandomizedSearchCV,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data_preprocessing import prepare_model_data
from src.evaluate import (
    build_model_comparison_table,
    calculate_regression_metrics,
    extract_feature_importance,
    get_feature_names_from_preprocessor,
)
from src.utils import (
    BEST_MODEL_PATH,
    FEATURE_IMPORTANCE_PATH,
    METRICS_PATH,
    PREPROCESSOR_PATH,
    RANDOM_STATE,
    configure_logging,
    ensure_directories,
    load_raw_data,
    save_joblib_artifact,
    save_json,
)


logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


@dataclass(frozen=True)
class TrainingResult:
    """Container for a trained model's evaluation result."""

    model_name: str
    estimator: Any
    test_metrics: dict[str, float]
    cv_rmse_mean: float
    cv_rmse_std: float


def build_model_registry() -> dict[str, Any]:
    """Return the model registry used for baseline comparison."""
    models: dict[str, Any] = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }

    optional_models = [
        (
            "XGBoost",
            "xgboost",
            "XGBRegressor",
            {
                "objective": "reg:squarederror",
                "random_state": RANDOM_STATE,
                "n_estimators": 300,
                "learning_rate": 0.05,
                "max_depth": 4,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
            },
        ),
        (
            "CatBoost",
            "catboost",
            "CatBoostRegressor",
            {
                "loss_function": "RMSE",
                "random_state": RANDOM_STATE,
                "iterations": 400,
                "learning_rate": 0.05,
                "depth": 6,
                "verbose": False,
            },
        ),
        (
            "LightGBM",
            "lightgbm",
            "LGBMRegressor",
            {
                "objective": "regression",
                "random_state": RANDOM_STATE,
                "n_estimators": 400,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "verbose": -1,
            },
        ),
    ]

    for model_name, module_name, class_name, parameters in optional_models:
        try:
            module = __import__(module_name, fromlist=[class_name])
            model_class = getattr(module, class_name)
            models[model_name] = model_class(**parameters)
        except ImportError:
            logger.warning("%s is not installed; skipping %s", module_name, model_name)

    return models


def get_search_spaces() -> dict[str, dict[str, list[Any]]]:
    """Return hyperparameter search spaces for selected model families."""
    return {
        "Decision Tree": {
            "model__max_depth": [3, 5, 8, 12, None],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
        },
        "Random Forest": {
            "model__n_estimators": [200, 300, 500, 700],
            "model__max_depth": [4, 6, 8, 12, None],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
        },
        "Gradient Boosting": {
            "model__n_estimators": [150, 250, 400],
            "model__learning_rate": [0.03, 0.05, 0.08, 0.1],
            "model__max_depth": [2, 3, 4],
            "model__subsample": [0.8, 0.9, 1.0],
        },
        "XGBoost": {
            "model__n_estimators": [200, 300, 500],
            "model__max_depth": [3, 4, 5, 6],
            "model__learning_rate": [0.03, 0.05, 0.08, 0.1],
            "model__subsample": [0.75, 0.9, 1.0],
            "model__colsample_bytree": [0.75, 0.9, 1.0],
        },
        "CatBoost": {
            "model__iterations": [250, 400, 600],
            "model__depth": [4, 6, 8],
            "model__learning_rate": [0.03, 0.05, 0.08],
        },
        "LightGBM": {
            "model__n_estimators": [250, 400, 600],
            "model__num_leaves": [15, 31, 63],
            "model__learning_rate": [0.03, 0.05, 0.08],
            "model__subsample": [0.8, 0.9, 1.0],
        },
    }


def evaluate_pipeline(
    model_name: str,
    pipeline: Pipeline,
    features_train: pd.DataFrame,
    features_test: pd.DataFrame,
    target_train: pd.Series,
    target_test: pd.Series,
    cv: KFold,
) -> TrainingResult:
    """Fit and evaluate one model pipeline."""
    logger.info("Training %s", model_name)
    cv_scores = cross_val_score(
        pipeline,
        features_train,
        target_train,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=1,
    )
    pipeline.fit(features_train, target_train)
    predictions = pipeline.predict(features_test)
    metrics = calculate_regression_metrics(target_test, predictions)

    return TrainingResult(
        model_name=model_name,
        estimator=pipeline,
        test_metrics=metrics,
        cv_rmse_mean=float(-cv_scores.mean()),
        cv_rmse_std=float(cv_scores.std()),
    )


def tune_model(
    model_name: str,
    pipeline: Pipeline,
    features_train: pd.DataFrame,
    target_train: pd.Series,
    cv: KFold,
) -> Pipeline:
    """Tune a model pipeline with randomized or grid search."""
    search_spaces = get_search_spaces()
    parameter_grid = search_spaces.get(model_name)

    if not parameter_grid:
        return pipeline

    if model_name == "Decision Tree":
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=parameter_grid,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            n_jobs=1,
        )
    else:
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=parameter_grid,
            n_iter=25,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )

    logger.info("Tuning %s", model_name)
    search.fit(features_train, target_train)
    logger.info("%s best parameters: %s", model_name, search.best_params_)
    return search.best_estimator_


def train_models(tune_top_models: bool = True) -> tuple[pd.DataFrame, TrainingResult]:
    """Run the complete model training and selection workflow."""
    ensure_directories()
    raw_data = load_raw_data()
    features, target, preprocessor = prepare_model_data(raw_data)

    features_train, features_test, target_train, target_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results: list[TrainingResult] = []
    for model_name, model in build_model_registry().items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                ("model", model),
            ]
        )
        results.append(
            evaluate_pipeline(
                model_name,
                pipeline,
                features_train,
                features_test,
                target_train,
                target_test,
                cv,
            )
        )

    comparison_rows = [
        {
            "model": result.model_name,
            **result.test_metrics,
            "cv_rmse_mean": result.cv_rmse_mean,
            "cv_rmse_std": result.cv_rmse_std,
        }
        for result in results
    ]
    comparison_table = build_model_comparison_table(comparison_rows)

    best_model_name = str(comparison_table.loc[0, "model"])
    best_result = next(result for result in results if result.model_name == best_model_name)

    if tune_top_models:
        top_model_names = comparison_table.head(3)["model"].tolist()
        tuned_results: list[TrainingResult] = []

        for model_name in top_model_names:
            baseline = next(result for result in results if result.model_name == model_name)
            tuned_pipeline = tune_model(
                model_name,
                baseline.estimator,
                features_train,
                target_train,
                cv,
            )
            tuned_results.append(
                evaluate_pipeline(
                    f"{model_name} Tuned",
                    tuned_pipeline,
                    features_train,
                    features_test,
                    target_train,
                    target_test,
                    cv,
                )
            )

        all_results = results + tuned_results
        comparison_rows = [
            {
                "model": result.model_name,
                **result.test_metrics,
                "cv_rmse_mean": result.cv_rmse_mean,
                "cv_rmse_std": result.cv_rmse_std,
            }
            for result in all_results
        ]
        comparison_table = build_model_comparison_table(comparison_rows)
        best_model_name = str(comparison_table.loc[0, "model"])
        best_result = next(
            result for result in all_results if result.model_name == best_model_name
        )

    save_training_artifacts(best_result, comparison_table, features_train, target_train)
    return comparison_table, best_result


def save_training_artifacts(
    best_result: TrainingResult,
    comparison_table: pd.DataFrame,
    features_train: pd.DataFrame,
    target_train: pd.Series,
) -> None:
    """Persist the best model, preprocessor, metrics, and feature importances."""
    best_pipeline = best_result.estimator
    fitted_preprocessor = best_pipeline.named_steps["preprocessor"]
    fitted_model = best_pipeline.named_steps["model"]

    save_joblib_artifact(fitted_model, BEST_MODEL_PATH)
    save_joblib_artifact(fitted_preprocessor, PREPROCESSOR_PATH)

    transformed_train = fitted_preprocessor.transform(features_train)
    feature_names = get_feature_names_from_preprocessor(fitted_preprocessor)
    transformed_train_frame = pd.DataFrame(transformed_train, columns=feature_names)

    feature_importance = extract_feature_importance(
        fitted_model,
        feature_names,
        transformed_train_frame,
        target_train,
    )
    feature_importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    metrics_payload = {
        "best_model": best_result.model_name,
        "best_model_metrics": best_result.test_metrics,
        "model_comparison": comparison_table.to_dict(orient="records"),
    }
    save_json(metrics_payload, METRICS_PATH)

    logger.info("Saved best model to %s", BEST_MODEL_PATH)
    logger.info("Saved fitted preprocessor to %s", PREPROCESSOR_PATH)
    logger.info("Saved metrics to %s", METRICS_PATH)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train used car price models.")
    parser.add_argument(
        "--skip-tuning",
        action="store_true",
        help="Run baseline comparison only without hyperparameter tuning.",
    )
    return parser.parse_args()


def main() -> None:
    """Entrypoint for model training."""
    configure_logging()
    args = parse_args()
    comparison_table, best_result = train_models(tune_top_models=not args.skip_tuning)
    logger.info("Best model: %s", best_result.model_name)
    logger.info("\n%s", comparison_table.to_string(index=False))


if __name__ == "__main__":
    main()



