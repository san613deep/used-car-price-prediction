"""Inference utilities for used car price prediction."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.data_preprocessing import prepare_single_prediction_input
from src.utils import (
    BEST_MODEL_PATH,
    MODEL_INPUT_COLUMNS,
    PREPROCESSOR_PATH,
    load_joblib_artifact,
)


logger = logging.getLogger(__name__)


@dataclass
class PredictionService:
    """Load model artifacts once and serve predictions."""

    model: Any
    preprocessor: Any

    @classmethod
    def from_artifacts(cls) -> "PredictionService":
        """Create a prediction service from persisted joblib artifacts."""
        model = load_joblib_artifact(BEST_MODEL_PATH)
        preprocessor = load_joblib_artifact(PREPROCESSOR_PATH)
        return cls(model=model, preprocessor=preprocessor)

    def predict(self, payload: dict[str, object]) -> float:
        """Predict selling price for one vehicle payload."""
        validate_prediction_payload(payload)
        prepared_input = prepare_single_prediction_input(payload)
        transformed_input = self.preprocessor.transform(prepared_input)
        prediction = self.model.predict(transformed_input)
        return float(np.maximum(prediction[0], 0))

    def predict_batch(self, payloads: list[dict[str, object]]) -> list[float]:
        """Predict selling prices for multiple vehicle payloads."""
        if not payloads:
            return []

        for payload in payloads:
            validate_prediction_payload(payload)

        prepared_inputs = pd.concat(
            [prepare_single_prediction_input(payload) for payload in payloads],
            ignore_index=True,
        )
        transformed_inputs = self.preprocessor.transform(prepared_inputs)
        predictions = self.model.predict(transformed_inputs)
        return [float(value) for value in np.maximum(predictions, 0)]

    def explain_prediction_input(self, payload: dict[str, object]) -> pd.DataFrame:
        """Return the engineered single-row dataframe used before preprocessing."""
        validate_prediction_payload(payload)
        return prepare_single_prediction_input(payload)


def validate_prediction_payload(payload: dict[str, object]) -> None:
    """Validate that an inference payload includes required model inputs."""
    missing_fields = [field for field in MODEL_INPUT_COLUMNS if field not in payload]
    if missing_fields:
        raise ValueError(f"Missing required prediction fields: {missing_fields}")

    if int(payload["year"]) < 1990 or int(payload["year"]) > 2026:
        raise ValueError("year must be between 1990 and 2026")

    if float(payload["present_price"]) <= 0:
        raise ValueError("present_price must be greater than 0")

    if int(payload["driven_km"]) < 0:
        raise ValueError("driven_km cannot be negative")

    if int(payload["owner"]) < 0:
        raise ValueError("owner cannot be negative")


def predict_price(payload: dict[str, object]) -> float:
    """Convenience function for one-off predictions."""
    service = PredictionService.from_artifacts()
    predicted_price = service.predict(payload)
    logger.info("Generated prediction: %.2f", predicted_price)
    return predicted_price
