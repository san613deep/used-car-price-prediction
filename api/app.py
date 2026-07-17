"""FastAPI service for used car price prediction."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.predict import PredictionService  # noqa: E402
from src.utils import configure_logging  # noqa: E402


configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Used Car Price Prediction API",
    description="Production-ready API for predicting used car selling prices.",
    version="1.0.0",
)

prediction_service: PredictionService | None = None


class CarPriceRequest(BaseModel):
    """Request payload for a used car price prediction."""

    year: int = Field(..., ge=1990, le=2026, examples=[2017])
    present_price: float = Field(..., gt=0, examples=[9.85])
    driven_km: int = Field(..., ge=0, examples=[6900])
    fuel_type: str = Field(..., examples=["Petrol"])
    seller_type: str = Field(..., examples=["Dealer"])
    transmission: str = Field(..., examples=["Manual"])
    owner: int = Field(..., ge=0, examples=[0])


class CarPriceResponse(BaseModel):
    """Response payload for a predicted used car price."""

    predicted_price: float = Field(..., description="Predicted selling price in lakhs")
    currency_unit: str = Field(default="INR lakhs")
    model_status: str = Field(default="success")


@app.on_event("startup")
def load_model_artifacts() -> None:
    """Load model artifacts when the API server starts."""
    global prediction_service
    try:
        prediction_service = PredictionService.from_artifacts()
        logger.info("Prediction artifacts loaded successfully")
    except FileNotFoundError as exc:
        logger.warning("Model artifacts are not available yet: %s", exc)
        prediction_service = None


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return API health and model availability."""
    model_state = "loaded" if prediction_service is not None else "missing_artifacts"
    return {"status": "ok", "model_state": model_state}


@app.post("/predict", response_model=CarPriceResponse)
def predict_price(request: CarPriceRequest) -> CarPriceResponse:
    """Predict used car selling price from vehicle details."""
    if prediction_service is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts are unavailable. Run src/train.py before prediction.",
        )

    try:
        predicted_price = prediction_service.predict(request.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from exc

    return CarPriceResponse(predicted_price=round(predicted_price, 2))
