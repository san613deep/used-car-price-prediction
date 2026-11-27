# Used Car Price Prediction

An end-to-end regression project that estimates used-car resale prices from vehicle age, showroom price, mileage, fuel type, seller type, transmission and ownership history.

The repository separates preprocessing, feature engineering, model training, evaluation, prediction serving and the Streamlit interface. It is a reproducible portfolio project; model quality depends on the dataset supplied locally and should be judged from a fresh evaluation run.

## Problem and target

Predict the resale price (in INR lakhs) from vehicle attributes. The project expects a Cardekho-style CSV dataset and does not include a raw customer or vehicle dataset in the repository.

Place the dataset at `data/raw/car_data.csv`. Accepted source column names include the original Cardekho names such as `Selling_Price`, `Present_Price`, `Driven_kms`, `Fuel_Type`, `Selling_type`, `Transmission` and `Owner`.

## Workflow

```text
Local CSV dataset
       |
       v
Column normalization + data validation
       |
       v
Duplicate / invalid-row handling
       |
       v
Feature engineering
       |
       v
Train/test split + preprocessing pipeline
       |
       v
Baseline model comparison + optional tuning
       |
       v
Test metrics + feature importance
       |
       +----> FastAPI prediction endpoint
       |
       +----> Streamlit interface
```

## Technology

- Python, Pandas and NumPy
- Scikit-learn pipelines and model selection
- Linear Regression, Decision Tree, Random Forest and Gradient Boosting baselines
- Optional XGBoost, CatBoost and LightGBM models
- SHAP and feature-importance utilities
- FastAPI, Streamlit and Docker
- Pytest

## Repository layout

```text
api/                 FastAPI prediction service
data/raw/            User-provided source dataset (not committed)
data/processed/      Generated cleaned dataset (not committed)
models/              Trained model artifacts (generated locally)
notebooks/           EDA, feature engineering, training and evaluation
src/
  data_preprocessing.py
  feature_engineering.py
  train.py
  predict.py
  evaluate.py
  utils.py
tests/               Unit tests for feature and preprocessing logic
streamlit_app.py
Dockerfile
requirements.txt
README.md
```

## Setup

Python 3.10+ is recommended.

### Windows PowerShell

```powershell
git clone https://github.com/san613deep/used-car-price-prediction.git
cd used-car-price-prediction
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux

```bash
git clone https://github.com/san613deep/used-car-price-prediction.git
cd used-car-price-prediction
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Add the dataset

Create `data/raw/` if it does not exist and place `car_data.csv` inside it. The raw dataset is intentionally not committed; use a dataset you are permitted to download and redistribute.

## Train and evaluate

Full model comparison and tuning:

```bash
python src/train.py
```

Faster baseline run:

```bash
python src/train.py --skip-tuning
```

Training writes model artifacts and metrics under `models/`. These generated outputs are ignored by Git. Inspect `models/model_metrics.json` and the model-comparison output from your own run before reporting performance numbers.

## Run the API

```bash
uvicorn api.app:app --reload
```

Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

Example request to `POST /predict`:

```json
{
  "year": 2017,
  "present_price": 9.85,
  "driven_km": 6900,
  "fuel_type": "Petrol",
  "seller_type": "Dealer",
  "transmission": "Manual",
  "owner": 0
}
```

The API returns the predicted price in INR lakhs. Model artifacts must exist before the prediction endpoint can serve predictions.

## Run the Streamlit app

```bash
streamlit run streamlit_app.py
```

## Run tests

```bash
python -m pytest tests/ -q
```

The unit tests cover feature engineering, target-derived feature exclusion and basic cleaning behavior. GitHub Actions installs the lightweight test dependencies and runs these tests on pushes and pull requests. The tests do not train the full model or validate model accuracy.

## Evaluation and leakage controls

The project provides regression metrics including R², MAE, RMSE, MAPE, median absolute error and explained variance. Feature engineering can create target-derived columns for offline analysis, but `get_model_feature_columns` excludes target-derived values from model inputs.

Use a held-out test set and cross-validation to compare candidate models. Do not copy example metrics into a resume: run training on the dataset you actually use and report the resulting test metrics, dataset size and split strategy.

## Current limitations

- The raw dataset is not included; users must provide it locally.
- Model performance has not been asserted here because it depends on the dataset and training run.
- Training can be compute-intensive when optional boosting libraries and hyperparameter searches are enabled.
- The API is a portfolio prototype. Add authentication, request throttling, model versioning and monitoring before public production use.
- The available unit tests validate selected preprocessing behavior, not end-to-end training quality.

## Future improvements

- Add a documented dataset acquisition script and dataset checksum.
- Add integration tests for model training and prediction using a small synthetic fixture.
- Compare random splitting with a time-aware or grouped validation strategy where appropriate.
- Add model artifact metadata and a reproducible experiment report.
- Add API tests for missing artifacts, invalid categorical values and prediction responses.
