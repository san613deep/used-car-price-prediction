# Used Car Price Prediction

An end-to-end machine learning project for estimating used car resale prices from vehicle, usage, and seller attributes. The project is structured like a production ML repository rather than a single notebook, with reusable preprocessing code, model training, explainability, API deployment, Streamlit UI, and Docker support.

## Business Problem

Used car marketplaces need fast and reliable price estimates for inventory acquisition, seller onboarding, buyer transparency, and pricing operations. A poor price estimate can either reduce conversion by overpricing a vehicle or hurt margin by underpricing it.

This project predicts a car's selling price using information such as manufacturing year, showroom price, kilometers driven, fuel type, seller type, transmission, and previous owners.

## Architecture

```text
Raw Dataset
    |
    v
Data Cleaning
    |
    v
Feature Engineering
    |
    v
Scikit-learn Preprocessing Pipeline
    |
    v
Model Training and Tuning
    |
    v
Evaluation, Explainability, and Artifact Saving
    |
    +--> FastAPI /predict endpoint
    |
    +--> Streamlit prediction app
```

## Project Structure

```text
used-car-price-prediction/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   ├── 03_Model_Training.ipynb
│   └── 04_Model_Evaluation.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── predict.py
│   ├── evaluate.py
│   └── utils.py
├── models/
├── api/
│   └── app.py
├── streamlit_app.py
├── requirements.txt
├── Dockerfile
├── README.md
└── .gitignore
```

## Dataset

The project expects a Cardekho-style used car dataset with the following fields:

| Column | Description |
| --- | --- |
| `Year` | Manufacturing year |
| `Present_Price` | Current showroom price in INR lakhs |
| `Selling_Price` | Resale price in INR lakhs |
| `Driven_kms` | Total kilometers driven |
| `Fuel_Type` | Petrol, Diesel, or CNG |
| `Selling_type` | Dealer or Individual |
| `Transmission` | Manual or Automatic |
| `Owner` | Number of previous owners |

Place the dataset at:

```text
data/raw/car_data.csv
```

The utility layer also supports the original `car data.csv` filename when it exists in the parent workspace.

## EDA

The EDA notebook covers:

- Missing value analysis
- Duplicate detection
- Outlier detection
- Target distribution
- Numerical feature distributions
- Categorical feature analysis
- Correlation heatmap
- Pair plots
- Box plots
- Business insights after each visualization

## Feature Engineering

Engineered features include:

- `car_age`
- `km_per_year`
- `depreciation`
- `depreciation_rate`
- `brand`
- `is_luxury_brand`
- `vehicle_age_bucket`
- `price_segment`
- `log_present_price`
- `log_driven_km`

Target-derived features are available for EDA, but excluded from model inputs to prevent leakage.

## Modeling

The training pipeline compares:

- Linear Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost
- CatBoost
- LightGBM

The workflow uses 5-fold cross-validation and tunes stronger candidates with `GridSearchCV` or `RandomizedSearchCV`.

## Evaluation Metrics

Models are compared using:

- R2
- MAE
- RMSE
- MAPE
- Median Absolute Error
- Explained Variance Score
- Cross-validation RMSE

Evaluation plots include actual vs predicted, residuals, prediction error distribution, learning curve, validation curve, feature importance, and SHAP explanations.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Train Models

```bash
python src/train.py
```

For a faster baseline run:

```bash
python src/train.py --skip-tuning
```

Training saves:

```text
models/best_model.joblib
models/preprocessor.joblib
models/model_metrics.json
models/feature_importance.csv
```

## FastAPI Deployment

Run the API locally:

```bash
uvicorn api.app:app --reload
```

Prediction endpoint:

```http
POST /predict
```

Example request:

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

Example response:

```json
{
  "predicted_price": 7.21,
  "currency_unit": "INR lakhs",
  "model_status": "success"
}
```

## Streamlit App

```bash
streamlit run streamlit_app.py
```

The app includes sidebar inputs, predicted price, model performance, feature importance, and SHAP explanation panels.

## Docker

Build the image:

```bash
docker build -t used-car-price-prediction .
```

Run the container:

```bash
docker run -p 8000:8000 used-car-price-prediction
```

## Results

Results are generated after running `python src/train.py` and stored in `models/model_metrics.json`. The model comparison table is also displayed in the Streamlit app.

## Screenshots

Add screenshots after running the Streamlit app and notebooks:

- EDA summary
- Model comparison
- Feature importance
- SHAP explanation
- API documentation page

## Future Work

- Add make/model/version normalization from richer marketplace data
- Add city-level demand and supply features
- Track experiments with MLflow
- Add CI checks for formatting, tests, and API health
- Deploy the API on a cloud container service
- Add drift monitoring for price distribution changes
