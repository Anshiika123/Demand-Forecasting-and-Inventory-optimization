# Demand Forecasting and Inventory Optimization

An AI-powered app that forecasts product demand and generates inventory recommendations, with a Python/FastAPI backend and a React dashboard.

## Overview

- **Forecasting**: Prophet and XGBoost models trained on historical order data, combined via ensembling.
- **Explainability**: SHAP-based feature importance for the demand model.
- **Inventory optimization**: Reorder points and safety stock derived from forecasts, lead time, and target service level.
- **API**: FastAPI service exposing forecasts, inventory plans, and dashboard summaries.
- **Dashboard**: React + Vite frontend for browsing categories, forecasts, and inventory plans.

## Project Structure

```
demand-forecast/            # Python backend
  data/                      # Raw and processed datasets
  models/                    # Trained model artifacts (Prophet, XGBoost)
  notebooks/                 # Exploratory data analysis
  outputs/                   # Predictions, inventory plans, SHAP summary
  src/
    api.py                   # FastAPI app
    features.py              # Feature engineering
    train_prophet.py         # Prophet training script
    train_xgboost.py         # XGBoost training script
    ensemble.py               # Ensembling logic
    inventory_optimization.py # Reorder point / safety stock logic
    shap_explain.py          # SHAP explainability
    requirements.txt

demand-forecast-frontend/   # React + Vite dashboard
  src/
  index.html
  package.json
```

## Getting Started

### Backend

```bash
cd demand-forecast/src
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn api:app --reload
```

The API runs at `http://localhost:8000`.

### Frontend

```bash
cd demand-forecast-frontend
npm install
npm run dev
```

The dashboard runs at `http://localhost:5173` and expects the API at `http://localhost:8000`.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| GET | `/categories` | List available product categories |
| GET | `/forecast/{category_name}?days=7` | Demand forecast for a category |
| GET | `/inventory/{category_name}?lead_time_days=7&service_level=0.95` | Inventory plan (reorder point, safety stock) for a category |
| GET | `/dashboard-summary` | Aggregated summary for the dashboard |

## Model Training

```bash
cd demand-forecast/src
python train_prophet.py
python train_xgboost.py
python shap_explain.py
```

Trained artifacts are written to `demand-forecast/models/`, and predictions/plans to `demand-forecast/outputs/`.

## Tech Stack

- **Backend**: Python, FastAPI, XGBoost, Prophet, scikit-learn, pandas, SHAP
- **Frontend**: React, Vite, Recharts, Axios

## License

[MIT](LICENSE)
