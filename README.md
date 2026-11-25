# Demand Forecasting and Inventory Optimization

An AI-powered app that forecasts product demand and generates inventory recommendations, with a Python/FastAPI backend and a React dashboard.

## Why I built this

I wanted to explore a practical question retailers face: how much stock should they keep when demand changes from day to day? Too little can mean missed orders; too much ties up money in inventory. I enjoyed turning a forecasting experiment into a dashboard that helps make the next step easier to understand.

### The business problem

A retailer needs a sensible demand estimate before deciding when to reorder and how much extra stock to hold. A more complicated forecast is useful only if it improves on a simple, reliable starting point.

### What the data showed

In our academic evaluation, we looked at daily demand for 15 product categories in the DataCo Smart Supply Chain dataset. We compared XGBoost with straightforward forecasts using a later test period, while fixing issues that could make results look better than they really were. XGBoost had **22.2% WAPE**; a 30-day moving average had **22.1% WAPE**. For this dataset and test period, the more complex model did not show a meaningful overall accuracy gain.

### What I would recommend

Start with the 30-day moving average as a clear baseline. Try a more complex model only where testing on future periods shows that it consistently helps. Use the forecast alongside lead time and a chosen service level to plan reorder points and safety stock, then test those decisions against real inventory outcomes before claiming savings or fewer stockouts. The inventory figures here are exploratory; the dataset did not provide actual lead times or costs, and the plan was not deployed or simulated.

The academic comparison above explains the decision behind the project. The app in this repository explores forecasts with Prophet and XGBoost and presents inventory recommendations in a dashboard.

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
