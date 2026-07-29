import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from features import build_features, encode_category
from inventory_optimization import build_inventory_plan

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR.parent / "models"
DATA_DIR = BASE_DIR.parent / "data"

app = FastAPI(title="AI Demand Forecasting & Inventory Optimization API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = xgb.XGBRegressor()
model.load_model(str(MODELS_DIR / "xgboost_model.json"))

daily_demand = pd.read_csv(DATA_DIR / "daily_demand_by_category.csv", parse_dates=["order_date"])

FEATURE_COLS = [
    'Category Name_code', 'day_of_week', 'is_weekend', 'month', 'week_of_year',
    'lag_1', 'lag_7', 'lag_14', 'rolling_mean_7', 'rolling_mean_30',
    'avg_discount_rate', 'late_delivery_risk'
]

# The shipped prophet_model.pkl (train_prophet.py) is fit on a single
# category only — whichever had the most rows at training time. Prophet's
# .fit() needs a compiled Stan backend (cmdstanpy) that isn't available in
# every deployment environment, so we don't retrain per-category on request;
# we only call .predict() on this one pre-fit model, which needs no backend.
# For every other category the forecast is XGBoost-only (see "models_used"
# in the /forecast response) rather than silently mislabeled as an ensemble.
PROPHET_CATEGORY = "Camping & Hiking"
prophet_model = joblib.load(MODELS_DIR / "prophet_model.pkl")

# XGBoost/Prophet blend weight, from the offline WAPE sweep in ensemble.py
# (best_weight=0.6 on the PROPHET_CATEGORY holdout set).
XGB_WEIGHT = 0.6

_dashboard_cache = {"data": None, "expires_at": 0}
_DASHBOARD_CACHE_TTL_SECONDS = 600


def get_category_df(category_name: str) -> pd.DataFrame:
    cat_df = daily_demand[daily_demand['Category Name'] == category_name].copy()
    if len(cat_df) == 0:
        raise HTTPException(status_code=404, detail=f"Category '{category_name}' not found")
    return cat_df.sort_values('order_date')


def xgb_multistep_forecast(cat_df: pd.DataFrame, days: int) -> list[dict]:
    """Recursively forecasts `days` ahead, feeding each prediction back in
    as the "actual" quantity so lag/rolling features stay defined."""
    working = cat_df[['Category Name', 'order_date', 'quantity',
                       'avg_discount_rate', 'late_delivery_risk']].copy()
    last_discount = working['avg_discount_rate'].iloc[-1]
    last_risk = working['late_delivery_risk'].iloc[-1]
    category_name = working['Category Name'].iloc[0]
    last_date = working['order_date'].max()

    predictions = []
    for _ in range(days):
        next_date = last_date + pd.Timedelta(days=1)
        new_row = pd.DataFrame([{
            'Category Name': category_name,
            'order_date': next_date,
            'quantity': np.nan,
            'avg_discount_rate': last_discount,
            'late_delivery_risk': last_risk,
        }])
        working = pd.concat([working, new_row], ignore_index=True)

        feat_df = build_features(working)
        feat_df = encode_category(feat_df, category_col='Category Name')
        feat_df[['avg_discount_rate', 'late_delivery_risk']] = (
            feat_df[['avg_discount_rate', 'late_delivery_risk']].fillna(0)
        )

        latest_row = feat_df.iloc[[-1]][FEATURE_COLS]
        pred = max(float(model.predict(latest_row)[0]), 0)

        working.loc[working.index[-1], 'quantity'] = pred
        predictions.append({'date': str(next_date.date()), 'xgb_pred': round(pred, 1)})
        last_date = next_date

    return predictions


@app.get("/")
def root():
    return {"message": "Demand Forecasting & Inventory Optimization API is running"}


@app.get("/categories")
def get_categories():
    categories = daily_demand['Category Name'].unique().tolist()
    return {"categories": categories}


@app.get("/forecast/{category_name}")
def get_forecast(category_name: str, days: int = 7):
    cat_df = get_category_df(category_name)

    feat_df = build_features(cat_df)
    feat_df = encode_category(feat_df, category_col='Category Name')
    feat_df[['avg_discount_rate', 'late_delivery_risk']] = (
        feat_df[['avg_discount_rate', 'late_delivery_risk']].fillna(0)
    )
    feat_df = feat_df.dropna(subset=FEATURE_COLS)

    if len(feat_df) == 0:
        raise HTTPException(status_code=400, detail="Not enough history to build features for this category")

    xgb_horizon = xgb_multistep_forecast(cat_df, days)
    has_prophet = category_name == PROPHET_CATEGORY

    prophet_forecast = None
    if has_prophet:
        future = prophet_model.make_future_dataframe(periods=days)
        prophet_forecast = prophet_model.predict(future).tail(days).reset_index(drop=True)

    horizon = []
    for i, xgb_point in enumerate(xgb_horizon):
        point = {"date": xgb_point['date'], "xgb_pred": xgb_point['xgb_pred']}

        if has_prophet:
            prophet_point = prophet_forecast.iloc[i]
            prophet_pred = max(float(prophet_point['yhat']), 0)
            blended = XGB_WEIGHT * xgb_point['xgb_pred'] + (1 - XGB_WEIGHT) * prophet_pred
            point.update({
                "prophet_pred": round(prophet_pred, 1),
                "blended_pred": round(blended, 1),
                "yhat_lower": round(max(float(prophet_point['yhat_lower']), 0), 1),
                "yhat_upper": round(float(prophet_point['yhat_upper']), 1),
            })
        else:
            point["blended_pred"] = xgb_point['xgb_pred']

        horizon.append(point)

    return {
        "category": category_name,
        "predicted_next_day_demand": horizon[0]['blended_pred'],
        "last_actual_demand": float(feat_df.iloc[-1]['quantity']),
        "last_date": str(feat_df.iloc[-1]['order_date'].date()),
        "models_used": ["xgboost", "prophet"] if has_prophet else ["xgboost"],
        "xgb_weight": XGB_WEIGHT if has_prophet else 1.0,
        "horizon": horizon,
    }


@app.get("/inventory/{category_name}")
def get_inventory_plan(category_name: str, lead_time_days: int = 7, service_level: float = 0.95):
    plan = build_inventory_plan(
        daily_demand, category_name,
        lead_time_days=lead_time_days,
        service_level=service_level
    )

    if "error" in plan:
        raise HTTPException(status_code=404, detail=plan["error"])

    return plan


@app.get("/dashboard-summary")
def dashboard_summary():
    now = time.monotonic()
    if _dashboard_cache["data"] is not None and now < _dashboard_cache["expires_at"]:
        return _dashboard_cache["data"]

    top_categories = (
        daily_demand.groupby('Category Name')['quantity']
        .sum().sort_values(ascending=False).head(15).index.tolist()
    )

    results = []
    for cat in top_categories:
        try:
            plan = build_inventory_plan(daily_demand, cat, lead_time_days=7, service_level=0.95)
            results.append(plan)
        except Exception:
            continue

    response = {"summary": results}
    _dashboard_cache["data"] = response
    _dashboard_cache["expires_at"] = now + _DASHBOARD_CACHE_TTL_SECONDS
    return response
