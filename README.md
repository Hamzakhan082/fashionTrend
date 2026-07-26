# Fashion Trend Prediction

**What is in this season?** Using machine learning to identify fashion trends and predict next season's best-selling products.

[![Python](https://img.shields.io/badge/Python-3.14-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-passing-green)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-passing-brightgreen)](https://xgboost.readthedocs.io)

## Overview

This project builds a machine learning model to predict next season's best-selling fashion products by analyzing historical retail data. It compares multiple algorithms and identifies key features driving fashion trends.

### Research Questions

1. **RQ1** - Can ML accurately predict best-selling products?
2. **RQ2** - Which features most influence fashion trends?
3. **RQ3** - Which algorithm performs best?
4. **RQ4** - Can deep learning improve accuracy?
5. **RQ5** - Can external data (weather/holidays/social media) improve predictions?

## Dataset

- **Source:** [Global Fashion Retail Stores Dataset](https://www.kaggle.com/datasets/ricgomes/global-fashion-retail-storesdataset) (Kaggle)
- **Files:** 6 CSV files (transactions, customers, products, stores, discounts, employees)
- **Transactions:** 6.4M records across 35 stores
- **Products:** ~18K products in 3 categories

## Methodology

1. **Data Merging** — Join 6 tables on relational IDs
2. **Feature Engineering** — Lag features, rolling averages, season extraction, age grouping
3. **Aggregation** — Group by category, season, color, gender, and age group
4. **Time-based Train/Test Split** — 80% past data / 20% future data
5. **Model Training & Comparison** — Random Forest, XGBoost, LightGBM, CatBoost
6. **Feature Importance** — Identify key drivers of fashion trends

## Results

| Model | R2 | RMSE | MAE |
|-------|------|------|-----|
| XGBoost | 0.416 | 1414.14 | 205.73 |
| Random Forest | 0.279 | 1571.88 | 216.42 |
| CatBoost | 0.221 | 1633.83 | 459.91 |
| LightGBM | -0.166 | 1999.01 | 245.09 |

**Best model:** XGBoost (R = 0.42, RMSE = 1414)

## Key Features

| Feature | Importance | Description |
|---------|-----------|-------------|
| `city_count` | 1st | Reach across cities |
| `total_revenue_roll2` | 2nd | Revenue trend over last 2 seasons |
| `total_quantity_roll2` | 3rd | Sales momentum |
| `transaction_count_roll2` | 4th | Transaction frequency trend |
| `total_quantity_lag1` | 5th | Last season's sales volume |

## Next Season's Best-Sellers (Top 5)

| Category | Season | Color | Gender | AgeGroup | Predicted |
|----------|--------|-------|--------|----------|-----------|
| Feminine | Spring | WHITE | F | 65+ | 27,604 |
| Masculine | Spring | Unknown | D | 26-35 | 26,988 |
| Feminine | Spring | TURQUOISE | F | 26-35 | 24,993 |
| Feminine | Spring | TURQUOISE | F | 19-25 | 24,978 |
| Feminine | Winter | TURQUOISE | D | 36-50 | 22,071 |

## Files

- `fashion_trend_ml.py` - Full pipeline script
- `fashion_trend_improved.py` - Hyperparameter tuned version
- `fashion_trend_utils.py` - Shared preprocessing, splitting and evaluation helpers
- `fashion_trend_analysis.ipynb` - Interactive Jupyter notebook
- `tests/` - Unit tests for `fashion_trend_utils.py`

## Getting Started

```bash
# Install dependencies
pip install pandas numpy scikit-learn xgboost lightgbm catboost matplotlib seaborn

# Run the pipeline
python fashion_trend_ml.py
```

## Tests

The reusable pipeline logic lives in `fashion_trend_utils.py` and is covered by unit
tests that run without the Kaggle dataset (synthetic frames only).

```bash
pip install -r requirements-dev.txt
pytest                     # runs tests/ with coverage for fashion_trend_utils.py
```

## Room for Improvement

- Deep learning (RNN/LSTM) for time series modeling
- Additional external data (weather, holidays, social media trends)
- Hyperparameter tuning for all models
- Ensemble methods
