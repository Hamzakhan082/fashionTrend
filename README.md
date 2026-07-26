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

Full dataset (6.4M transactions), 4262 aggregated season/segment rows, trained on 2023-Winter → 2024 and tested on the following 20% of seasons (2024 → 2025-Spring).

| Model | R2 | RMSE | MAE |
|-------|------|------|-----|
| XGBoost | 0.924 | 1803.18 | 400.58 |
| Random Forest | 0.894 | 2127.68 | 399.24 |
| CatBoost | 0.890 | 2168.12 | 613.43 |
| LightGBM | 0.887 | 2196.78 | 448.12 |

**Best model:** XGBoost (R2 = 0.92, RMSE = 1803)

After grid-search tuning (`fashion_trend_improved.py`), XGBoost reaches CV R2 = 0.942 with
`learning_rate=0.05, max_depth=6, n_estimators=100` and test R2 = 0.916 — i.e. tuning does not
beat the default configuration on the held-out future seasons.

## Key Features

| Feature | Importance | Description |
|---------|-----------|-------------|
| `city_count` | 1st | Reach across cities |
| `transaction_count_roll2` | 2nd | Transaction frequency trend |
| `total_revenue_roll2` | 3rd | Revenue trend over last 2 seasons |
| `total_quantity_roll2` | 4th | Sales momentum |
| `total_quantity_lag1` | 5th | Last season's sales volume |

## Next Season's Best-Sellers (Top 5)

| Category | Season | Color | Gender | AgeGroup | Predicted | Actual |
|----------|--------|-------|--------|----------|-----------|--------|
| Feminine | Winter | Unknown | F | 36-50 | 67,715 | 75,423 |
| Feminine | Winter | Unknown | F | 19-25 | 67,151 | 84,760 |
| Masculine | Winter | Unknown | F | 19-25 | 59,905 | 66,484 |
| Feminine | Winter | Unknown | F | 26-35 | 59,042 | 67,276 |
| Masculine | Winter | Unknown | F | 36-50 | 54,815 | 49,587 |

## Files

- `fashion_trend_utils.py` - Shared pipeline utilities (loading, preprocessing, feature engineering, splitting, evaluation)
- `fashion_trend_ml.py` - Full pipeline script
- `fashion_trend_improved.py` - Hyperparameter tuned version
- `fashion_trend_analysis.ipynb` - Interactive Jupyter notebook

Data and output directories default to the original Windows paths and can be overridden with the
`FASHION_TREND_DATA_DIR` and `FASHION_TREND_OUTPUT_DIR` environment variables.

## Getting Started

```bash
# Install dependencies
pip install pandas numpy scikit-learn xgboost lightgbm catboost matplotlib seaborn

# Run the pipeline
python fashion_trend_ml.py
```

## Room for Improvement

- Deep learning (RNN/LSTM) for time series modeling
- Additional external data (weather, holidays, social media trends)
- Hyperparameter tuning for all models
- Ensemble methods
