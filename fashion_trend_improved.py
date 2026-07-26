"""
Improved model with hyperparameter tuning
"""
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

import fashion_trend_utils as ftu

# Load, preprocess, aggregate and engineer features
data = ftu.load_and_prepare_data()
agg_data = ftu.add_lag_features(ftu.aggregate_by_season(data))
ftu.encode_categoricals(agg_data)

X, y, feature_cols, numeric_feats = ftu.build_features(agg_data)
X_train, X_test, y_train, y_test, agg_data_sorted, split_idx = ftu.time_based_split(
    agg_data, X, y
)
X_train, X_test, scaler = ftu.scale_features(X_train, X_test, numeric_feats)

print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# ============================================================
# HYPERPARAMETER TUNING with TimeSeriesSplit
# ============================================================
ftu.print_section("HYPERPARAMETER TUNING")

tscv = TimeSeriesSplit(n_splits=3)

search_spaces = {
    'XGBoost': (
        XGBRegressor(random_state=42, verbosity=0),
        {
            'n_estimators': [100, 200],
            'max_depth': [6, 10, 15],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8, 1.0],
        },
    ),
    'Random Forest': (
        RandomForestRegressor(random_state=42, n_jobs=-1),
        {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
        },
    ),
    'LightGBM': (
        LGBMRegressor(random_state=42, verbose=-1),
        {
            'n_estimators': [100, 200],
            'max_depth': [6, 10, -1],
            'learning_rate': [0.05, 0.1],
            'num_leaves': [31, 50],
        },
    ),
    'CatBoost': (
        CatBoostRegressor(random_state=42, verbose=0),
        {
            'iterations': [100, 200],
            'depth': [6, 10],
            'learning_rate': [0.05, 0.1],
        },
    ),
}

tuned_models = {}
for name, (estimator, params) in search_spaces.items():
    print(f"\nTuning {name}...")
    grid = GridSearchCV(estimator, params, cv=tscv, scoring='r2', n_jobs=-1)
    grid.fit(X_train, y_train)
    tuned_models[name] = grid.best_estimator_
    print(f"  Best params: {grid.best_params_}")
    print(f"  Best CV R2: {grid.best_score_:.4f}")

# ============================================================
# EVALUATE ON TEST SET
# ============================================================
ftu.print_section("TUNED MODEL TEST PERFORMANCE")

tuned_results = ftu.evaluate_models(tuned_models, X_test, y_test)
ftu.save_results_table(tuned_results, 'tuned_model_comparison.csv')

# ============================================================
# COMPARE BEFORE vs AFTER
# ============================================================
ftu.print_section("BEFORE vs AFTER TUNING")

best_after = ftu.best_model_name(tuned_results)
print(f"Best tuned model: {best_after} with R2 = {tuned_results[best_after]['R2']:.4f}")

# ============================================================
# PREDICT NEXT SEASON
# ============================================================
ftu.print_section("TOP 20 PREDICTIONS (TUNED MODEL)")

best_tuned = tuned_models[best_after]
top20 = ftu.save_predictions(
    agg_data_sorted, split_idx, best_tuned.predict(X_test), y_test,
    'tuned_next_season_predictions.csv'
)
print(top20.to_string())
