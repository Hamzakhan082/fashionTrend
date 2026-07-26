"""
Improved model with hyperparameter tuning
"""
import os, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

BASE = r'C:\Users\hamza\Machine Learning Projects'
OUTPUT = r'C:\Users\hamza\Machine Learning Projects\Fashion Trend dataset'


def load_csv(filename, **kwargs):
    """Load a dataset CSV, failing with an actionable message instead of a raw traceback."""
    path = os.path.join(BASE, filename)
    try:
        return pd.read_csv(path, **kwargs)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Required data file not found: {path}. "
            f"Check that BASE ({BASE!r}) points to the dataset directory."
        ) from exc
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Data file is empty or has no columns: {path}") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"Failed to parse data file: {path} ({exc})") from exc


# Load & preprocess
customers = load_csv('customers.csv', low_memory=False)
products = load_csv('products.csv', low_memory=False)
transactions = load_csv('transactions.csv', low_memory=False)
stores = load_csv('stores.csv', low_memory=False)

customers['Age'] = 2024 - pd.to_datetime(customers['Date Of Birth'], errors='coerce').dt.year
customers['AgeGroup'] = pd.cut(customers['Age'], bins=[0,18,25,35,50,65,120], labels=['0-18','19-25','26-35','36-50','51-65','65+'])
products['ProductColor'] = products['Color'].fillna('Unknown')
products = products.drop(columns=['Color'], errors='ignore')

transactions['Date'] = pd.to_datetime(transactions['Date'], errors='coerce')
transactions['Month'] = transactions['Date'].dt.month
transactions['Year'] = transactions['Date'].dt.year
def get_season(m):
    if m in [12,1,2]: return 'Winter'
    elif m in [3,4,5]: return 'Spring'
    elif m in [6,7,8]: return 'Summer'
    else: return 'Fall'
transactions['Season'] = transactions['Month'].apply(get_season)

# Merge
data = transactions.merge(products[['Product ID','Category','Sub Category','ProductColor','Production Cost']], on='Product ID', how='left')
data = data.merge(customers[['Customer ID','Gender','Age','AgeGroup','City']], on='Customer ID', how='left')
data = data.merge(stores[['Store ID','City']], on='Store ID', how='left', suffixes=('_cust','_store'))

# Aggregate
agg_data = data.groupby(['Category','Season','Year','ProductColor','Gender','AgeGroup']).agg(
    total_quantity=('Quantity','sum'),
    total_revenue=('Line Total','sum'),
    avg_unit_price=('Unit Price','mean'),
    avg_discount=('Discount','mean'),
    transaction_count=('Date','count'),
    avg_age=('Age','mean'),
    city_count=('City_cust','nunique'),
    production_cost=('Production Cost','mean')
).reset_index()

# Lags
season_order = {'Spring':0,'Summer':1,'Fall':2,'Winter':3}
agg_data['SeasonNum'] = agg_data['Season'].map(season_order)
agg_data['TimeStep'] = agg_data['Year']*4 + agg_data['SeasonNum']
agg_data = agg_data.sort_values(['Category','ProductColor','Gender','AgeGroup','TimeStep'])

group_cols = ['Category','ProductColor','Gender','AgeGroup']
target = 'total_quantity'
lag_cols = ['total_revenue','avg_unit_price','avg_discount','transaction_count']

for col in lag_cols + [target]:
    for lag in range(1,3):
        agg_data[f'{col}_lag{lag}'] = agg_data.groupby(group_cols, group_keys=False)[col].shift(lag)
    agg_data[f'{col}_roll2'] = agg_data.groupby(group_cols, group_keys=False)[col].transform(lambda x: x.rolling(2,min_periods=1).mean())

agg_data = agg_data.dropna().reset_index(drop=True)

# Encode
for col in ['Category','Season','ProductColor','Gender','AgeGroup']:
    agg_data[f'{col}_enc'] = LabelEncoder().fit_transform(agg_data[col])

feature_cols = ['SeasonNum','TimeStep',
    'total_quantity_lag1','total_quantity_lag2','total_quantity_roll2',
    'total_revenue_lag1','total_revenue_lag2','total_revenue_roll2',
    'avg_unit_price_lag1','avg_unit_price_lag2','avg_unit_price_roll2',
    'avg_discount_lag1','avg_discount_lag2','avg_discount_roll2',
    'transaction_count_lag1','transaction_count_lag2','transaction_count_roll2',
    'avg_age','city_count','production_cost','avg_unit_price',
    'Category_enc','Season_enc','ProductColor_enc','Gender_enc','AgeGroup_enc']

X = agg_data[feature_cols].copy()
y = agg_data[target].values

# Time split
agg_data_sorted = agg_data.sort_values('TimeStep').reset_index(drop=True)
split_idx = int(len(agg_data_sorted)*0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

numeric_feats = ['avg_age','city_count','production_cost','avg_unit_price',
    'total_quantity_lag1','total_quantity_lag2','total_quantity_roll2',
    'total_revenue_lag1','total_revenue_lag2','total_revenue_roll2',
    'avg_unit_price_lag1','avg_unit_price_lag2','avg_unit_price_roll2',
    'avg_discount_lag1','avg_discount_lag2','avg_discount_roll2',
    'transaction_count_lag1','transaction_count_lag2','transaction_count_roll2']

scaler = StandardScaler()
X_train[numeric_feats] = scaler.fit_transform(X_train[numeric_feats])
X_test[numeric_feats] = scaler.transform(X_test[numeric_feats])

print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# ============================================================
# HYPERPARAMETER TUNING with TimeSeriesSplit
# ============================================================
print("\n" + "="*60)
print("HYPERPARAMETER TUNING")
print("="*60)

tscv = TimeSeriesSplit(n_splits=3)

tuned_models = {}

# XGBoost tuning
print("\nTuning XGBoost...")
xgb_params = {
    'n_estimators': [100, 200],
    'max_depth': [6, 10, 15],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0]
}
xgb_grid = GridSearchCV(XGBRegressor(random_state=42, verbosity=0),
                        xgb_params, cv=tscv, scoring='r2', n_jobs=-1)
xgb_grid.fit(X_train, y_train)
tuned_models['XGBoost'] = xgb_grid.best_estimator_
print(f"  Best params: {xgb_grid.best_params_}")
print(f"  Best CV R2: {xgb_grid.best_score_:.4f}")

# Random Forest tuning
print("\nTuning Random Forest...")
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}
rf_grid = GridSearchCV(RandomForestRegressor(random_state=42, n_jobs=-1),
                       rf_params, cv=tscv, scoring='r2', n_jobs=-1)
rf_grid.fit(X_train, y_train)
tuned_models['Random Forest'] = rf_grid.best_estimator_
print(f"  Best params: {rf_grid.best_params_}")
print(f"  Best CV R2: {rf_grid.best_score_:.4f}")

# LightGBM tuning
print("\nTuning LightGBM...")
lgb_params = {
    'n_estimators': [100, 200],
    'max_depth': [6, 10, -1],
    'learning_rate': [0.05, 0.1],
    'num_leaves': [31, 50]
}
lgb_grid = GridSearchCV(LGBMRegressor(random_state=42, verbose=-1),
                        lgb_params, cv=tscv, scoring='r2', n_jobs=-1)
lgb_grid.fit(X_train, y_train)
tuned_models['LightGBM'] = lgb_grid.best_estimator_
print(f"  Best params: {lgb_grid.best_params_}")
print(f"  Best CV R2: {lgb_grid.best_score_:.4f}")

# CatBoost tuning
print("\nTuning CatBoost...")
cb_params = {
    'iterations': [100, 200],
    'depth': [6, 10],
    'learning_rate': [0.05, 0.1]
}
cb_grid = GridSearchCV(CatBoostRegressor(random_state=42, verbose=0),
                       cb_params, cv=tscv, scoring='r2', n_jobs=-1)
cb_grid.fit(X_train, y_train)
tuned_models['CatBoost'] = cb_grid.best_estimator_
print(f"  Best params: {cb_grid.best_params_}")
print(f"  Best CV R2: {cb_grid.best_score_:.4f}")

# ============================================================
# EVALUATE ON TEST SET
# ============================================================
print("\n" + "="*60)
print("TUNED MODEL TEST PERFORMANCE")
print("="*60)

tuned_results = {}
for name, model in tuned_models.items():
    y_pred = model.predict(X_test)
    tuned_results[name] = {
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'MAE': mean_absolute_error(y_test, y_pred),
        'R2': r2_score(y_test, y_pred)
    }
    print(f"{name:15s} | RMSE: {tuned_results[name]['RMSE']:>8.2f} | MAE: {tuned_results[name]['MAE']:>8.2f} | R2: {tuned_results[name]['R2']:.4f}")

# Save comparison
pd.DataFrame(tuned_results).T.to_csv(os.path.join(OUTPUT, 'tuned_model_comparison.csv'))

# ============================================================
# COMPARE BEFORE vs AFTER
# ============================================================
print("\n" + "="*60)
print("BEFORE vs AFTER TUNING")
print("="*60)

# Best before: XGBoost R2=0.4163
# Get actual best after
best_after = max(tuned_results.items(), key=lambda x: x[1]['R2'])
print(f"Best tuned model: {best_after[0]} with R2 = {best_after[1]['R2']:.4f}")

# ============================================================
# PREDICT NEXT SEASON
# ============================================================
print("\n" + "="*60)
print("TOP 20 PREDICTIONS (TUNED MODEL)")
print("="*60)

best_tuned = tuned_models[best_after[0]]
preds = best_tuned.predict(X_test)

pred_df = agg_data_sorted.iloc[split_idx:][['Category','Season','ProductColor','Gender','AgeGroup','Year']].copy()
pred_df['predicted_quantity'] = preds
pred_df['actual_quantity'] = y_test
top20 = pred_df.sort_values('predicted_quantity', ascending=False).head(20)
print(top20.to_string())
top20.to_csv(os.path.join(OUTPUT, 'tuned_next_season_predictions.csv'), index=False)
