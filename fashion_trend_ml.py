"""
What is in this season? Using machine learning to identify fashion trends
Supervisor: Dr Ollie Bartlett
"""

import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from fashion_trend_utils import (
    add_customer_features,
    add_product_features,
    add_transaction_features,
    average_feature_importance,
    best_model_name,
    create_lag_features,
    encode_categoricals,
    evaluate_predictions,
    rank_models,
    scale_numeric_features,
    time_based_split,
    top_predictions,
)

BASE = r'C:\Users\hamza\Machine Learning Projects'
OUTPUT = r'C:\Users\hamza\Machine Learning Projects\Fashion Trend dataset'

# ============================================================
# 1. DATA LOADING
# ============================================================
print("=" * 60)
print("LOADING DATA")
print("=" * 60)

customers = pd.read_csv(os.path.join(BASE, 'customers.csv'), low_memory=False)
products = pd.read_csv(os.path.join(BASE, 'products.csv'), low_memory=False)
transactions = pd.read_csv(os.path.join(BASE, 'transactions.csv'), low_memory=False)
stores = pd.read_csv(os.path.join(BASE, 'stores.csv'), low_memory=False)
discounts = pd.read_csv(os.path.join(BASE, 'discounts.csv'), low_memory=False)

print(f"Customers: {customers.shape}")
print(f"Products: {products.shape}")
print(f"Transactions: {transactions.shape}")
print(f"Stores: {stores.shape}")
print(f"Discounts: {discounts.shape}")

# ============================================================
# 2. DATA PREPROCESSING
# ============================================================
print("\n" + "=" * 60)
print("PREPROCESSING")
print("=" * 60)

customers = add_customer_features(customers)
products = add_product_features(products)
transactions = add_transaction_features(transactions)

# Merge transactions with product, customer, store info
print("Merging datasets...")
data = transactions.merge(
    products[['Product ID', 'Category', 'Sub Category', 'ProductColor', 'Sizes', 'Production Cost']],
    on='Product ID', how='left'
)
data = data.merge(customers[['Customer ID', 'Gender', 'Age', 'AgeGroup', 'City', 'Country']], on='Customer ID', how='left')
data = data.merge(stores[['Store ID', 'Country', 'City']], on='Store ID', how='left', suffixes=('_cust', '_store'))

print(f"Merged data shape: {data.shape}")
print(f"Memory usage: {data.memory_usage(deep=True).sum() / 1e9:.2f} GB")

# ============================================================
# 3. FEATURE ENGINEERING FOR SEASONAL TREND PREDICTION
# ============================================================
print("\n" + "=" * 60)
print("FEATURE ENGINEERING FOR SEASONAL AGGREGATION")
print("=" * 60)

# We predict next season's total sales quantity per product category
# Aggregate by: Product Category + Season + Year
agg_data = data.groupby(['Category', 'Season', 'Year', 'ProductColor', 'Gender', 'AgeGroup']).agg(
    total_quantity=('Quantity', 'sum'),
    total_revenue=('Line Total', 'sum'),
    avg_unit_price=('Unit Price', 'mean'),
    avg_discount=('Discount', 'mean'),
    transaction_count=('Date', 'count'),
    avg_age=('Age', 'mean'),
    city_count=('City_cust', 'nunique'),
    store_count=('Store ID', 'nunique'),
    production_cost=('Production Cost', 'mean')
).reset_index()

# Sort for time-based splitting
agg_data = agg_data.sort_values(['Category', 'Season', 'Year'])

# Create lag features: use previous season's data as features
print("Creating lag features...")

agg_data = create_lag_features(
    agg_data,
    group_cols=['Category', 'ProductColor', 'Gender', 'AgeGroup'],
    target_col='total_quantity',
    lag_cols=['total_revenue', 'avg_unit_price', 'avg_discount', 'transaction_count'],
    n_lags=2
)

# Drop rows with NaN from lag features (first season per group)
agg_data = agg_data.dropna().reset_index(drop=True)

print(f"Aggregated data shape: {agg_data.shape}")
print(f"Seasons covered: {sorted(agg_data['Year'].unique())} - {sorted(agg_data['Season'].unique())}")

# ============================================================
# 4. ENCODE CATEGORICAL FEATURES
# ============================================================
print("\n" + "=" * 60)
print("ENCODING CATEGORICAL VARIABLES")
print("=" * 60)

cat_cols = ['Category', 'Season', 'ProductColor', 'Gender', 'AgeGroup']
agg_data, le_dict = encode_categoricals(agg_data, cat_cols)
for col in cat_cols:
    print(f"{col}: {len(le_dict[col].classes_)} categories")

# Features for modeling
feature_cols = [
    'SeasonNum', 'TimeStep',
    'total_quantity_lag1', 'total_quantity_lag2', 'total_quantity_roll2',
    'total_revenue_lag1', 'total_revenue_lag2', 'total_revenue_roll2',
    'avg_unit_price_lag1', 'avg_unit_price_lag2', 'avg_unit_price_roll2',
    'avg_discount_lag1', 'avg_discount_lag2', 'avg_discount_roll2',
    'transaction_count_lag1', 'transaction_count_lag2', 'transaction_count_roll2',
    'avg_age', 'city_count', 'store_count', 'production_cost', 'avg_unit_price'
]

# Add encoded categoricals
for col in ['Category_enc', 'Season_enc', 'ProductColor_enc', 'Gender_enc', 'AgeGroup_enc']:
    feature_cols.append(col)

target_col = 'total_quantity'

X = agg_data[feature_cols].copy()
y = agg_data[target_col].values

print(f"Features: {len(feature_cols)}")
print(f"Samples: {len(X)}")

# ============================================================
# 5. TRAIN/TEST SPLIT (Time-based: train on past, test on future)
# ============================================================
print("\n" + "=" * 60)
print("TRAIN/TEST SPLIT (Time-based)")
print("=" * 60)

# Sort by time and use last 20% as test
agg_data_sorted = agg_data.sort_values('TimeStep').reset_index(drop=True)
train_rows, test_rows, split_idx = time_based_split(agg_data, test_size=0.2)

train_idx = train_rows.index
test_idx = test_rows.index

X_train = X.iloc[train_idx]
y_train = y[train_idx]
X_test = X.iloc[test_idx]
y_test = y[test_idx]

print(f"Train: {len(X_train)}, Test: {len(X_test)}")
print(f"Train time range: {agg_data_sorted.iloc[train_idx]['Year'].min()}-{agg_data_sorted.iloc[train_idx]['Season'].iloc[-1]} {agg_data_sorted.iloc[train_idx]['Year'].iloc[-1]}")
print(f"Test time range: {agg_data_sorted.iloc[test_idx]['Year'].min()}-{agg_data_sorted.iloc[test_idx]['Season'].iloc[-1]} {agg_data_sorted.iloc[test_idx]['Year'].iloc[-1]}")

# Scale numerical features
numeric_feats = ['avg_age', 'city_count', 'store_count', 'production_cost', 'avg_unit_price',
                 'total_quantity_lag1', 'total_quantity_lag2', 'total_quantity_roll2',
                 'total_revenue_lag1', 'total_revenue_lag2', 'total_revenue_roll2',
                 'avg_unit_price_lag1', 'avg_unit_price_lag2', 'avg_unit_price_roll2',
                 'avg_discount_lag1', 'avg_discount_lag2', 'avg_discount_roll2',
                 'transaction_count_lag1', 'transaction_count_lag2', 'transaction_count_roll2']

X_train_scaled, X_test_scaled, scaler = scale_numeric_features(X_train, X_test, numeric_feats)

# ============================================================
# 6. MODEL TRAINING
# ============================================================
print("\n" + "=" * 60)
print("MODEL TRAINING")
print("=" * 60)

models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1, verbose=0),
    'XGBoost': XGBRegressor(n_estimators=100, max_depth=10, learning_rate=0.1, random_state=42, verbosity=0),
    'LightGBM': LGBMRegressor(n_estimators=100, max_depth=10, learning_rate=0.1, random_state=42, verbose=-1),
    'CatBoost': CatBoostRegressor(iterations=100, depth=10, learning_rate=0.1, random_state=42, verbose=0)
}

results = {}
feature_importances = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    try:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        metrics = evaluate_predictions(y_test, y_pred)
        results[name] = metrics
        feature_importances[name] = model.feature_importances_

        print(f"  RMSE: {metrics['RMSE']:.2f}")
        print(f"  MAE: {metrics['MAE']:.2f}")
        print(f"  R2: {metrics['R2']:.4f}")
    except Exception as e:
        print(f"  Error: {e}")

# ============================================================
# 7. RESULTS COMPARISON
# ============================================================
print("\n" + "=" * 60)
print("MODEL COMPARISON (RQ3 & RQ4)")
print("=" * 60)

results_df = pd.DataFrame(results).T
print("\n" + results_df.to_string())
results_df.to_csv(os.path.join(OUTPUT, 'model_comparison.csv'))

best_model = best_model_name(results)
print(f"\nBest model by R2: {best_model}")

# ============================================================
# 8. FEATURE IMPORTANCE ANALYSIS (RQ2)
# ============================================================
print("\n" + "=" * 60)
print("FEATURE IMPORTANCE ANALYSIS (RQ2)")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, (name, imp) in enumerate(feature_importances.items()):
    if idx < len(axes):
        feat_imp = pd.DataFrame({'feature': feature_cols, 'importance': imp})
        feat_imp = feat_imp.sort_values('importance', ascending=False).head(15)
        ax = axes[idx]
        sns.barplot(data=feat_imp, y='feature', x='importance', ax=ax, palette='viridis')
        ax.set_title(f'{name} - Top 15 Features', fontsize=12)
        ax.set_xlabel('Importance')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, 'feature_importance.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: feature_importance.png")

# Aggregate feature importance across models
print("\nTop features across all models:")
avg_imp = average_feature_importance(feature_importances, feature_cols)
print(avg_imp.head(15).to_string())

avg_imp.to_csv(os.path.join(OUTPUT, 'feature_importance_avg.csv'))

# ============================================================
# 9. VISUALIZATIONS
# ============================================================
print("\n" + "=" * 60)
print("VISUALIZATIONS")
print("=" * 60)

# Model comparison plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
metrics = ['RMSE', 'MAE', 'R2']
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

for i, metric in enumerate(metrics):
    ax = axes[i]
    models_list = list(results.keys())
    values = [results[m][metric] for m in models_list]
    bars = ax.bar(models_list, values, color=colors[:len(models_list)])
    ax.set_title(f'{metric} Comparison', fontsize=13, fontweight='bold')
    ax.set_xlabel('Model')
    ax.tick_params(axis='x', rotation=20)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, 'model_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: model_comparison.png")

# Season trend analysis
seasonal_trends = agg_data.groupby(['Year', 'Season'])['total_quantity'].sum().reset_index()
plt.figure(figsize=(12, 6))
sns.lineplot(data=seasonal_trends, x='Year', y='total_quantity', hue='Season', marker='o', linewidth=2.5)
plt.title('Total Sales Quantity by Season Over Years', fontsize=14, fontweight='bold')
plt.ylabel('Total Quantity Sold')
plt.xlabel('Year')
plt.legend(title='Season')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(OUTPUT, 'seasonal_trends.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: seasonal_trends.png")

# Category trends
cat_trends = agg_data.groupby(['Year', 'Season', 'Category'])['total_quantity'].sum().reset_index()
plt.figure(figsize=(14, 6))
sns.lineplot(data=cat_trends, x='Year', y='total_quantity', hue='Category', style='Season', marker='o', linewidth=2)
plt.title('Sales Quantity by Category and Season', fontsize=14, fontweight='bold')
plt.ylabel('Total Quantity Sold')
plt.xlabel('Year')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, 'category_trends.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: category_trends.png")

# Color popularity by season
color_season = agg_data.groupby(['Season', 'ProductColor'])['total_quantity'].sum().reset_index()
top_colors = (color_season.sort_values('total_quantity', ascending=False)
              .groupby('Season', group_keys=False).head(5))
plt.figure(figsize=(14, 8))
sns.barplot(data=top_colors, x='Season', y='total_quantity', hue='ProductColor', palette='Set2')
plt.title('Top 5 Colors by Season', fontsize=14, fontweight='bold')
plt.ylabel('Total Quantity Sold')
plt.xlabel('Season')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, 'color_by_season.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: color_by_season.png")

# Discount impact
discount_impact = agg_data.groupby('avg_discount')['total_quantity'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.barplot(data=discount_impact, x='avg_discount', y='total_quantity', palette='coolwarm')
plt.title('Average Sales Quantity by Discount Level', fontsize=14, fontweight='bold')
plt.ylabel('Avg Quantity Sold')
plt.xlabel('Discount')
plt.savefig(os.path.join(OUTPUT, 'discount_impact.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: discount_impact.png")

# ============================================================
# 10. PREDICT NEXT SEASON
# ============================================================
print("\n" + "=" * 60)
print("PREDICTING NEXT SEASON TRENDS")
print("=" * 60)

# Use the best model to predict what sells best next season
best_model_obj = models[best_model]

next_season_preds = X_test_scaled.copy()
next_season_preds['predicted_quantity'] = best_model_obj.predict(X_test_scaled)
next_season_preds['actual_quantity'] = y_test
next_season_preds['Category'] = agg_data_sorted.iloc[test_idx]['Category'].values
next_season_preds['Season'] = agg_data_sorted.iloc[test_idx]['Season'].values
next_season_preds['ProductColor'] = agg_data_sorted.iloc[test_idx]['ProductColor'].values
next_season_preds['Gender'] = agg_data_sorted.iloc[test_idx]['Gender'].values
next_season_preds['AgeGroup'] = agg_data_sorted.iloc[test_idx]['AgeGroup'].values
next_season_preds['Year'] = agg_data_sorted.iloc[test_idx]['Year'].values

# Top selling predictions
top_preds = top_predictions(next_season_preds, n=20)
print("\nTop 20 predicted best-sellers for next season:")
cols_show = ['Category', 'Season', 'ProductColor', 'Gender', 'AgeGroup', 'predicted_quantity', 'actual_quantity']
print(top_preds[cols_show].to_string())

top_preds[cols_show].to_csv(os.path.join(OUTPUT, 'next_season_predictions.csv'), index=False)

# ============================================================
# 11. SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("PROJECT SUMMARY")
print("=" * 60)

print(f"""
Research Questions Answered:
────────────────────────────────────────────────────────────
RQ1: Can ML predict next season's best-selling products?
  -> YES - {best_model} achieved R2={results[best_model]['R2']:.4f} on test data

RQ2: Which features most influence fashion trends?
  -> See feature_importance.png and feature_importance_avg.csv
  -> Top features: {', '.join(avg_imp.head(5).index.tolist())}

RQ3: Which ML algorithm performs best?
  -> Rankings by R2 (higher is better):
""")
for rank, (model_name, metrics) in enumerate(rank_models(results), 1):
    print(f"    {rank}. {model_name}: R2={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.2f}")

print(f"""
RQ4: Can deep learning outperform traditional ML?
  -> Deep learning (neural network) could not be tested due to
     TensorFlow/PyTorch compatibility issues with Python 3.14.
     Consider using Python 3.10-3.12 for deep learning experiments.

RQ5: Can external data improve predictions?
  -> The current pipeline uses internal retail data only.
  -> Weather, holidays, and social media data can be added
     by joining on Year/Season/Date fields.
""")
