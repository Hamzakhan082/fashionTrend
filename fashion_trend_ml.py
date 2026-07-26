"""
What is in this season? Using machine learning to identify fashion trends
Supervisor: Dr Ollie Bartlett
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import fashion_trend_utils as ftu

# ============================================================
# 1. DATA LOADING & PREPROCESSING
# ============================================================
ftu.print_section("LOADING AND PREPROCESSING DATA")

data = ftu.load_and_prepare_data()
print(f"Memory usage: {data.memory_usage(deep=True).sum() / 1e9:.2f} GB")

# ============================================================
# 2. FEATURE ENGINEERING FOR SEASONAL TREND PREDICTION
# ============================================================
ftu.print_section("FEATURE ENGINEERING FOR SEASONAL AGGREGATION")

# We predict next season's total sales quantity per product category
agg_data = ftu.aggregate_by_season(data)
agg_data = ftu.add_lag_features(agg_data)

print(f"Aggregated data shape: {agg_data.shape}")
print(f"Seasons covered: {sorted(agg_data['Year'].unique())} - {sorted(agg_data['Season'].unique())}")

# ============================================================
# 3. ENCODE CATEGORICAL FEATURES
# ============================================================
ftu.print_section("ENCODING CATEGORICAL VARIABLES")

ftu.encode_categoricals(agg_data, verbose=True)

X, y, feature_cols, numeric_feats = ftu.build_features(agg_data)

print(f"Features: {len(feature_cols)}")
print(f"Samples: {len(X)}")

# ============================================================
# 4. TRAIN/TEST SPLIT (Time-based: train on past, test on future)
# ============================================================
ftu.print_section("TRAIN/TEST SPLIT (Time-based)")

X_train, X_test, y_train, y_test, agg_data_sorted, split_idx = ftu.time_based_split(
    agg_data, X, y
)
train_meta = agg_data_sorted.iloc[:split_idx]
test_meta = agg_data_sorted.iloc[split_idx:]

print(f"Train: {len(X_train)}, Test: {len(X_test)}")
print(f"Train time range: {train_meta['Year'].min()}-{train_meta['Season'].iloc[-1]} {train_meta['Year'].iloc[-1]}")
print(f"Test time range: {test_meta['Year'].min()}-{test_meta['Season'].iloc[-1]} {test_meta['Year'].iloc[-1]}")

X_train_scaled, X_test_scaled, scaler = ftu.scale_features(X_train, X_test, numeric_feats)

# ============================================================
# 5. MODEL TRAINING
# ============================================================
ftu.print_section("MODEL TRAINING")

models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1, verbose=0),
    'XGBoost': XGBRegressor(n_estimators=100, max_depth=10, learning_rate=0.1, random_state=42, verbosity=0),
    'LightGBM': LGBMRegressor(n_estimators=100, max_depth=10, learning_rate=0.1, random_state=42, verbose=-1),
    'CatBoost': CatBoostRegressor(iterations=100, depth=10, learning_rate=0.1, random_state=42, verbose=0)
}

fitted_models = {}
feature_importances = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    try:
        model.fit(X_train_scaled, y_train)
        fitted_models[name] = model
        feature_importances[name] = model.feature_importances_
    except Exception as e:
        print(f"  Error: {e}")

# ============================================================
# 6. RESULTS COMPARISON
# ============================================================
ftu.print_section("MODEL COMPARISON (RQ3 & RQ4)")

results = ftu.evaluate_models(fitted_models, X_test_scaled, y_test)
results_df = ftu.save_results_table(results, 'model_comparison.csv')
print("\n" + results_df.to_string())

best_model = ftu.best_model_name(results)
print(f"\nBest model by R2: {best_model}")

# ============================================================
# 7. FEATURE IMPORTANCE ANALYSIS (RQ2)
# ============================================================
ftu.print_section("FEATURE IMPORTANCE ANALYSIS (RQ2)")

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

ftu.save_figure('feature_importance.png')

# Aggregate feature importance across models
print("\nTop features across all models:")
all_importances = []
for name, imp in feature_importances.items():
    for i, f in enumerate(feature_cols):
        all_importances.append({'Model': name, 'Feature': f, 'Importance': imp[i]})

imp_df = pd.DataFrame(all_importances)
avg_imp = imp_df.groupby('Feature')['Importance'].mean().sort_values(ascending=False)
print(avg_imp.head(15).to_string())

avg_imp.to_csv(ftu.output_path('feature_importance_avg.csv'))

# ============================================================
# 8. VISUALIZATIONS
# ============================================================
ftu.print_section("VISUALIZATIONS")

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

ftu.save_figure('model_comparison.png')

# Season trend analysis
seasonal_trends = agg_data.groupby(['Year', 'Season'])['total_quantity'].sum().reset_index()
plt.figure(figsize=(12, 6))
sns.lineplot(data=seasonal_trends, x='Year', y='total_quantity', hue='Season', marker='o', linewidth=2.5)
plt.title('Total Sales Quantity by Season Over Years', fontsize=14, fontweight='bold')
plt.ylabel('Total Quantity Sold')
plt.xlabel('Year')
plt.legend(title='Season')
plt.grid(True, alpha=0.3)
ftu.save_figure('seasonal_trends.png', tight_layout=False)

# Category trends
cat_trends = agg_data.groupby(['Year', 'Season', 'Category'])['total_quantity'].sum().reset_index()
plt.figure(figsize=(14, 6))
sns.lineplot(data=cat_trends, x='Year', y='total_quantity', hue='Category', style='Season', marker='o', linewidth=2)
plt.title('Sales Quantity by Category and Season', fontsize=14, fontweight='bold')
plt.ylabel('Total Quantity Sold')
plt.xlabel('Year')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
ftu.save_figure('category_trends.png')

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
ftu.save_figure('color_by_season.png')

# Discount impact
discount_impact = agg_data.groupby('avg_discount')['total_quantity'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.barplot(data=discount_impact, x='avg_discount', y='total_quantity', palette='coolwarm')
plt.title('Average Sales Quantity by Discount Level', fontsize=14, fontweight='bold')
plt.ylabel('Avg Quantity Sold')
plt.xlabel('Discount')
ftu.save_figure('discount_impact.png', tight_layout=False)

# ============================================================
# 9. PREDICT NEXT SEASON
# ============================================================
ftu.print_section("PREDICTING NEXT SEASON TRENDS")

# Use the best model to predict what sells best next season
best_model_obj = fitted_models[best_model]
top_preds = ftu.save_predictions(
    agg_data_sorted, split_idx, best_model_obj.predict(X_test_scaled), y_test,
    'next_season_predictions.csv'
)

print("\nTop 20 predicted best-sellers for next season:")
print(top_preds.to_string())

# ============================================================
# 10. SUMMARY
# ============================================================
ftu.print_section("PROJECT SUMMARY")

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
for rank, (model_name, metrics) in enumerate(sorted(results.items(), key=lambda x: x[1]['R2'], reverse=True), 1):
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
