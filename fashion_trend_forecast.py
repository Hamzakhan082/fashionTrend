"""
Forecast the next (unobserved) season's best-selling segments.

Unlike `fashion_trend_ml.py`, which backtests on held-out past seasons, this script
trains on every observed season and then predicts the season that follows the data,
rolling each segment's latest observations forward into its lag/rolling features.
A walk-forward check first re-fits on everything except the final observed season and
scores the forecast for that season, so the reported forecast comes with an honest
out-of-sample error estimate.
"""

import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

import fashion_trend_utils as ftu

TOP_N = 20


def make_model():
    return XGBRegressor(n_estimators=100, max_depth=10, learning_rate=0.1,
                        random_state=42, verbosity=0)


def fit_and_forecast(train_data, forecast_source, numeric_feats):
    """Fit on `train_data` and predict the season following `forecast_source`."""
    X_train = train_data[ftu.feature_columns(numeric_feats)].copy()
    y_train = train_data[ftu.TARGET_COL].values

    encoders = {col: LabelEncoder().fit(train_data[col]) for col in ftu.CAT_COLS}
    future = ftu.build_next_season_features(forecast_source, encoders)
    X_future = future[ftu.feature_columns(numeric_feats)].copy()

    X_train_scaled, X_future_scaled, _ = ftu.scale_features(X_train, X_future, numeric_feats)
    model = make_model().fit(X_train_scaled, y_train)
    future['predicted_quantity'] = model.predict(X_future_scaled)
    return future


ftu.print_section("LOADING DATA AND BUILDING SEASONAL FEATURES")

data = ftu.load_and_prepare_data()

# Trailing partially-observed seasons would be forecast against incomplete actuals
# and would feed misleadingly low lag values into the forecast, so they are dropped.
while True:
    coverage = data.groupby(['Year', 'Season'], observed=True)['Month'].nunique().reset_index()
    coverage = ftu.add_time_index(coverage).sort_values('TimeStep')
    latest = coverage.iloc[-1]
    if latest['Month'] >= 3:
        break
    print(f"Dropping partial season {latest['Season']} {int(latest['Year'])} "
          f"({int(latest['Month'])}/3 months observed)")
    data = data[(data['Year'] != latest['Year']) | (data['Season'] != latest['Season'])]

agg_data = ftu.add_lag_features(ftu.aggregate_by_season(data))
ftu.encode_categoricals(agg_data, verbose=True)
_, _, feature_cols, numeric_feats = ftu.build_features(agg_data)

last_step = agg_data['TimeStep'].max()
last_row = agg_data[agg_data['TimeStep'] == last_step].iloc[0]
print(f"Observed seasons: {agg_data['TimeStep'].nunique()} "
      f"(latest: {last_row['Season']} {int(last_row['Year'])})")

# ============================================================
# WALK-FORWARD CHECK ON THE LAST OBSERVED SEASON
# ============================================================
ftu.print_section("WALK-FORWARD CHECK (forecasting the last observed season)")

history = agg_data[agg_data['TimeStep'] < last_step]
holdout = agg_data[agg_data['TimeStep'] == last_step]

forecast_holdout = fit_and_forecast(history, history, numeric_feats)
compared = forecast_holdout.merge(
    holdout[ftu.GROUP_COLS + ['TimeStep', ftu.TARGET_COL]],
    on=ftu.GROUP_COLS + ['TimeStep'], how='inner',
)

if compared.empty:
    print("No overlapping segments to score.")
else:
    metrics = ftu.regression_metrics(compared[ftu.TARGET_COL], compared['predicted_quantity'])
    print(f"Segments scored: {len(compared)} of {len(holdout)} in {last_row['Season']} "
          f"{int(last_row['Year'])}")
    print(f"RMSE: {metrics['RMSE']:.2f} | MAE: {metrics['MAE']:.2f} | R2: {metrics['R2']:.4f}")

    top_actual = set(compared.nlargest(TOP_N, ftu.TARGET_COL)
                     .set_index(ftu.GROUP_COLS).index)
    top_pred = set(compared.nlargest(TOP_N, 'predicted_quantity')
                   .set_index(ftu.GROUP_COLS).index)
    print(f"Top-{TOP_N} best-seller overlap: {len(top_actual & top_pred)}/{TOP_N}")

# ============================================================
# FORECAST THE NEXT UNOBSERVED SEASON
# ============================================================
ftu.print_section("NEXT SEASON FORECAST")

forecast = fit_and_forecast(agg_data, agg_data, numeric_feats)
forecast = forecast.sort_values('predicted_quantity', ascending=False)

cols_show = ftu.GROUP_COLS + ['Season', 'Year', 'predicted_quantity']
print(f"Forecast horizon: {forecast['Season'].iloc[0]} {int(forecast['Year'].iloc[0])}")
print(f"\nTop {TOP_N} predicted best-sellers:")
print(forecast.head(TOP_N)[cols_show].to_string(index=False))

forecast[cols_show].to_csv(ftu.output_path('next_season_forecast.csv'), index=False)
print(f"\nSaved: next_season_forecast.csv ({len(forecast)} segments)")
