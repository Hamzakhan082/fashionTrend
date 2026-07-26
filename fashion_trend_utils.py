"""
Reusable, side-effect free helpers shared by the fashion trend pipelines.

Importing this module never reads data files or trains models, so every
function here can be unit tested in isolation.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

SEASON_ORDER = {'Spring': 0, 'Summer': 1, 'Fall': 2, 'Winter': 3}

AGE_BINS = [0, 18, 25, 35, 50, 65, 120]
AGE_LABELS = ['0-18', '19-25', '26-35', '36-50', '51-65', '65+']

REFERENCE_YEAR = 2024

SEASONS_BY_MONTH = {
    12: 'Winter', 1: 'Winter', 2: 'Winter',
    3: 'Spring', 4: 'Spring', 5: 'Spring',
    6: 'Summer', 7: 'Summer', 8: 'Summer',
    9: 'Fall', 10: 'Fall', 11: 'Fall',
}


def get_season(month):
    """Map a calendar month number to a meteorological season name."""
    try:
        return SEASONS_BY_MONTH[int(month)]
    except (TypeError, ValueError, KeyError):
        raise ValueError(f'Invalid month: {month!r}')


def add_customer_features(customers, reference_year=REFERENCE_YEAR):
    """Add ``Age`` and binned ``AgeGroup`` columns derived from date of birth."""
    customers = customers.copy()
    birth_year = pd.to_datetime(customers['Date Of Birth'], errors='coerce').dt.year
    customers['Age'] = reference_year - birth_year
    customers['AgeGroup'] = pd.cut(customers['Age'], bins=AGE_BINS, labels=AGE_LABELS)
    return customers


def add_product_features(products):
    """Normalise product colours and sizes, replacing ``Color`` with ``ProductColor``."""
    products = products.copy()
    products['ProductColor'] = products['Color'].fillna('Unknown') if 'Color' in products else 'Unknown'
    products = products.drop(columns=['Color'], errors='ignore')
    if 'Sizes' in products:
        products['Sizes'] = products['Sizes'].fillna('Unknown')
    return products


def add_transaction_features(transactions):
    """Add calendar features (year, month, quarter, weekday, weekend flag, season)."""
    transactions = transactions.copy()
    transactions['Date'] = pd.to_datetime(transactions['Date'], errors='coerce')
    transactions['Year'] = transactions['Date'].dt.year
    transactions['Month'] = transactions['Date'].dt.month
    transactions['Quarter'] = transactions['Date'].dt.quarter
    transactions['Weekday'] = transactions['Date'].dt.weekday
    transactions['IsWeekend'] = (transactions['Weekday'] >= 5).astype(int)
    transactions['Season'] = transactions['Month'].map(SEASONS_BY_MONTH)
    return transactions


def add_time_step(df):
    """Add ``SeasonNum`` and a monotonically increasing ``TimeStep`` per season."""
    df = df.copy()
    df['SeasonNum'] = df['Season'].map(SEASON_ORDER)
    df['TimeStep'] = df['Year'] * 4 + df['SeasonNum']
    return df


def create_lag_features(df, group_cols, target_col, lag_cols, n_lags=2, window=2):
    """Create per-group lag and rolling-mean features for time series prediction.

    Rows are ordered by ``TimeStep`` so lags follow the seasonal calendar
    (Spring, Summer, Fall, Winter) rather than alphabetical season names.
    """
    df = add_time_step(df).sort_values('TimeStep').reset_index(drop=True)

    for col in list(lag_cols) + [target_col]:
        grouped = df.groupby(group_cols, group_keys=False, observed=False)[col]
        for lag in range(1, n_lags + 1):
            df[f'{col}_lag{lag}'] = grouped.shift(lag)
        df[f'{col}_roll{window}'] = grouped.transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )

    return df


def encode_categoricals(df, cat_cols, suffix='_enc'):
    """Label-encode ``cat_cols`` in place, returning the frame and fitted encoders."""
    df = df.copy()
    encoders = {}
    for col in cat_cols:
        encoder = LabelEncoder()
        df[col + suffix] = encoder.fit_transform(df[col])
        encoders[col] = encoder
    return df, encoders


def time_based_split(df, test_size=0.2, time_col='TimeStep'):
    """Sort chronologically and split into past (train) and future (test) frames."""
    if not 0 < test_size < 1:
        raise ValueError(f'test_size must be in (0, 1), got {test_size}')
    ordered = df.sort_values(time_col).reset_index(drop=True)
    split_idx = int(len(ordered) * (1 - test_size))
    return ordered.iloc[:split_idx], ordered.iloc[split_idx:], split_idx


def scale_numeric_features(train, test, numeric_feats):
    """Standardise ``numeric_feats``, fitting the scaler on the training rows only."""
    train_scaled, test_scaled = train.copy(), test.copy()
    scaler = StandardScaler()
    train_scaled[numeric_feats] = scaler.fit_transform(train[numeric_feats])
    test_scaled[numeric_feats] = scaler.transform(test[numeric_feats])
    return train_scaled, test_scaled, scaler


def evaluate_predictions(y_true, y_pred):
    """Return RMSE, MAE and R2 for a set of predictions."""
    return {
        'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'MAE': float(mean_absolute_error(y_true, y_pred)),
        'R2': float(r2_score(y_true, y_pred)),
    }


def rank_models(results, metric='R2', higher_is_better=True):
    """Rank ``{model: {metric: value}}`` results best-first by ``metric``."""
    if not results:
        return []
    return sorted(results.items(), key=lambda item: item[1][metric], reverse=higher_is_better)


def best_model_name(results, metric='R2', higher_is_better=True):
    """Name of the best scoring model, or ``None`` when there are no results."""
    ranked = rank_models(results, metric=metric, higher_is_better=higher_is_better)
    return ranked[0][0] if ranked else None


def average_feature_importance(feature_importances, feature_cols):
    """Mean importance per feature across models, sorted descending."""
    records = [
        {'Model': name, 'Feature': feature, 'Importance': importances[i]}
        for name, importances in feature_importances.items()
        for i, feature in enumerate(feature_cols)
    ]
    imp_df = pd.DataFrame(records, columns=['Model', 'Feature', 'Importance'])
    if imp_df.empty:
        return pd.Series(dtype=float, name='Importance')
    return imp_df.groupby('Feature')['Importance'].mean().sort_values(ascending=False)


def top_predictions(df, n=20, pred_col='predicted_quantity'):
    """Highest ``n`` rows by predicted quantity."""
    return df.nlargest(n, pred_col)
