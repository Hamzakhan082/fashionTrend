"""Shared utilities for the fashion trend prediction pipeline.

Data loading, preprocessing, aggregation, feature engineering, splitting,
scaling and evaluation helpers used by the analysis scripts and notebook.
"""

import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

BASE = os.environ.get('FASHION_TREND_DATA_DIR', r'C:\Users\hamza\Machine Learning Projects')
OUTPUT = os.environ.get(
    'FASHION_TREND_OUTPUT_DIR',
    r'C:\Users\hamza\Machine Learning Projects\Fashion Trend dataset',
)

DATASET_NAMES = ['customers', 'products', 'transactions', 'stores', 'discounts']

# Columns the pipeline actually consumes. Restricting the read keeps the 6.4M-row
# transaction table (and the merged frame) within a few GB of memory.
PIPELINE_COLUMNS = {
    'customers': ['Customer ID', 'Date Of Birth', 'Gender', 'City', 'Country'],
    'products': ['Product ID', 'Category', 'Sub Category', 'Color', 'Sizes', 'Production Cost'],
    'transactions': ['Customer ID', 'Product ID', 'Store ID', 'Unit Price', 'Quantity', 'Date',
                     'Discount', 'Line Total'],
    'stores': ['Store ID', 'Country', 'City'],
    'discounts': None,
}

SEASON_ORDER = {'Spring': 0, 'Summer': 1, 'Fall': 2, 'Winter': 3}
SEASON_BY_NUM = {num: name for name, num in SEASON_ORDER.items()}
SEASON_BY_MONTH = {
    12: 'Winter', 1: 'Winter', 2: 'Winter',
    3: 'Spring', 4: 'Spring', 5: 'Spring',
    6: 'Summer', 7: 'Summer', 8: 'Summer',
    9: 'Fall', 10: 'Fall', 11: 'Fall',
}

AGE_BINS = [0, 18, 25, 35, 50, 65, 120]
AGE_LABELS = ['0-18', '19-25', '26-35', '36-50', '51-65', '65+']
REFERENCE_YEAR = 2024

GROUP_COLS = ['Category', 'ProductColor', 'Gender', 'AgeGroup']
AGG_KEYS = ['Category', 'Season', 'Year', 'ProductColor', 'Gender', 'AgeGroup']
CAT_COLS = ['Category', 'Season', 'ProductColor', 'Gender', 'AgeGroup']
TARGET_COL = 'total_quantity'
LAG_COLS = ['total_revenue', 'avg_unit_price', 'avg_discount', 'transaction_count']
LAG_FEATURE_BASES = LAG_COLS + [TARGET_COL]


def print_section(title):
    """Print a banner-style section header."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def data_path(filename):
    return os.path.join(BASE, filename)


def output_path(filename):
    return os.path.join(OUTPUT, filename)


def load_datasets(names=DATASET_NAMES, verbose=True, columns=PIPELINE_COLUMNS):
    """Load the raw CSV tables, keyed by dataset name.

    `columns` maps a dataset name to the columns to read (None reads every column);
    pass `columns={}` to always read full tables.
    """
    datasets = {}
    for name in names:
        usecols = (columns or {}).get(name)
        datasets[name] = pd.read_csv(data_path(f'{name}.csv'), low_memory=False,
                                     usecols=usecols)
        if verbose:
            print(f"{name.capitalize()}: {datasets[name].shape}")
    return datasets


def get_season(month):
    """Map a calendar month to its meteorological season."""
    return SEASON_BY_MONTH.get(month)


def preprocess_customers(customers):
    customers['Age'] = REFERENCE_YEAR - pd.to_datetime(
        customers['Date Of Birth'], errors='coerce'
    ).dt.year
    customers['AgeGroup'] = pd.cut(customers['Age'], bins=AGE_BINS, labels=AGE_LABELS)
    return customers


def preprocess_products(products):
    products['ProductColor'] = products['Color'].fillna('Unknown')
    products = products.drop(columns=['Color'], errors='ignore')
    if 'Sizes' in products.columns:
        products['Sizes'] = products['Sizes'].fillna('Unknown')
    return products


def preprocess_transactions(transactions):
    transactions['Date'] = pd.to_datetime(transactions['Date'], errors='coerce')
    transactions['Year'] = transactions['Date'].dt.year
    transactions['Month'] = transactions['Date'].dt.month
    transactions['Quarter'] = transactions['Date'].dt.quarter
    transactions['Weekday'] = transactions['Date'].dt.weekday  # 0=Monday
    transactions['IsWeekend'] = (transactions['Weekday'] >= 5).astype(int)
    transactions['Season'] = transactions['Month'].map(SEASON_BY_MONTH)
    return transactions


def merge_datasets(transactions, products, customers, stores):
    """Join transactions with product, customer and store attributes."""
    data = transactions.merge(
        products[['Product ID', 'Category', 'Sub Category', 'ProductColor', 'Sizes',
                  'Production Cost']],
        on='Product ID', how='left',
    )
    data = data.merge(
        customers[['Customer ID', 'Gender', 'Age', 'AgeGroup', 'City', 'Country']],
        on='Customer ID', how='left',
    )
    data = data.merge(
        stores[['Store ID', 'Country', 'City']],
        on='Store ID', how='left', suffixes=('_cust', '_store'),
    )
    return data


def load_and_prepare_data(verbose=True):
    """Load the raw tables, preprocess them and return the merged transaction frame."""
    datasets = load_datasets(verbose=verbose)
    customers = preprocess_customers(datasets['customers'])
    products = preprocess_products(datasets['products'])
    transactions = preprocess_transactions(datasets['transactions'])
    data = merge_datasets(transactions, products, customers, datasets['stores'])
    if verbose:
        print(f"Merged data shape: {data.shape}")
    return data


def aggregate_by_season(data):
    """Aggregate transaction-level data into one row per category/season/segment."""
    return data.groupby(AGG_KEYS).agg(
        total_quantity=('Quantity', 'sum'),
        total_revenue=('Line Total', 'sum'),
        avg_unit_price=('Unit Price', 'mean'),
        avg_discount=('Discount', 'mean'),
        transaction_count=('Date', 'count'),
        avg_age=('Age', 'mean'),
        city_count=('City_cust', 'nunique'),
        store_count=('Store ID', 'nunique'),
        production_cost=('Production Cost', 'mean'),
    ).reset_index()


def add_time_index(df):
    """Add ordinal season/time-step columns derived from Season and Year."""
    df['SeasonNum'] = df['Season'].map(SEASON_ORDER)
    df['TimeStep'] = df['Year'] * 4 + df['SeasonNum']
    return df


def add_lag_features(df, group_cols=GROUP_COLS, target_col=TARGET_COL, lag_cols=LAG_COLS,
                     n_lags=2, roll_window=2):
    """Add lag and rolling-average features per group for time series prediction."""
    df = add_time_index(df)
    df = df.sort_values(list(group_cols) + ['TimeStep']).reset_index(drop=True)

    for col in list(lag_cols) + [target_col]:
        grouped = df.groupby(group_cols, group_keys=False)[col]
        for lag in range(1, n_lags + 1):
            df[f'{col}_lag{lag}'] = grouped.shift(lag)
        df[f'{col}_roll{roll_window}'] = grouped.transform(
            lambda x: x.rolling(window=roll_window, min_periods=1).mean()
        )

    return df.dropna().reset_index(drop=True)


def next_season(year, season):
    """The (year, season) that follows the given one in calendar order."""
    season_num = SEASON_ORDER[season]
    next_num = (season_num + 1) % 4
    next_name = SEASON_BY_NUM[next_num]
    return (year + 1 if next_num == 0 else year), next_name


def build_next_season_features(agg_data, encoders, group_cols=GROUP_COLS, target_col=TARGET_COL,
                               lag_cols=LAG_COLS, n_lags=2, roll_window=2,
                               static_cols=('avg_age', 'city_count', 'store_count',
                                            'production_cost', 'avg_unit_price', 'avg_discount')):
    """Build one unobserved next-season row per group, ready to be predicted.

    Each group's most recent observed seasons are rolled forward: `lag1` becomes the
    latest observed value, `lag2` the one before it, and `roll{roll_window}` the mean of
    the last `roll_window` observations. Static segment attributes are carried forward.
    Returns a frame with the identity columns, the derived features, and no target.
    """
    history = agg_data.sort_values(list(group_cols) + ['TimeStep'])
    rows = []

    for keys, group in history.groupby(list(group_cols), observed=True):
        latest = group.iloc[-1]
        year, season = next_season(int(latest['Year']), latest['Season'])
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        row['Year'] = year
        row['Season'] = season
        row['SeasonNum'] = SEASON_ORDER[season]
        row['TimeStep'] = year * 4 + row['SeasonNum']

        for col in list(lag_cols) + [target_col]:
            values = group[col].to_numpy()
            for lag in range(1, n_lags + 1):
                row[f'{col}_lag{lag}'] = values[-lag] if len(values) >= lag else np.nan
            row[f'{col}_roll{roll_window}'] = values[-roll_window:].mean()

        for col in static_cols:
            if col in group.columns:
                row[col] = latest[col]

        rows.append(row)

    future = pd.DataFrame(rows).dropna().reset_index(drop=True)
    for col, encoder in encoders.items():
        future[f'{col}_enc'] = encoder.transform(future[col])
    return future


def encode_categoricals(df, cat_cols=CAT_COLS, verbose=False):
    """Label-encode categorical columns into `<col>_enc`, returning the encoders."""
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[f'{col}_enc'] = le.fit_transform(df[col])
        encoders[col] = le
        if verbose:
            print(f"{col}: {len(le.classes_)} categories")
    return encoders


def lag_feature_names(bases=LAG_FEATURE_BASES, n_lags=2, roll_window=2):
    """Names of the lag/rolling columns produced by `add_lag_features`."""
    names = []
    for base in bases:
        names.extend(f'{base}_lag{lag}' for lag in range(1, n_lags + 1))
        names.append(f'{base}_roll{roll_window}')
    return names


def numeric_feature_names(extra=('avg_age', 'city_count', 'store_count', 'production_cost',
                                 'avg_unit_price')):
    return list(extra) + lag_feature_names()


def feature_columns(numeric_feats=None):
    """Model feature list: time index, numeric features and encoded categoricals."""
    numeric_feats = numeric_feature_names() if numeric_feats is None else list(numeric_feats)
    return ['SeasonNum', 'TimeStep'] + numeric_feats + [f'{col}_enc' for col in CAT_COLS]


def build_features(agg_data, numeric_feats=None):
    """Return (X, y, feature_cols, numeric_feats) ready for modelling."""
    numeric_feats = numeric_feature_names() if numeric_feats is None else list(numeric_feats)
    feature_cols = feature_columns(numeric_feats)
    X = agg_data[feature_cols].copy()
    y = agg_data[TARGET_COL].values
    return X, y, feature_cols, numeric_feats


def time_based_split(agg_data, X, y, train_frac=0.8):
    """Split chronologically: train on the past, test on the future.

    Returns (X_train, X_test, y_train, y_test, agg_data_sorted, split_idx), where
    every returned frame/array follows the same chronological ordering.
    """
    order = agg_data['TimeStep'].sort_values(kind='mergesort').index
    agg_data_sorted = agg_data.loc[order].reset_index(drop=True)
    split_idx = int(len(agg_data_sorted) * train_frac)

    X_sorted = X.loc[order].reset_index(drop=True)
    y_sorted = np.asarray(y)[order]

    return (X_sorted.iloc[:split_idx], X_sorted.iloc[split_idx:],
            y_sorted[:split_idx], y_sorted[split_idx:], agg_data_sorted, split_idx)


def scale_features(X_train, X_test, numeric_feats):
    """Standardise numeric features, fitting on the training split only."""
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numeric_feats] = scaler.fit_transform(X_train[numeric_feats])
    X_test_scaled[numeric_feats] = scaler.transform(X_test[numeric_feats])
    return X_train_scaled, X_test_scaled, scaler


def regression_metrics(y_true, y_pred):
    """RMSE / MAE / R2 for a set of predictions."""
    return {
        'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'MAE': float(mean_absolute_error(y_true, y_pred)),
        'R2': float(r2_score(y_true, y_pred)),
    }


def evaluate_models(models, X_test, y_test, verbose=True):
    """Score already-fitted models, returning {name: metrics}."""
    results = {}
    for name, model in models.items():
        metrics = regression_metrics(y_test, model.predict(X_test))
        results[name] = metrics
        if verbose:
            print(f"{name:15s} | RMSE: {metrics['RMSE']:>8.2f} "
                  f"| MAE: {metrics['MAE']:>8.2f} | R2: {metrics['R2']:.4f}")
    return results


def best_model_name(results):
    """Name of the model with the highest R2."""
    return max(results.items(), key=lambda item: item[1]['R2'])[0]


def save_results_table(results, filename):
    """Persist a {name: metrics} mapping as CSV and return the DataFrame."""
    results_df = pd.DataFrame(results).T
    results_df.to_csv(output_path(filename))
    return results_df


def save_figure(filename, dpi=150, tight_layout=True):
    """Save the current matplotlib figure into the output directory and close it."""
    import matplotlib.pyplot as plt

    if tight_layout:
        plt.tight_layout()
    plt.savefig(output_path(filename), dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def save_predictions(agg_data_sorted, split_idx, y_pred, y_true, filename, top_n=20,
                     id_cols=('Category', 'Season', 'ProductColor', 'Gender', 'AgeGroup', 'Year')):
    """Build the top-N predicted best-sellers table, save it and return it."""
    pred_df = agg_data_sorted.iloc[split_idx:][list(id_cols)].reset_index(drop=True)
    pred_df['predicted_quantity'] = y_pred
    pred_df['actual_quantity'] = y_true
    top_preds = pred_df.nlargest(top_n, 'predicted_quantity')
    top_preds.to_csv(output_path(filename), index=False)
    return top_preds
