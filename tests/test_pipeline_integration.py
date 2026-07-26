"""End-to-end checks of the preprocessing pipeline on a tiny synthetic dataset."""
import numpy as np
import pandas as pd

from fashion_trend_utils import (
    add_customer_features,
    add_product_features,
    add_transaction_features,
    create_lag_features,
    encode_categoricals,
    evaluate_predictions,
    scale_numeric_features,
    time_based_split,
)

GROUP_COLS = ['Category', 'ProductColor', 'Gender', 'AgeGroup']
LAG_COLS = ['total_revenue', 'avg_unit_price']
TARGET = 'total_quantity'


def _raw_tables():
    dates = pd.date_range('2022-01-15', periods=24, freq='MS')
    transactions = pd.DataFrame({
        'Date': dates,
        'Product ID': [1, 2] * 12,
        'Customer ID': [10, 11] * 12,
        'Quantity': np.arange(1, 25),
        'Line Total': np.arange(1, 25) * 10.0,
        'Unit Price': 10.0,
        'Discount': 0.1,
    })
    products = pd.DataFrame({
        'Product ID': [1, 2],
        'Category': ['Feminine', 'Masculine'],
        'Color': ['RED', None],
        'Production Cost': [3.0, 4.0],
    })
    customers = pd.DataFrame({
        'Customer ID': [10, 11],
        'Date Of Birth': ['1990-01-01', '2005-01-01'],
        'Gender': ['F', 'M'],
    })
    return transactions, products, customers


def _aggregate():
    transactions, products, customers = _raw_tables()
    data = add_transaction_features(transactions)
    data = data.merge(
        add_product_features(products)[['Product ID', 'Category', 'ProductColor', 'Production Cost']],
        on='Product ID', how='left',
    )
    data = data.merge(
        add_customer_features(customers)[['Customer ID', 'Gender', 'Age', 'AgeGroup']],
        on='Customer ID', how='left',
    )
    return data.groupby(['Category', 'Season', 'Year', 'ProductColor', 'Gender', 'AgeGroup'], observed=True).agg(
        total_quantity=('Quantity', 'sum'),
        total_revenue=('Line Total', 'sum'),
        avg_unit_price=('Unit Price', 'mean'),
        avg_discount=('Discount', 'mean'),
        transaction_count=('Date', 'count'),
        avg_age=('Age', 'mean'),
    ).reset_index()


def test_pipeline_produces_usable_feature_matrix():
    agg = create_lag_features(_aggregate(), GROUP_COLS, TARGET, LAG_COLS, n_lags=2)
    agg = agg.dropna().reset_index(drop=True)
    agg, encoders = encode_categoricals(agg, ['Category', 'Season', 'ProductColor', 'Gender', 'AgeGroup'])

    assert not agg.empty
    assert set(encoders) == {'Category', 'Season', 'ProductColor', 'Gender', 'AgeGroup'}
    assert agg[[f'{TARGET}_lag1', f'{TARGET}_lag2']].notna().all().all()
    assert agg['TimeStep'].between(2022 * 4, 2024 * 4 + 3).all()


def test_pipeline_split_and_scaling_are_leakage_free():
    agg = create_lag_features(_aggregate(), GROUP_COLS, TARGET, LAG_COLS, n_lags=2).dropna().reset_index(drop=True)
    train, test, split_idx = time_based_split(agg, test_size=0.25)

    numeric_feats = [f'{TARGET}_lag1', 'total_revenue_lag1']
    train_scaled, test_scaled, scaler = scale_numeric_features(train, test, numeric_feats)

    assert split_idx == len(train)
    assert train['TimeStep'].max() <= test['TimeStep'].min()
    assert train_scaled[numeric_feats].mean().abs().max() < 1e-9
    assert scaler.n_features_in_ == len(numeric_feats)
    # a naive "repeat last season" forecast should score better than random noise
    baseline = evaluate_predictions(test[TARGET], test[f'{TARGET}_lag1'])
    noise = evaluate_predictions(test[TARGET], np.zeros(len(test)))
    assert baseline['RMSE'] < noise['RMSE']
