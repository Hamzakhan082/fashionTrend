import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from fashion_trend_utils import (
    AGE_LABELS,
    SEASON_ORDER,
    add_customer_features,
    add_product_features,
    add_time_step,
    add_transaction_features,
    average_feature_importance,
    best_model_name,
    create_lag_features,
    encode_categoricals,
    evaluate_predictions,
    get_season,
    rank_models,
    scale_numeric_features,
    time_based_split,
    top_predictions,
)


@pytest.mark.parametrize('month,expected', [
    (12, 'Winter'), (1, 'Winter'), (2, 'Winter'),
    (3, 'Spring'), (4, 'Spring'), (5, 'Spring'),
    (6, 'Summer'), (7, 'Summer'), (8, 'Summer'),
    (9, 'Fall'), (10, 'Fall'), (11, 'Fall'),
])
def test_get_season_covers_every_month(month, expected):
    assert get_season(month) == expected


def test_get_season_accepts_numeric_strings_and_floats():
    assert get_season('7') == 'Summer'
    assert get_season(7.0) == 'Summer'


@pytest.mark.parametrize('bad', [0, 13, -1, None, 'July', float('nan')])
def test_get_season_rejects_invalid_months(bad):
    with pytest.raises(ValueError):
        get_season(bad)


def test_add_customer_features_computes_age_and_group():
    customers = pd.DataFrame({
        'Customer ID': [1, 2, 3, 4],
        'Date Of Birth': ['1990-05-01', '2010-01-01', '1950-12-31', 'not-a-date'],
    })

    result = add_customer_features(customers, reference_year=2024)

    assert result['Age'].tolist()[:3] == [34, 14, 74]
    assert pd.isna(result['Age'].iloc[3])
    assert result['AgeGroup'].tolist()[:3] == ['26-35', '0-18', '65+']
    assert pd.isna(result['AgeGroup'].iloc[3])
    assert list(result['AgeGroup'].cat.categories) == AGE_LABELS


def test_add_customer_features_does_not_mutate_input():
    customers = pd.DataFrame({'Date Of Birth': ['1990-05-01']})
    original = customers.copy()

    add_customer_features(customers)

    assert_frame_equal(customers, original)


def test_add_customer_features_honours_reference_year():
    customers = pd.DataFrame({'Date Of Birth': ['1990-01-01']})

    assert add_customer_features(customers, reference_year=2030)['Age'].iloc[0] == 40


def test_add_product_features_fills_missing_colour_and_sizes():
    products = pd.DataFrame({
        'Product ID': [1, 2],
        'Color': ['Red', None],
        'Sizes': [None, 'M|L'],
    })

    result = add_product_features(products)

    assert 'Color' not in result.columns
    assert result['ProductColor'].tolist() == ['Red', 'Unknown']
    assert result['Sizes'].tolist() == ['Unknown', 'M|L']


def test_add_product_features_without_optional_columns():
    result = add_product_features(pd.DataFrame({'Product ID': [1]}))

    assert result['ProductColor'].tolist() == ['Unknown']
    assert 'Sizes' not in result.columns


def test_add_transaction_features_derives_calendar_columns():
    transactions = pd.DataFrame({'Date': ['2023-07-15', '2023-12-04', 'garbage']})

    result = add_transaction_features(transactions)

    assert result['Year'].tolist()[:2] == [2023, 2023]
    assert result['Month'].tolist()[:2] == [7, 12]
    assert result['Quarter'].tolist()[:2] == [3, 4]
    # 2023-07-15 is a Saturday, 2023-12-04 a Monday
    assert result['Weekday'].tolist()[:2] == [5, 0]
    assert result['IsWeekend'].tolist()[:2] == [1, 0]
    assert result['Season'].tolist()[:2] == ['Summer', 'Winter']
    assert pd.isna(result['Date'].iloc[2])
    assert pd.isna(result['Season'].iloc[2])


def test_add_time_step_is_monotonic_across_seasons():
    df = pd.DataFrame({
        'Year': [2022, 2022, 2022, 2022, 2023],
        'Season': ['Spring', 'Summer', 'Fall', 'Winter', 'Spring'],
    })

    result = add_time_step(df)

    assert result['SeasonNum'].tolist() == [0, 1, 2, 3, 0]
    assert result['TimeStep'].is_monotonic_increasing
    assert result['TimeStep'].iloc[-1] - result['TimeStep'].iloc[0] == 4


def test_add_time_step_uses_shared_season_order():
    assert SEASON_ORDER == {'Spring': 0, 'Summer': 1, 'Fall': 2, 'Winter': 3}


def _lag_frame():
    return pd.DataFrame({
        'Category': ['Feminine'] * 4 + ['Masculine'] * 2,
        'Season': ['Spring', 'Summer', 'Fall', 'Winter', 'Spring', 'Summer'],
        'Year': [2023, 2023, 2023, 2023, 2023, 2023],
        'total_quantity': [10.0, 20.0, 30.0, 40.0, 100.0, 200.0],
        'total_revenue': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    })


def test_create_lag_features_shifts_within_groups_only():
    result = create_lag_features(
        _lag_frame(),
        group_cols=['Category'],
        target_col='total_quantity',
        lag_cols=['total_revenue'],
        n_lags=2,
    )
    feminine = result[result['Category'] == 'Feminine'].sort_values('TimeStep')
    masculine = result[result['Category'] == 'Masculine'].sort_values('TimeStep')

    assert feminine['total_quantity_lag1'].tolist() == pytest.approx([np.nan, 10, 20, 30], nan_ok=True)
    assert feminine['total_quantity_lag2'].tolist() == pytest.approx([np.nan, np.nan, 10, 20], nan_ok=True)
    # the first row of a new group must not borrow from the previous group
    assert pd.isna(masculine['total_quantity_lag1'].iloc[0])
    assert masculine['total_quantity_lag1'].iloc[1] == 100


def test_create_lag_features_rolling_mean_uses_current_and_previous():
    result = create_lag_features(
        _lag_frame(),
        group_cols=['Category'],
        target_col='total_quantity',
        lag_cols=[],
        n_lags=1,
    )
    feminine = result[result['Category'] == 'Feminine'].sort_values('TimeStep')

    assert feminine['total_quantity_roll2'].tolist() == [10.0, 15.0, 25.0, 35.0]


def test_create_lag_features_respects_n_lags_and_window():
    result = create_lag_features(
        _lag_frame(),
        group_cols=['Category'],
        target_col='total_quantity',
        lag_cols=['total_revenue'],
        n_lags=3,
        window=3,
    )

    for col in ('total_quantity', 'total_revenue'):
        assert {f'{col}_lag1', f'{col}_lag2', f'{col}_lag3', f'{col}_roll3'} <= set(result.columns)
    assert 'total_quantity_roll2' not in result.columns


def test_create_lag_features_does_not_mutate_input():
    df = _lag_frame()
    original = df.copy()

    create_lag_features(df, ['Category'], 'total_quantity', ['total_revenue'])

    assert_frame_equal(df, original)


def test_encode_categoricals_adds_encoded_columns_and_encoders():
    df = pd.DataFrame({'Season': ['Winter', 'Spring', 'Winter'], 'Gender': ['F', 'M', 'F']})

    result, encoders = encode_categoricals(df, ['Season', 'Gender'])

    assert result['Season_enc'].tolist() == [1, 0, 1]
    assert result['Gender_enc'].tolist() == [0, 1, 0]
    assert list(encoders['Season'].classes_) == ['Spring', 'Winter']
    assert 'Season_enc' not in df.columns


def test_time_based_split_keeps_future_rows_in_test():
    df = pd.DataFrame({'TimeStep': [8093, 8090, 8092, 8091, 8094], 'value': [4, 1, 3, 2, 5]})

    train, test, split_idx = time_based_split(df, test_size=0.2)

    assert split_idx == 4
    assert train['value'].tolist() == [1, 2, 3, 4]
    assert test['value'].tolist() == [5]
    assert train['TimeStep'].max() < test['TimeStep'].min()
    assert list(train.index) == [0, 1, 2, 3]


@pytest.mark.parametrize('test_size', [0, 1, -0.1, 1.5])
def test_time_based_split_rejects_invalid_test_size(test_size):
    with pytest.raises(ValueError):
        time_based_split(pd.DataFrame({'TimeStep': [1, 2]}), test_size=test_size)


def test_scale_numeric_features_fits_on_train_only():
    train = pd.DataFrame({'a': [0.0, 2.0], 'keep': [1, 2]})
    test = pd.DataFrame({'a': [4.0], 'keep': [3]})

    train_scaled, test_scaled, scaler = scale_numeric_features(train, test, ['a'])

    assert train_scaled['a'].tolist() == [-1.0, 1.0]
    # mean 1, std 1 from train -> test value 4 maps to 3.0
    assert test_scaled['a'].tolist() == [3.0]
    assert train_scaled['keep'].tolist() == [1, 2]
    assert scaler.mean_.tolist() == [1.0]
    assert train['a'].tolist() == [0.0, 2.0]


def test_evaluate_predictions_perfect_and_imperfect():
    perfect = evaluate_predictions([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert perfect == pytest.approx({'RMSE': 0.0, 'MAE': 0.0, 'R2': 1.0})

    metrics = evaluate_predictions([1.0, 2.0, 3.0, 4.0], [2.0, 2.0, 4.0, 4.0])
    assert metrics['RMSE'] == pytest.approx(np.sqrt(0.5))
    assert metrics['MAE'] == pytest.approx(0.5)
    assert metrics['R2'] == pytest.approx(0.6)
    assert all(isinstance(v, float) for v in metrics.values())


def test_rank_models_and_best_model_name():
    results = {
        'XGBoost': {'R2': 0.42, 'RMSE': 1414.0},
        'Random Forest': {'R2': 0.28, 'RMSE': 1571.0},
        'LightGBM': {'R2': -0.17, 'RMSE': 1999.0},
    }

    assert [name for name, _ in rank_models(results)] == ['XGBoost', 'Random Forest', 'LightGBM']
    assert best_model_name(results) == 'XGBoost'
    assert best_model_name(results, metric='RMSE', higher_is_better=False) == 'XGBoost'
    assert [name for name, _ in rank_models(results, metric='RMSE')] == ['LightGBM', 'Random Forest', 'XGBoost']


def test_rank_models_handles_empty_results():
    assert rank_models({}) == []
    assert best_model_name({}) is None


def test_average_feature_importance_averages_across_models():
    importances = {
        'XGBoost': np.array([0.6, 0.4]),
        'Random Forest': np.array([0.2, 0.8]),
    }

    avg = average_feature_importance(importances, ['lag1', 'lag2'])

    assert avg.index.tolist() == ['lag2', 'lag1']
    assert avg['lag1'] == pytest.approx(0.4)
    assert avg['lag2'] == pytest.approx(0.6)


def test_average_feature_importance_handles_no_models():
    assert average_feature_importance({}, ['lag1']).empty


def test_top_predictions_returns_largest_rows_in_order():
    df = pd.DataFrame({
        'Category': ['a', 'b', 'c'],
        'predicted_quantity': [5.0, 50.0, 25.0],
    })

    top = top_predictions(df, n=2)

    assert top['Category'].tolist() == ['b', 'c']
    assert len(top_predictions(df, n=10)) == 3
