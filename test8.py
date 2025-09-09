import pytest
import pandas as pd
import numpy as np

from model8 import (
    handle_zero_values,
    prepare_quarterly_data_enhanced,
    apply_variance_reduction_techniques,
    analyze_and_forecast_enhanced
)

@pytest.fixture
def sample_ts():
    idx = pd.period_range('2020Q1', periods=8, freq='Q')
    vals = [0, 10, 0, 30, 40, 0, 60, 0]
    return pd.Series(vals, index=idx)

@pytest.fixture
def sample_operator_df(monkeypatch):
    # Patch the global df in model8
    df = pd.DataFrame({
        'Country': ['Ghana'],
        'Operator name': ['Glo'],
        '1Q 2020': [0], '2Q 2020': [10], '3Q 2020': [0], '4Q 2020': [30],
        '1Q 2021': [40], '2Q 2021': [0], '3Q 2021': [60], '4Q 2021': [0],
        'Annual total': [140]
    })
    monkeypatch.setattr('model8.df', df)

def test_handle_zero_values_methods(sample_ts):
    methods = ['interpolation', 'seasonal_average', 'annual_distribution', 'smart_interpolation', 'set_null']
    for method in methods:
        cleaned, info = handle_zero_values(sample_ts, annual_total=140, method=method)
        assert isinstance(cleaned, pd.Series)
        # set_null method can have NaNs, others should not
        if method != 'set_null':
            assert not cleaned.isna().any()
        # No zeros should remain except possibly for set_null
        if method != 'set_null':
            assert (cleaned == 0).sum() == 0

def test_prepare_quarterly_data_enhanced(sample_operator_df):
    ts, info = prepare_quarterly_data_enhanced('Ghana', 'Glo', zero_handling_method='smart_interpolation')
    assert isinstance(ts, pd.Series)
    assert not ts.isna().any()
    assert (ts == 0).sum() == 0
    assert info['original_zeros'] > 0

def test_apply_variance_reduction_techniques(sample_ts):
    for method in ['log_transform', 'box_cox', 'diff', 'smooth', 'outlier_removal', 'hybrid']:
        transformed, info = apply_variance_reduction_techniques(sample_ts.replace(0, 1), method=method)
        assert isinstance(transformed, pd.Series)
        assert not transformed.isna().any()
        assert 'variance_reduction' in info

def test_analyze_and_forecast_enhanced_runs(sample_operator_df):
    results, zero_info, transform_info = analyze_and_forecast_enhanced('Ghana', 'Glo', 'hybrid', 'smart_interpolation')
    # Results may be None if not enough data, but zero_info and transform_info should always be dicts
    assert isinstance(zero_info, dict)
    assert isinstance(transform_info, dict)
    if results is not None:
        assert isinstance(results, pd.DataFrame)
        assert 'Actual' in results.columns
        assert 'Predicted' in results.columns
        assert not results.isna().any().any()