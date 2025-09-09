import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from pmdarima import auto_arima
import warnings
from sklearn.model_selection import TimeSeriesSplit
from scipy import stats
from monitoring_dashboard import log_forecast
from monitor import log_forecast

def cross_validate_model(ts, order, seasonal_order, n_splits=3):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    metrics = []
    
    for train_index, test_index in tscv.split(ts):
        train, test = ts.iloc[train_index], ts.iloc[test_index]
        model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
        model_fit = model.fit(disp=False)
        forecast = model_fit.forecast(steps=len(test))
        rmse = np.sqrt(mean_squared_error(test, forecast))
        metrics.append(rmse)
    
    return np.mean(metrics)

warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv('cleaned.csv')

def prepare_quarterly_data(country, operator):
    """
    Prepare quarterly time series data for a specific country and operator
    """
    # Filter for specific country and operator
    operator_data = df[(df['Country'] == country) & (df['Operator name'] == operator)]
    
    if operator_data.empty:
        raise ValueError(f"No data found for {operator} in {country}")
    
    # Extract quarterly columns
    q_cols = [col for col in df.columns if any(q in col for q in ['1Q', '2Q', '3Q', '4Q'])]
    
    # Get the first row (should be only one row per operator-country)
    q_data = operator_data[q_cols].iloc[0]
    
    # Parse quarterly data
    quarters = []
    values = []
    
    for col, val in q_data.items():
        try:
            # Handle different column formats: "1Q 2015" or "1Q2015"
            if ' ' in col:
                q, year = col.split()
            else:
                q = col[:2]  # "1Q"
                year = col[2:]
            
            quarter = int(q[0])
            year = int(year)
            quarters.append(f'{year}-Q{quarter}')
            values.append(float(val))
        except Exception as e:
            print(f"Skipping invalid column {col}: {str(e)}")
    
    # Create time series
    ts = pd.Series(values, index=pd.PeriodIndex(quarters, freq='Q'))
    ts = ts.sort_index()
    
    # Handle missing values - forward fill then backward fill
    ts = ts.ffill().bfill()
    
    return ts

def apply_variance_reduction_techniques(ts, method='log_transform'):
    """
    Apply various variance reduction techniques to the time series
    
    Parameters:
    ts: Time series data
    method: 'log_transform', 'box_cox', 'diff', 'smooth', 'outlier_removal', 'hybrid'
    
    Returns:
    Transformed time series and transformation info
    """
    original_ts = ts.copy()
    transform_info = {'method': method, 'original_std': ts.std(), 'original_mean': ts.mean()}
    
    if method == 'log_transform':
        # Log transformation - good for exponential growth patterns
        ts_transformed = np.log(ts + 1)  # +1 to handle any zeros
        transform_info['inverse_func'] = lambda x: np.exp(x) - 1
        
    elif method == 'box_cox':
        # Box-Cox transformation - automatically finds optimal lambda
        from scipy.stats import boxcox
        ts_transformed, lambda_param = boxcox(ts + 1)  # +1 to ensure positive values
        ts_transformed = pd.Series(ts_transformed, index=ts.index)
        transform_info['lambda'] = lambda_param
        transform_info['inverse_func'] = lambda x: np.power(x * lambda_param + 1, 1/lambda_param) - 1 if lambda_param != 0 else np.exp(x) - 1
        
    elif method == 'diff':
        # First differencing to remove trend
        ts_transformed = ts.diff().dropna()
        transform_info['inverse_func'] = lambda x: x.cumsum() + ts.iloc[0]
        
    elif method == 'smooth':
        # Moving average smoothing
        window = min(4, len(ts) // 4)  # Use quarterly smoothing or adjust based on data length
        ts_transformed = ts.rolling(window=window, center=True).mean().fillna(method='bfill').fillna(method='ffill')
        transform_info['window'] = window
        transform_info['inverse_func'] = lambda x: x  # No inverse needed for smoothing
        
    elif method == 'outlier_removal':
        # Remove outliers using IQR method
        Q1 = ts.quantile(0.25)
        Q3 = ts.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Cap outliers instead of removing them to preserve time series continuity
        ts_transformed = ts.clip(lower=lower_bound, upper=upper_bound)
        transform_info['bounds'] = (lower_bound, upper_bound)
        transform_info['outliers_capped'] = sum((ts < lower_bound) | (ts > upper_bound))
        transform_info['inverse_func'] = lambda x: x  # No inverse transformation needed
        
    elif method == 'hybrid':
        # Combination of techniques
        # Step 1: Remove outliers
        Q1 = ts.quantile(0.25)
        Q3 = ts.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        ts_step1 = ts.clip(lower=lower_bound, upper=upper_bound)
        
        # Step 2: Log transform
        ts_transformed = np.log(ts_step1 + 1)
        
        # Step 3: Light smoothing
        window = 3
        ts_transformed = ts_transformed.rolling(window=window, center=True).mean().fillna(method='bfill').fillna(method='ffill')
        
        transform_info['bounds'] = (lower_bound, upper_bound)
        transform_info['outliers_capped'] = sum((ts < lower_bound) | (ts > upper_bound))
        transform_info['window'] = window
        transform_info['inverse_func'] = lambda x: np.exp(x) - 1
    
    else:
        ts_transformed = ts.copy()
        transform_info['inverse_func'] = lambda x: x
    
    transform_info['transformed_std'] = ts_transformed.std()
    transform_info['transformed_mean'] = ts_transformed.mean()
    transform_info['variance_reduction'] = (1 - transform_info['transformed_std'] / transform_info['original_std']) * 100
    
    return ts_transformed, transform_info

def analyze_and_forecast_enhanced(country, operator, variance_method='hybrid'):
    """
    Enhanced analysis with variance reduction techniques
    """
    print(f"\n{'='*60}")
    print(f"ENHANCED ANALYSIS: {operator} in {country}")
    print(f"Variance Reduction Method: {variance_method}")
    print(f"{'='*60}\n")
    
    # Prepare quarterly data
    ts_original = prepare_quarterly_data(country, operator)
    
    print("ORIGINAL DATA STATISTICS:")
    print(f"Mean connections: {ts_original.mean():,.2f}")
    print(f"Standard deviation: {ts_original.std():,.2f}")
    print(f"Coefficient of Variation: {(ts_original.std()/ts_original.mean())*100:.2f}%")
    
    # Apply variance reduction
    ts_transformed, transform_info = apply_variance_reduction_techniques(ts_original, variance_method)
    
    print(f"\nTRANSFORMED DATA STATISTICS:")
    print(f"Mean: {transform_info['transformed_mean']:.4f}")
    print(f"Standard deviation: {transform_info['transformed_std']:.4f}")
    print(f"Variance reduction: {transform_info['variance_reduction']:.2f}%")
    if 'outliers_capped' in transform_info:
        print(f"Outliers capped: {transform_info['outliers_capped']}")
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Original time series
    ts_original.plot(ax=axes[0,0], title='Original Time Series')
    axes[0,0].set_ylabel('Connections')
    axes[0,0].grid(True)
    
    # Transformed time series
    ts_transformed.plot(ax=axes[0,1], title=f'Transformed Time Series ({variance_method})')
    axes[0,1].set_ylabel('Transformed Values')
    axes[0,1].grid(True)
    
    # Distribution comparison
    axes[1,0].hist(ts_original, bins=15, alpha=0.7, label='Original')
    axes[1,0].set_title('Original Distribution')
    axes[1,0].set_xlabel('Connections')
    axes[1,0].set_ylabel('Frequency')
    
    axes[1,1].hist(ts_transformed, bins=15, alpha=0.7, label='Transformed', color='orange')
    axes[1,1].set_title('Transformed Distribution')
    axes[1,1].set_xlabel('Transformed Values')
    axes[1,1].set_ylabel('Frequency')
    
    plt.suptitle(f'Variance Reduction Analysis: {operator} in {country}', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Model fitting on transformed data
    print(f"\nMODELING ON TRANSFORMED DATA:")
    print(f"Data range: {ts_transformed.index[0]} to {ts_transformed.index[-1]}")
    print(f"Number of quarters: {len(ts_transformed)}")
    
    # Auto ARIMA on transformed data
    print("\nSearching for best SARIMA parameters on transformed data...")
    auto_model = auto_arima(
        ts_transformed,
        start_p=0, d=1, start_q=0,
        start_P=0, D=1, start_Q=0,
        max_p=3, max_q=3,
        max_P=2, max_Q=2,
        m=4,  # Quarterly seasonality
        seasonal=True,
        stepwise=True,
        suppress_warnings=True,
        error_action='ignore',
        trace=False
    )
    
    print(f"Best model for transformed data: SARIMA{auto_model.order}{auto_model.seasonal_order}")
    
    # Split data
    train_transformed = ts_transformed.iloc[:-4]
    test_transformed = ts_transformed.iloc[-4:]
    test_original = ts_original.iloc[-4:]
    
    # Fit model
    order = auto_model.order
    seasonal_order = auto_model.seasonal_order
    
    model = SARIMAX(train_transformed, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)
    
    # Forecast on transformed scale
    forecast_transformed = model_fit.get_forecast(steps=4)
    forecast_mean_transformed = forecast_transformed.predicted_mean
    conf_int_transformed = forecast_transformed.conf_int()
    
    # Transform back to original scale
    inverse_func = transform_info['inverse_func']
    forecast_mean_original = inverse_func(forecast_mean_transformed)
    
    # Handle confidence intervals for non-linear transformations
    if variance_method in ['log_transform', 'box_cox', 'hybrid']:
        conf_int_lower = inverse_func(conf_int_transformed.iloc[:, 0])
        conf_int_upper = inverse_func(conf_int_transformed.iloc[:, 1])
    else:
        conf_int_lower = conf_int_transformed.iloc[:, 0]
        conf_int_upper = conf_int_transformed.iloc[:, 1]
    
    # Create results DataFrame
    if isinstance(forecast_mean_original, pd.Series):
        forecast_values = forecast_mean_original.values
    else:
        forecast_values = forecast_mean_original
        
    results = pd.DataFrame({
        'Actual': test_original.values,
        'Predicted': forecast_values
    }, index=test_original.index)
    
    results['Error'] = results['Actual'] - results['Predicted']
    results['AbsoluteError'] = np.abs(results['Error'])
    results['PercentageError'] = (results['Error'] / results['Actual']) * 100
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(results['Actual'], results['Predicted']))
    mae = mean_absolute_error(results['Actual'], results['Predicted'])
    mape = mean_absolute_percentage_error(results['Actual'], results['Predicted'])
    
    # Cross-validation on transformed data
    cv_rmse_transformed = cross_validate_model(ts_transformed, order, seasonal_order)
    
    print("\nFORECAST RESULTS:")
    print(results)
    print("\nEVALUATION METRICS:")
    print(f"RMSE: {rmse:,.2f}")
    print(f"MAE: {mae:,.2f}")
    print(f"MAPE: {mape:.2%}")
    print(f"Cross-validated RMSE (transformed scale): {cv_rmse_transformed:.4f}")
    
    # Enhanced plotting
    plt.figure(figsize=(16, 10))
    
    # Convert to timestamps for plotting
    ts_idx = ts_original.index.to_timestamp()
    results_idx = results.index.to_timestamp()
    
    # Plot historical data
    plt.plot(ts_idx, ts_original.values, 'o-', label='Historical Data', alpha=0.7, linewidth=2)
    
    # Plot forecast
    plt.plot(results_idx, results['Predicted'], 's--', color='red', 
             label='Forecast', markersize=8, linewidth=2)
    
    # Confidence interval
    plt.fill_between(results_idx, conf_int_lower, conf_int_upper, 
                    color='pink', alpha=0.3, label='95% Confidence Interval')
    
    # Annotate metrics
    plt.text(0.02, 0.98, f'RMSE: {rmse:,.0f}\nMAE: {mae:,.0f}\nMAPE: {mape:.1%}\nVariance Reduction: {transform_info["variance_reduction"]:.1f}%', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.title(f'Enhanced Forecast with {variance_method.title()} Transformation: {operator} in {country}', fontsize=16)
    plt.xlabel('Quarter')
    plt.ylabel('Connections')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    # Prepare enhanced logging data
    forecast_data = []
    for idx, row in results.iterrows():
        forecast_data.append({
            'date': idx.to_timestamp().isoformat(),
            'actual': row['Actual'],
            'forecast': row['Predicted'],
            'error': row['Error']
        })

    metrics = {
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'cv_rmse_transformed': cv_rmse_transformed,
        'variance_reduction_pct': transform_info['variance_reduction'],
        'transformation_method': variance_method
    }

    # Log to monitoring database
    log_forecast(country, operator, metrics, forecast_data)
    
    return results, transform_info

def compare_variance_methods(country, operator):
    """
    Compare different variance reduction methods
    """
    methods = ['log_transform', 'box_cox', 'outlier_removal', 'smooth', 'hybrid']
    results_summary = []
    
    print(f"\n{'='*70}")
    print(f"COMPARING VARIANCE REDUCTION METHODS: {operator} in {country}")
    print(f"{'='*70}\n")
    
    ts_original = prepare_quarterly_data(country, operator)
    original_std = ts_original.std()
    
    for method in methods:
        try:
            print(f"\nTesting method: {method}")
            ts_transformed, transform_info = apply_variance_reduction_techniques(ts_original, method)
            
            results_summary.append({
                'Method': method,
                'Original_Std': original_std,
                'Transformed_Std': transform_info['transformed_std'],
                'Variance_Reduction_%': transform_info['variance_reduction'],
                'Data_Points': len(ts_transformed)
            })
            
        except Exception as e:
            print(f"Method {method} failed: {str(e)}")
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results_summary)
    print("\nVARIANCE REDUCTION COMPARISON:")
    print(comparison_df.to_string(index=False))
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    methods_list = comparison_df['Method'].tolist()
    reductions = comparison_df['Variance_Reduction_%'].tolist()
    
    bars = plt.bar(methods_list, reductions, color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'plum'])
    plt.title(f'Variance Reduction Comparison: {operator} in {country}')
    plt.xlabel('Transformation Method')
    plt.ylabel('Variance Reduction (%)')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, reduction in zip(bars, reductions):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{reduction:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return comparison_df

# Example usage
if __name__ == "__main__":
    # Compare different methods first
    comparison = compare_variance_methods('Nigeria', 'Airtel')
    
    # Use the best method (or try different ones)
    print("\n" + "="*80)
    print("RUNNING ENHANCED ANALYSIS WITH HYBRID METHOD")
    print("="*80)

    ghana_results, transform_info = analyze_and_forecast_enhanced('Nigeria', 'Airtel', 'hybrid')

    # You can try other methods:
    # ghana_results_log = analyze_and_forecast_enhanced('Ghana', 'MTN', 'log_transform')
    # ghana_results_boxcox = analyze_and_forecast_enhanced('Ghana', 'MTN', 'box_cox')