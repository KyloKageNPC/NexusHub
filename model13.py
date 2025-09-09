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
from scipy.interpolate import interp1d
# from monitoring_dashboard import log_forecast
# from monitor import log_forecast

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

def handle_zero_values(ts, annual_total=None, method='smart_interpolation'):
    """
    Enhanced zero value handling with multiple strategies
    
    Parameters:
    ts: Time series with potential zero values
    annual_total: Annual total subscribers (if available)
    method: 'interpolation', 'seasonal_average', 'annual_distribution', 'smart_interpolation', 'set_null'
    
    Returns:
    Cleaned time series and information about changes made
    """
    original_ts = ts.copy()
    zero_info = {
        'method': method,
        'original_zeros': (ts == 0).sum(),
        'zero_positions': ts[ts == 0].index.tolist(),
        'total_points': len(ts),
        'changes_made': []
    }
    
    if method == 'set_null':
        # Convert zeros to NaN for exclusion from modeling
        ts_cleaned = ts.replace(0, np.nan)
        zero_info['changes_made'].append(f"Converted {zero_info['original_zeros']} zeros to NaN")
        
    elif method == 'interpolation':
        # Simple linear interpolation for zeros
        ts_cleaned = ts.replace(0, np.nan)
        ts_cleaned = ts_cleaned.interpolate(method='linear')
        
        # If still NaN at edges, use forward/backward fill
        ts_cleaned = ts_cleaned.fillna(method='ffill').fillna(method='bfill')
        
        zero_info['changes_made'].append(f"Interpolated {zero_info['original_zeros']} zero values")
        
    elif method == 'seasonal_average':
        # Replace zeros with seasonal averages
        ts_cleaned = ts.copy()
        quarters = [ts.index[i].quarter for i in range(len(ts))]
        
        for quarter in [1, 2, 3, 4]:
            quarter_mask = [q == quarter for q in quarters]
            quarter_data = ts[quarter_mask]
            non_zero_avg = quarter_data[quarter_data > 0].mean()
            
            if not np.ism(non_zero_avg):
                zero_positions_in_quarter = ts[(ts == 0) & quarter_mask].index
                ts_cleaned.loc[zero_positions_in_quarter] = non_zero_avg
                zero_info['changes_made'].append(f"Replaced {len(zero_positions_in_quarter)} zeros in Q{quarter} with seasonal average: {non_zero_avg:.2f}")
        
    elif method == 'annual_distribution' and annual_total is not None:
        # Distribute annual total across quarters based on existing patterns
        ts_cleaned = ts.copy()
        
        # Calculate typical quarterly distribution from non-zero years
        non_zero_years = {}
        for idx in ts.index:
            year = idx.year
            if year not in non_zero_years:
                non_zero_years[year] = []
            non_zero_years[year].append(ts[idx])
        
        # Find years with all non-zero quarters
        complete_years = {year: values for year, values in non_zero_years.items() 
                         if all(v > 0 for v in values)}
        
        if complete_years:
            # Calculate average quarterly distribution
            all_quarters = []
            for values in complete_years.values():
                total = sum(values)
                quarterly_props = [v/total for v in values]
                all_quarters.append(quarterly_props)
            
            avg_quarterly_dist = np.mean(all_quarters, axis=0)
            
            # Apply to years with zeros
            for year in non_zero_years:
                year_data = [ts[idx] for idx in ts.index if idx.year == year]
                if 0 in year_data:
                    # Use annual total to distribute
                    distributed_values = [annual_total * prop for prop in avg_quarterly_dist]
                    year_indices = [idx for idx in ts.index if idx.year == year]
                    for idx, value in zip(year_indices, distributed_values):
                        if ts[idx] == 0:
                            ts_cleaned[idx] = value
                    zero_info['changes_made'].append(f"Distributed annual total for year {year}")
        
    elif method == 'smart_interpolation':
        # Combination of methods based on data availability
        ts_cleaned = ts.copy()
        
        # Strategy 1: If isolated zeros, use interpolation
        zero_positions = ts == 0
        isolated_zeros = []
        
        for i, is_zero in enumerate(zero_positions):
            if is_zero:
                # Check if it's isolated (non-zero neighbors)
                has_left_neighbor = i > 0 and not zero_positions.iloc[i-1]
                has_right_neighbor = i < len(zero_positions)-1 and not zero_positions.iloc[i+1]
                
                if has_left_neighbor and has_right_neighbor:
                    isolated_zeros.append(i)
        
        # Interpolate isolated zeros
        if isolated_zeros:
            ts_temp = ts.replace(0, np.ism.nan)
            ts_temp = ts_temp.interpolate(method='linear')
            for i in isolated_zeros:
                ts_cleaned.iloc[i] = ts_temp.iloc[i]
            zero_info['changes_made'].append(f"Interpolated {len(isolated_zeros)} isolated zeros")
        
        # Strategy 2: For consecutive zeros, use seasonal patterns
        remaining_zeros = ts_cleaned == 0
        if remaining_zeros.any():
            quarters = [ts_cleaned.index[i].quarter for i in range(len(ts_cleaned))]
            
            for quarter in [1, 2, 3, 4]:
                quarter_mask = [q == quarter for q in quarters]
                quarter_data = ts_cleaned[quarter_mask]
                non_zero_avg = quarter_data[quarter_data > 0].mean()
                
                if not np.isnan(non_zero_avg):
                    zero_positions_in_quarter = ts_cleaned[(ts_cleaned == 0) & quarter_mask].index
                    ts_cleaned.loc[zero_positions_in_quarter] = non_zero_avg * 0.8  # Conservative estimate
                    zero_info['changes_made'].append(f"Replaced {len(zero_positions_in_quarter)} zeros in Q{quarter} with 80% of seasonal average: {non_zero_avg * 0.8:.2f}")
        
        # Strategy 3: If still zeros remain, use overall trend
        if (ts_cleaned == 0).any():
            overall_avg = ts_cleaned[ts_cleaned > 0].mean()
            remaining_zeros = ts_cleaned == 0
            ts_cleaned[remaining_zeros] = overall_avg * 0.5  # Very conservative
            zero_info['changes_made'].append(f"Replaced remaining {remaining_zeros.sum()} zeros with 50% of overall average: {overall_avg * 0.5:.2f}")
    
    else:
        ts_cleaned = ts.copy()
        zero_info['changes_made'].append("No changes made - unknown method or missing annual data")
    
    zero_info['final_zeros'] = (ts_cleaned == 0).sum()
    zero_info['final_nans'] = ts_cleaned.isna().sum()
    
    return ts_cleaned, zero_info

def prepare_subscriber_data(country, operator, zero_handling_method='smart_interpolation'):
    """
    Prepare subscriber count data for forecasting
    """
    # Filter for specific country and operator
    operator_data = df[(df['Country'] == country) & (df['Operator name'] == operator)]
    
    if operator_data.empty:
        raise ValueError(f"No data found for {operator} in {country}")
    
    # Look for subscriber-related columns
    subscriber_cols = []
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['subscriber', 'user', 'customer', 'connection']):
            # Check if it's a quarterly column
            if any(q in col for q in ['1Q', '2Q', '3Q', '4Q']):
                subscriber_cols.append(col)
    
    if not subscriber_cols:
        # If no explicit subscriber columns, use quarterly connection data as proxy
        subscriber_cols = [col for col in df.columns if any(q in col for q in ['1Q', '2Q', '3Q', '4Q'])]
        print(f"No explicit subscriber columns found. Using quarterly data as proxy: {len(subscriber_cols)} columns")
    
    # Get annual total if available
    annual_total = None
    annual_cols = [col for col in df.columns if 'annual' in col.lower() and 'total' in col.lower()]
    if annual_cols:
        annual_total = operator_data[annual_cols[0]].iloc[0]
        try:
            annual_total = float(annual_total)
        except:
            annual_total = None
    
    # Get the first row (should be only one row per operator-country)
    q_data = operator_data[subscriber_cols].iloc[0]
    
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
            
            # Convert to float and handle various representations of zero/missing
            try:
                val_float = float(val)
                # Sometimes very small values are effectively zero
                if val_float < 1:
                    val_float = 0
                values.append(val_float)
            except:
                values.append(0)  # Treat unparseable values as zero
                
        except Exception as e:
            print(f"Skipping invalid column {col}: {str(e)}")
    
    # Create time series
    ts = pd.Series(values, index=pd.PeriodIndex(quarters, freq='Q'))
    ts = ts.sort_index()
    
    # Handle zero values using specified method
    ts_cleaned, zero_info = handle_zero_values(ts, annual_total, zero_handling_method)
    
    # Final cleanup - handle any remaining NaN values
    if ts_cleaned.isna().any():
        ts_cleaned = ts_cleaned.fillna(method='ffill').fillna(method='bfill')
        # If still NaN (all data was missing), fill with a minimal value
        if ts_cleaned.isna().any():
            ts_cleaned = ts_cleaned.fillna(1.0)
    
    return ts_cleaned, zero_info

def apply_variance_reduction_techniques(ts, method='log_transform'):
    """
    Apply various variance reduction techniques to the time series
    Enhanced to handle small values better with robust error handling
    """
    original_ts = ts.copy()
    transform_info = {'method': method, 'original_std': ts.std(), 'original_mean': ts.mean()}
    
    # Add small constant to avoid log(0) issues and handle edge cases
    min_val = ts.min()
    if min_val <= 0:
        shift_constant = abs(min_val) + 1
        ts = ts + shift_constant
        transform_info['shift_constant'] = shift_constant
    else:
        transform_info['shift_constant'] = 0
    
    try:
        if method == 'log_transform':
            ts_transformed = np.log(ts)
            transform_info['inverse_func'] = lambda x: np.maximum(np.exp(x) - transform_info['shift_constant'], 0.1)
            
        elif method == 'box_cox':
            from scipy.stats import boxcox
            # Ensure all values are positive for Box-Cox
            ts_positive = np.maximum(ts, 0.1)
            ts_transformed, lambda_param = boxcox(ts_positive)
            ts_transformed = pd.Series(ts_transformed, index=ts.index)
            transform_info['lambda'] = lambda_param
            
            def safe_inverse_boxcox(x):
                try:
                    if lambda_param != 0:
                        result = np.power(x * lambda_param + 1, 1/lambda_param)
                    else:
                        result = np.exp(x)
                    return np.maximum(result - transform_info['shift_constant'], 0.1)
                except:
                    return np.full_like(x, ts.mean())
            
            transform_info['inverse_func'] = safe_inverse_boxcox
            
        elif method == 'diff':
            ts_transformed = ts.diff().dropna()
            first_val = ts.iloc[0]
            
            def safe_inverse_diff(x):
                try:
                    result = x.cumsum() + first_val - transform_info['shift_constant']
                    return np.maximum(result, 0.1)
                except:
                    return np.full_like(x, ts.mean())
            
            transform_info['inverse_func'] = safe_inverse_diff
            
        elif method == 'smooth':
            window = min(4, max(2, len(ts) // 4))
            ts_transformed = ts.rolling(window=window, center=True).mean().fillna(method='bfill').fillna(method='ffill')
            transform_info['window'] = window
            transform_info['inverse_func'] = lambda x: np.maximum(x - transform_info['shift_constant'], 0.1)
            
        elif method == 'outlier_removal':
            Q1 = ts.quantile(0.25)
            Q3 = ts.quantile(0.75)
            IQR = Q3 - Q1
            
            # Handle case where IQR is 0 (all values are the same)
            if IQR == 0:
                lower_bound = Q1 - ts.std()
                upper_bound = Q3 + ts.std()
            else:
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
            
            ts_transformed = ts.clip(lower=lower_bound, upper=upper_bound)
            transform_info['bounds'] = (lower_bound, upper_bound)
            transform_info['outliers_capped'] = sum((ts < lower_bound) | (ts > upper_bound))
            transform_info['inverse_func'] = lambda x: np.maximum(x - transform_info['shift_constant'], 0.1)
            
        elif method == 'hybrid':
            # Enhanced hybrid method with better error handling
            Q1 = ts.quantile(0.25)
            Q3 = ts.quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR == 0:
                lower_bound = Q1 - ts.std()
                upper_bound = Q3 + ts.std()
            else:
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
            
            ts_step1 = ts.clip(lower=lower_bound, upper=upper_bound)
            
            # Choose transformation based on data characteristics
            median_val = ts_step1.median()
            mean_val = ts_step1.mean()
            
            if median_val < 10:  # Very small values
                ts_transformed = np.sqrt(ts_step1)
                transform_info['sub_method'] = 'sqrt'
                transform_info['inverse_func'] = lambda x: np.maximum(np.power(np.maximum(x, 0), 2) - transform_info['shift_constant'], 0.1)
            elif median_val < 100:  # Medium values
                ts_transformed = np.log(ts_step1)
                transform_info['sub_method'] = 'log'
                transform_info['inverse_func'] = lambda x: np.maximum(np.exp(x) - transform_info['shift_constant'], 0.1)
            else:  # Large values
                ts_transformed = np.log(ts_step1)
                transform_info['sub_method'] = 'log'
                transform_info['inverse_func'] = lambda x: np.maximum(np.exp(x) - transform_info['shift_constant'], 0.1)
            
            # Light smoothing only if we have enough data points
            if len(ts_transformed) >= 5:
                window = min(3, len(ts_transformed) // 3)
                ts_transformed = ts_transformed.rolling(window=window, center=True).mean().fillna(method='bfill').fillna(method='ffill')
                transform_info['window'] = window
            
            transform_info['bounds'] = (lower_bound, upper_bound)
            transform_info['outliers_capped'] = sum((ts < lower_bound) | (ts > upper_bound))
        
        else:
            ts_transformed = ts.copy()
            transform_info['inverse_func'] = lambda x: np.maximum(x - transform_info['shift_constant'], 0.1)
    
    except Exception as e:
        print(f"Warning: Transformation method '{method}' failed ({str(e)}), using original data")
        ts_transformed = ts.copy()
        transform_info['inverse_func'] = lambda x: np.maximum(x - transform_info['shift_constant'], 0.1)
        transform_info['transformation_failed'] = True
    
    # Final safety checks
    if ts_transformed.isna().any():
        ts_transformed = ts_transformed.fillna(ts_transformed.mean())
    
    if ts_transformed.std() == 0:
        print("Warning: Transformed data has zero variance")
        ts_transformed = ts_transformed + np.random.normal(0, 0.01, len(ts_transformed))
    
    transform_info['transformed_std'] = ts_transformed.std()
    transform_info['transformed_mean'] = ts_transformed.mean()
    
    # Handle division by zero in variance reduction calculation
    if transform_info['original_std'] != 0:
        transform_info['variance_reduction'] = (1 - transform_info['transformed_std'] / transform_info['original_std']) * 100
    else:
        transform_info['variance_reduction'] = 0
    
    return ts_transformed, transform_info

def forecast_subscriber_count(country, operator, forecast_periods=8, variance_method='hybrid', zero_method='smart_interpolation'):
    """
    Forecast subscriber count over time
    """
    print(f"\n{'='*70}")
    print(f"SUBSCRIBER COUNT FORECASTING: {operator} in {country}")
    print(f"Forecast periods: {forecast_periods} quarters")
    print(f"Zero Handling Method: {zero_method}")
    print(f"Variance Reduction Method: {variance_method}")
    print(f"{'='*70}\n")
    
    # Prepare subscriber data with zero handling
    ts_cleaned, zero_info = prepare_subscriber_data(country, operator, zero_method)
    
    print("ZERO VALUE HANDLING RESULTS:")
    print(f"Original zeros: {zero_info['original_zeros']}")
    print(f"Final zeros: {zero_info['final_zeros']}")
    print(f"Final NaNs: {zero_info['final_nans']}")
    print("Changes made:")
    for change in zero_info['changes_made']:
        print(f"  - {change}")
    
    print(f"\nSUBSCRIBER DATA STATISTICS:")
    print(f"Mean subscribers: {ts_cleaned.mean():,.0f}")
    print(f"Standard deviation: {ts_cleaned.std():,.0f}")
    print(f"Min subscribers: {ts_cleaned.min():,.0f}")
    print(f"Max subscribers: {ts_cleaned.max():,.0f}")
    print(f"Data points: {len(ts_cleaned)} quarters")
    print(f"Time range: {ts_cleaned.index[0]} to {ts_cleaned.index[-1]}")
    
    # Calculate growth rate
    if len(ts_cleaned) > 1:
        growth_rate = ((ts_cleaned.iloc[-1] / ts_cleaned.iloc[0]) ** (1/len(ts_cleaned)) - 1) * 100
        print(f"Average quarterly growth rate: {growth_rate:.2f}%")
    
    # Apply variance reduction
    ts_transformed, transform_info = apply_variance_reduction_techniques(ts_cleaned, variance_method)
    
    print(f"\nTRANSFORMED DATA STATISTICS:")
    print(f"Mean: {transform_info['transformed_mean']:.4f}")
    print(f"Standard deviation: {transform_info['transformed_std']:.4f}")
    print(f"Variance reduction: {transform_info['variance_reduction']:.2f}%")
    
    # Enhanced plotting
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    
    # Original subscriber count over time
    ts_cleaned.plot(ax=axes[0,0], title='Subscriber Count Over Time', marker='o', linewidth=2, markersize=6)
    axes[0,0].set_ylabel('Subscribers')
    axes[0,0].grid(True, alpha=0.7)
    axes[0,0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    
    # Transformed time series
    ts_transformed.plot(ax=axes[0,1], title=f'Transformed Time Series ({variance_method})', marker='s', linewidth=2, markersize=6)
    axes[0,1].set_ylabel('Transformed Values')
    axes[0,1].grid(True, alpha=0.7)
    
    # Quarterly growth rate
    if len(ts_cleaned) > 1:
        growth_rates = ts_cleaned.pct_change().dropna() * 100
        growth_rates.plot(ax=axes[1,0], title='Quarterly Growth Rate', marker='^', linewidth=2, markersize=6)
        axes[1,0].set_ylabel('Growth Rate (%)')
        axes[1,0].grid(True, alpha=0.7)
        axes[1,0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # Distribution of subscriber counts
    axes[1,1].hist(ts_cleaned, bins=10, alpha=0.7, edgecolor='black')
    axes[1,1].set_title('Distribution of Subscriber Counts')
    axes[1,1].set_xlabel('Subscribers')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].grid(True, alpha=0.7)
    
    plt.suptitle(f'Subscriber Data Analysis: {operator} in {country}', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Continue with modeling...
    print(f"\nFORECASTING SUBSCRIBER COUNT:")
    
    # Auto ARIMA
    print("Searching for best SARIMA parameters...")
    auto_model = auto_arima(
        ts_transformed,
        start_p=0, d=1, start_q=0,
        start_P=0, D=1, start_Q=0,
        max_p=3, max_q=3,
        max_P=2, max_Q=2,
        m=4,
        seasonal=True,
        stepwise=True,
        suppress_warnings=True,
        error_action='ignore',
        trace=False
    )
    
    print(f"Best model: SARIMA{auto_model.order}{auto_model.seasonal_order}")
    
    # Fit model on all data and forecast
    model = SARIMAX(ts_transformed, order=auto_model.order, seasonal_order=auto_model.seasonal_order)
    model_fit = model.fit(disp=False)
    
    # Generate forecast
    forecast_transformed = model_fit.get_forecast(steps=forecast_periods)
    forecast_mean_transformed = forecast_transformed.predicted_mean
    conf_int_transformed = forecast_transformed.conf_int()
    
    # Create future periods
    last_period = ts_cleaned.index[-1]
    future_periods = pd.period_range(start=last_period + 1, periods=forecast_periods, freq='Q')
    
    # Transform back to original scale
    inverse_func = transform_info['inverse_func']
    try:
        forecast_mean_original = inverse_func(forecast_mean_transformed)
        forecast_lower = inverse_func(conf_int_transformed.iloc[:, 0])
        forecast_upper = inverse_func(conf_int_transformed.iloc[:, 1])
        
        # Ensure forecasts are reasonable
        forecast_mean_original = np.maximum(forecast_mean_original, 1)
        forecast_lower = np.maximum(forecast_lower, 1)
        forecast_upper = np.maximum(forecast_upper, 1)
        
        # Ensure lower bound is less than upper bound
        forecast_lower = np.minimum(forecast_lower, forecast_mean_original)
        forecast_upper = np.maximum(forecast_upper, forecast_mean_original)
        
    except Exception as e:
        print(f"Warning: Inverse transformation failed ({str(e)}), using trend-based forecast")
        # Fallback: use trend from last few periods
        recent_trend = ts_cleaned.iloc[-4:].pct_change().mean()
        if np.isnan(recent_trend):
            recent_trend = 0.05  # Default 5% growth
        
        forecast_mean_original = []
        last_value = ts_cleaned.iloc[-1]
        for i in range(forecast_periods):
            next_value = last_value * (1 + recent_trend)
            forecast_mean_original.append(next_value)
            last_value = next_value
        
        forecast_mean_original = np.array(forecast_mean_original)
        forecast_lower = forecast_mean_original * 0.8
        forecast_upper = forecast_mean_original * 1.2
    
    # Create forecast results
    forecast_results = pd.DataFrame({
        'Forecast': forecast_mean_original,
        'Lower_Bound': forecast_lower,
        'Upper_Bound': forecast_upper,
    }, index=future_periods)
    
    print("\nFORECAST RESULTS:")
    print(forecast_results.round(0))
    
    # Calculate projected annual growth
    if len(forecast_results) >= 4:
        annual_growth = ((forecast_results.iloc[-1]['Forecast'] / ts_cleaned.iloc[-1]) ** (1/len(forecast_results)) - 1) * 100
        print(f"\nProjected annual growth rate: {annual_growth:.2f}%")
    
    # Create comprehensive forecast visualization
    plt.figure(figsize=(20, 10))
    
    # Historical data
    ts_idx = ts_cleaned.index.to_timestamp()
    forecast_idx = forecast_results.index.to_timestamp()
    
    plt.plot(ts_idx, ts_cleaned.values, 'o-', label='Historical Subscriber Count', 
             linewidth=3, markersize=8, color='blue')
    
    # Forecast
    plt.plot(forecast_idx, forecast_results['Forecast'], 's--', 
             label=f'Forecast ({forecast_periods} quarters)', linewidth=3, markersize=8, color='red')
    
    # Confidence intervals
    plt.fill_between(forecast_idx, forecast_results['Lower_Bound'], forecast_results['Upper_Bound'],
                     alpha=0.3, color='red', label='95% Confidence Interval')
    
    # Highlight zero positions if any
    if zero_info['zero_positions']:
        zero_idx = [idx.to_timestamp() for idx in zero_info['zero_positions']]
        zero_vals = [ts_cleaned[idx] for idx in zero_info['zero_positions']]
        plt.scatter(zero_idx, zero_vals, color='yellow', s=150, 
                   label='Originally Zero Values', zorder=5, edgecolors='black')
    
    # Add summary statistics
    stats_text = f"""
    Current Subscribers: {ts_cleaned.iloc[-1]:,.0f}
    Forecast End: {forecast_results.iloc[-1]['Forecast']:,.0f}
    Forecast Growth: {((forecast_results.iloc[-1]['Forecast'] / ts_cleaned.iloc[-1]) - 1) * 100:.1f}%
    Zero Values Handled: {zero_info['original_zeros']}
    Method: {zero_method}
    """
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.title(f'Subscriber Count Forecast: {operator} in {country}', fontsize=16, fontweight='bold')
    plt.xlabel('Quarter', fontsize=12)
    plt.ylabel('Subscribers', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.7)
    
    # Format y-axis to show numbers in readable format
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    
    plt.tight_layout()
    plt.show()
    
    # Model diagnostics
    print("\nMODEL DIAGNOSTICS:")
    print(f"AIC: {model_fit.aic:.2f}")
    print(f"BIC: {model_fit.bic:.2f}")
    print(f"Log-likelihood: {model_fit.llf:.2f}")

def forecast_multiple_operators(country, operators, forecast_periods=8, variance_method='hybrid', zero_method='smart_interpolation'):
    """
    Forecast and plot subscriber counts for multiple operators in one country.
    """
    plt.figure(figsize=(20, 10))
    colors = plt.cm.get_cmap('tab10', len(operators))
    for i, operator in enumerate(operators):
        try:
            ts_cleaned, zero_info = prepare_subscriber_data(country, operator, zero_method)
            ts_transformed, transform_info = apply_variance_reduction_techniques(ts_cleaned, variance_method)
            auto_model = auto_arima(
                ts_transformed,
                start_p=0, d=1, start_q=0,
                start_P=0, D=1, start_Q=0,
                max_p=3, max_q=3,
                max_P=2, max_Q=2,
                m=4,
                seasonal=True,
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore',
                trace=False
            )
            model = SARIMAX(ts_transformed, order=auto_model.order, seasonal_order=auto_model.seasonal_order)
            model_fit = model.fit(disp=False)
            forecast_transformed = model_fit.get_forecast(steps=forecast_periods)
            forecast_mean_transformed = forecast_transformed.predicted_mean
            conf_int_transformed = forecast_transformed.conf_int()
            last_period = ts_cleaned.index[-1]
            future_periods = pd.period_range(start=last_period + 1, periods=forecast_periods, freq='Q')
            inverse_func = transform_info['inverse_func']
            forecast_mean_original = inverse_func(forecast_mean_transformed)
            forecast_mean_original = np.maximum(forecast_mean_original, 1)
            ts_idx = ts_cleaned.index.to_timestamp()
            forecast_idx = future_periods.to_timestamp()
            # Plot historical
            plt.plot(ts_idx, ts_cleaned.values, 'o-', label=f'{operator} (history)', color=colors(i))
            # Plot forecast
            plt.plot(forecast_idx, forecast_mean_original, '--', label=f'{operator} (forecast)', color=colors(i))
        except Exception as e:
            print(f"Error processing {operator}: {e}")
    plt.title(f'Subscriber Count Forecasts: {", ".join(operators)} in {country}', fontsize=16, fontweight='bold')
    plt.xlabel('Quarter', fontsize=12)
    plt.ylabel('Subscribers', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.7)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example: Compare MTN and Expresso in Ghana
    forecast_multiple_operators(
        country="Ghana",
        operators=["MTN", "Tigo"],
        forecast_periods=8,
        variance_method='hybrid',
        zero_method='smart_interpolation'
    )