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

warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv('cleaner_processed.csv')

def handle_zero_values(ts, annual_total=None, method='smart_interpolation'):
    """
    Enhanced zero value handling with multiple strategies
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
        ts_cleaned = ts.replace(0, np.nan)
        zero_info['changes_made'].append(f"Converted {zero_info['original_zeros']} zeros to NaN")
        
    elif method == 'interpolation':
        ts_cleaned = ts.replace(0, np.nan)
        ts_cleaned = ts_cleaned.interpolate(method='linear')
        ts_cleaned = ts_cleaned.fillna(method='ffill').fillna(method='bfill')
        zero_info['changes_made'].append(f"Interpolated {zero_info['original_zeros']} zero values")
        
    elif method == 'seasonal_average':
        ts_cleaned = ts.copy()
        quarters = [ts.index[i].quarter for i in range(len(ts))]
        
        for quarter in [1, 2, 3, 4]:
            quarter_mask = [q == quarter for q in quarters]
            quarter_data = ts[quarter_mask]
            non_zero_avg = quarter_data[quarter_data > 0].mean()
            
            if not np.isnan(non_zero_avg):
                zero_positions_in_quarter = ts[(ts == 0) & quarter_mask].index
                ts_cleaned.loc[zero_positions_in_quarter] = non_zero_avg
                zero_info['changes_made'].append(f"Replaced {len(zero_positions_in_quarter)} zeros in Q{quarter} with seasonal average: {non_zero_avg:.2f}")
        
    elif method == 'smart_interpolation':
        ts_cleaned = ts.copy()
        
        # Strategy 1: Interpolate isolated zeros
        zero_positions = ts == 0
        isolated_zeros = []
        
        for i, is_zero in enumerate(zero_positions):
            if is_zero:
                has_left_neighbor = i > 0 and not zero_positions.iloc[i-1]
                has_right_neighbor = i < len(zero_positions)-1 and not zero_positions.iloc[i+1]
                
                if has_left_neighbor and has_right_neighbor:
                    isolated_zeros.append(i)
        
        if isolated_zeros:
            ts_temp = ts.replace(0, np.nan)
            ts_temp = ts_temp.interpolate(method='linear')
            for i in isolated_zeros:
                ts_cleaned.iloc[i] = ts_temp.iloc[i]
            zero_info['changes_made'].append(f"Interpolated {len(isolated_zeros)} isolated zeros")
        
        # Strategy 2: Use seasonal patterns for remaining zeros
        remaining_zeros = ts_cleaned == 0
        if remaining_zeros.any():
            quarters = [ts_cleaned.index[i].quarter for i in range(len(ts_cleaned))]
            
            for quarter in [1, 2, 3, 4]:
                quarter_mask = [q == quarter for q in quarters]
                quarter_data = ts_cleaned[quarter_mask]
                non_zero_avg = quarter_data[quarter_data > 0].mean()
                
                if not np.isnan(non_zero_avg):
                    zero_positions_in_quarter = ts_cleaned[(ts_cleaned == 0) & quarter_mask].index
                    ts_cleaned.loc[zero_positions_in_quarter] = non_zero_avg * 0.8
                    zero_info['changes_made'].append(f"Replaced {len(zero_positions_in_quarter)} zeros in Q{quarter} with 80% of seasonal average: {non_zero_avg * 0.8:.2f}")
        
        # Strategy 3: Use overall trend for remaining zeros
        if (ts_cleaned == 0).any():
            overall_avg = ts_cleaned[ts_cleaned > 0].mean()
            remaining_zeros = ts_cleaned == 0
            ts_cleaned[remaining_zeros] = overall_avg * 0.5
            zero_info['changes_made'].append(f"Replaced remaining {remaining_zeros.sum()} zeros with 50% of overall average: {overall_avg * 0.5:.2f}")
    
    else:
        ts_cleaned = ts.copy()
        zero_info['changes_made'].append("No changes made - unknown method")
    
    zero_info['final_zeros'] = (ts_cleaned == 0).sum()
    zero_info['final_nans'] = ts_cleaned.isna().sum()
    
    return ts_cleaned, zero_info

def prepare_subscriber_data(country, operator, zero_handling_method='smart_interpolation', start_period='2014-Q1'):
    """
    Prepare subscriber count data for forecasting
    Modified to filter data from start_period onwards (e.g., '2014-Q1')
    """
    operator_data = df[(df['Country'] == country) & (df['Operator name'] == operator)]
    
    if operator_data.empty:
        raise ValueError(f"No data found for {operator} in {country}")
    
    # Look for subscriber-related columns
    subscriber_cols = []
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['subscriber', 'user', 'customer', 'connection']):
            if any(q in col for q in ['1Q', '2Q', '3Q', '4Q']):
                subscriber_cols.append(col)
    
    if not subscriber_cols:
        subscriber_cols = [col for col in df.columns if any(q in col for q in ['1Q', '2Q', '3Q', '4Q'])]
    
    # Get annual total if available
    annual_total = None
    annual_cols = [col for col in df.columns if 'annual' in col.lower() and 'total' in col.lower()]
    if annual_cols:
        annual_total = operator_data[annual_cols[0]].iloc[0]
        try:
            annual_total = float(annual_total)
        except:
            annual_total = None
    
    q_data = operator_data[subscriber_cols].iloc[0]
    
    # Parse start period
    start_year, start_quarter = start_period.split('-Q')
    start_year = int(start_year)
    start_quarter = int(start_quarter)
    
    # Parse quarterly data
    quarters = []
    values = []
    
    for col, val in q_data.items():
        try:
            if ' ' in col:
                q, year = col.split()
            else:
                q = col[:2]
                year = col[2:]
            
            quarter = int(q[0])
            year = int(year)
            
            # FILTER: Only include data from start_period onwards
            if (year > start_year) or (year == start_year and quarter >= start_quarter):
                quarters.append(f'{year}-Q{quarter}')
                
                try:
                    val_float = float(val)
                    if val_float < 1:
                        val_float = 0
                    values.append(val_float)
                except:
                    values.append(0)
                
        except Exception as e:
            continue
    
    if not quarters:
        raise ValueError(f"No data found for {operator} from {start_period} onwards")
    
    # Create time series
    ts = pd.Series(values, index=pd.PeriodIndex(quarters, freq='Q'))
    ts = ts.sort_index()
    
    # Handle zero values
    ts_cleaned, zero_info = handle_zero_values(ts, annual_total, zero_handling_method)
    
    # Final cleanup
    if ts_cleaned.isna().any():
        ts_cleaned = ts_cleaned.fillna(method='ffill').fillna(method='bfill')
        if ts_cleaned.isna().any():
            ts_cleaned = ts_cleaned.fillna(1.0)
    
    return ts_cleaned, zero_info

def apply_variance_reduction_techniques(ts, method='log_transform'):
    """
    Apply variance reduction techniques to the time series
    """
    original_ts = ts.copy()
    transform_info = {'method': method, 'original_std': ts.std(), 'original_mean': ts.mean()}
    
    # Handle negative or zero values
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
            
        elif method == 'hybrid':
            # Outlier removal
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
            
            if median_val < 100:
                ts_transformed = np.log(ts_step1)
                transform_info['sub_method'] = 'log'
                transform_info['inverse_func'] = lambda x: np.maximum(np.exp(x) - transform_info['shift_constant'], 0.1)
            else:
                ts_transformed = np.log(ts_step1)
                transform_info['sub_method'] = 'log'
                transform_info['inverse_func'] = lambda x: np.maximum(np.exp(x) - transform_info['shift_constant'], 0.1)
            
            # Light smoothing
            if len(ts_transformed) >= 5:
                window = min(3, len(ts_transformed) // 3)
                ts_transformed = ts_transformed.rolling(window=window, center=True).mean().fillna(method='bfill').fillna(method='ffill')
        
        else:
            ts_transformed = ts.copy()
            transform_info['inverse_func'] = lambda x: np.maximum(x - transform_info['shift_constant'], 0.1)
    
    except Exception as e:
        print(f"Warning: Transformation method '{method}' failed, using original data")
        ts_transformed = ts.copy()
        transform_info['inverse_func'] = lambda x: np.maximum(x - transform_info['shift_constant'], 0.1)
    
    # Safety checks
    if ts_transformed.isna().any():
        ts_transformed = ts_transformed.fillna(ts_transformed.mean())
    
    if ts_transformed.std() == 0:
        ts_transformed = ts_transformed + np.random.normal(0, 0.01, len(ts_transformed))
    
    transform_info['transformed_std'] = ts_transformed.std()
    transform_info['transformed_mean'] = ts_transformed.mean()
    
    if transform_info['original_std'] != 0:
        transform_info['variance_reduction'] = (1 - transform_info['transformed_std'] / transform_info['original_std']) * 100
    else:
        transform_info['variance_reduction'] = 0
    
    return ts_transformed, transform_info

def forecast_single_operator(country, operator, forecast_periods=20, variance_method='hybrid', zero_method='smart_interpolation', start_period='2014-Q1', verbose=False):
    """
    Forecast subscriber count for a single operator (helper function)
    Modified to include start_period parameter (e.g., '2014-Q1')
    """
    if verbose:
        print(f"\nProcessing {operator} in {country}...")
    
    try:
        # Prepare data with period filter
        ts_cleaned, zero_info = prepare_subscriber_data(country, operator, zero_method, start_period)
        
        # Apply variance reduction
        ts_transformed, transform_info = apply_variance_reduction_techniques(ts_cleaned, variance_method)
        
        # Auto ARIMA
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
        
        # Fit model
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
            
            # Ensure forecasts are reasonable and not NaN
            forecast_mean_original = pd.Series(forecast_mean_original, index=future_periods)
            forecast_lower = pd.Series(forecast_lower, index=future_periods)  
            forecast_upper = pd.Series(forecast_upper, index=future_periods)
            
            # Handle any NaN values
            if forecast_mean_original.isna().any():
                print(f"Warning: NaN values detected in forecast for {operator}, using fallback method")
                raise ValueError("NaN values in forecast")
            
            # Ensure forecasts are reasonable
            forecast_mean_original = np.maximum(forecast_mean_original, 1)
            forecast_lower = np.maximum(forecast_lower, 1)
            forecast_upper = np.maximum(forecast_upper, 1)
            
            # Ensure bounds are correct
            forecast_lower = np.minimum(forecast_lower, forecast_mean_original)
            forecast_upper = np.maximum(forecast_upper, forecast_mean_original)
            
        except Exception as e:
            if verbose:
                print(f"Warning: Inverse transformation failed for {operator}, using trend-based forecast")
            # Fallback method using trend
            recent_values = ts_cleaned.iloc[-min(4, len(ts_cleaned)):]
            if len(recent_values) > 1:
                recent_trend = recent_values.pct_change().mean()
                if np.isnan(recent_trend) or recent_trend == 0:
                    recent_trend = 0.02  # Default 2% growth
            else:
                recent_trend = 0.02
            
            forecast_mean_original = []
            last_value = ts_cleaned.iloc[-1]
            for i in range(forecast_periods):
                next_value = last_value * (1 + recent_trend)
                forecast_mean_original.append(next_value)
                last_value = next_value
            
            forecast_mean_original = pd.Series(forecast_mean_original, index=future_periods)
            forecast_lower = forecast_mean_original * 0.85
            forecast_upper = forecast_mean_original * 1.15
        
        # Create results DataFrame
        forecast_results = pd.DataFrame({
            'Forecast': forecast_mean_original,
            'Lower_Bound': forecast_lower,
            'Upper_Bound': forecast_upper,
        }, index=future_periods)
        
        # Additional validation
        if forecast_results.isna().any().any():
            if verbose:
                print(f"Warning: NaN values still present in final forecast for {operator}")
            forecast_results = forecast_results.fillna(method='ffill').fillna(method='bfill')
            if forecast_results.isna().any().any():
                # Last resort: use simple linear trend
                last_value = ts_cleaned.iloc[-1]
                forecast_results = pd.DataFrame({
                    'Forecast': [last_value * (1.02 ** i) for i in range(1, forecast_periods + 1)],
                    'Lower_Bound': [last_value * (1.02 ** i) * 0.85 for i in range(1, forecast_periods + 1)],
                    'Upper_Bound': [last_value * (1.02 ** i) * 1.15 for i in range(1, forecast_periods + 1)]
                }, index=future_periods)
        
        return {
            'operator': operator,
            'historical_data': ts_cleaned,
            'forecast_results': forecast_results,
            'zero_info': zero_info,
            'model_info': {
                'order': auto_model.order,
                'seasonal_order': auto_model.seasonal_order,
                'aic': model_fit.aic,
                'bic': model_fit.bic
            }
        }
        
    except Exception as e:
        if verbose:
            print(f"Error processing {operator}: {str(e)}")
        return None

def forecast_multiple_operators(country, operators, forecast_periods=20, variance_method='hybrid', zero_method='smart_interpolation', start_period='2014-Q1'):
    """
    Forecast subscriber count for multiple operators and display on one graph
    Modified to include start_period parameter (e.g., '2014-Q1')
    """
    print(f"\n{'='*80}")
    print(f"MULTI-OPERATOR SUBSCRIBER FORECASTING: {country}")
    print(f"Operators: {', '.join(operators)}")
    print(f"Forecast periods: {forecast_periods} quarters")
    print(f"Data starts from: {start_period}")
    print(f"{'='*80}\n")
    
    results = {}
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    # Process each operator
    for i, operator in enumerate(operators):
        print(f"Processing {operator}...")
        result = forecast_single_operator(
            country, operator, forecast_periods, variance_method, zero_method, start_period, verbose=True
        )
        
        if result is not None:
            results[operator] = result
            results[operator]['color'] = colors[i % len(colors)]
            
            # Print summary with better NaN handling
            hist_data = result['historical_data']
            forecast_data = result['forecast_results']
            
            print(f"  Historical data: {len(hist_data)} quarters ({hist_data.index[0]} to {hist_data.index[-1]})")
            print(f"  Current subscribers: {hist_data.iloc[-1]:,.0f}")
            
            # Check for NaN values before printing
            final_forecast = forecast_data.iloc[-1]['Forecast']
            if pd.isna(final_forecast):
                print(f"  Forecast end: Unable to calculate (NaN)")
                print(f"  Projected growth: Unable to calculate (NaN)")
            else:
                print(f"  Forecast end: {final_forecast:,.0f}")
                growth = ((final_forecast / hist_data.iloc[-1]) - 1) * 100
                print(f"  Projected growth: {growth:.1f}%")
            
            print(f"  Zero values handled: {result['zero_info']['original_zeros']}")
            print(f"  Model: SARIMA{result['model_info']['order']}{result['model_info']['seasonal_order']}")
            print()
        else:
            print(f"  Failed to process {operator}")
    
    if not results:
        print("No operators could be processed successfully.")
        return
    
    # Create the combined visualization
    plt.figure(figsize=(20, 12))
    
    # Plot each operator
    for operator, result in results.items():
        hist_data = result['historical_data']
        forecast_data = result['forecast_results']
        color = result['color']
        
        # Convert to timestamp for plotting
        hist_idx = hist_data.index.to_timestamp()
        forecast_idx = forecast_data.index.to_timestamp()
        
        # Historical data
        plt.plot(hist_idx, hist_data.values, 'o-', 
                label=f'{operator} (Historical)', 
                linewidth=2.5, markersize=6, color=color)
        
        # Forecast (only plot if not NaN)
        forecast_values = forecast_data['Forecast'].values
        if not pd.isna(forecast_values).any():
            plt.plot(forecast_idx, forecast_values, 's--', 
                    label=f'{operator} (Forecast)', 
                    linewidth=2.5, markersize=6, color=color, alpha=0.8)
            
            # Confidence intervals
            plt.fill_between(forecast_idx, 
                            forecast_data['Lower_Bound'].values, 
                            forecast_data['Upper_Bound'].values,
                            alpha=0.2, color=color, 
                            label=f'{operator} (95% CI)')
        
        # Highlight zero positions if any
        if result['zero_info']['zero_positions']:
            zero_idx = [idx.to_timestamp() for idx in result['zero_info']['zero_positions']]
            zero_vals = [hist_data[idx] for idx in result['zero_info']['zero_positions']]
            plt.scatter(zero_idx, zero_vals, color='yellow', s=100, 
                       zorder=5, edgecolors='black', alpha=0.7)
    
    # Create summary statistics table
    summary_stats = []
    for operator, result in results.items():
        hist_data = result['historical_data']
        forecast_data = result['forecast_results']
        
        current_subs = hist_data.iloc[-1]
        forecast_end = forecast_data.iloc[-1]['Forecast']
        
        if pd.isna(forecast_end):
            forecast_end_str = "N/A"
            growth_str = "N/A"
        else:
            forecast_end_str = f"{forecast_end:,.0f}"
            growth = ((forecast_end / current_subs) - 1) * 100
            growth_str = f"{growth:.1f}%"
        
        summary_stats.append([
            operator,
            f"{current_subs:,.0f}",
            forecast_end_str,
            growth_str
        ])
    
    # Add statistics table as text
    stats_text = "OPERATOR COMPARISON:\n"
    stats_text += f"{'Operator':<12} {'Current':<12} {'Forecast':<12} {'Growth':<8}\n"
    stats_text += "-" * 50 + "\n"
    
    for row in summary_stats:
        stats_text += f"{row[0]:<12} {row[1]:<12} {row[2]:<12} {row[3]:<8}\n"
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Formatting
    plt.title(f'Multi-Operator Subscriber Forecasting: {country} (From {start_period})', fontsize=16, fontweight='bold')
    plt.xlabel('Quarter', fontsize=12)
    plt.ylabel('Subscribers', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.7)
    
    # Format y-axis
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    
    # Set x-axis to start from start_period
    plot_start_year = int(start_period.split('-Q')[0])
    plt.xlim(left=pd.Timestamp(f'{plot_start_year}-01-01'))
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed comparison
    print("\nDETAILED COMPARISON:")
    print("="*60)
    
    # Market share analysis (if we have data for the same periods)
    common_periods = None
    for operator, result in results.items():
        hist_data = result['historical_data']
        if common_periods is None:
            common_periods = set(hist_data.index)
        else:
            common_periods = common_periods.intersection(set(hist_data.index))
    
    if common_periods and len(common_periods) > 0:
        print("\nMARKET SHARE ANALYSIS (Latest Available Data):")
        latest_period = max(common_periods)
        total_subs = sum(results[op]['historical_data'][latest_period] for op in results.keys())
        
        for operator, result in results.items():
            subs = result['historical_data'][latest_period]
            market_share = (subs / total_subs) * 100
            print(f"  {operator}: {subs:,.0f} subscribers ({market_share:.1f}% market share)")
        
        print(f"  Total Market: {total_subs:,.0f} subscribers")
    
    # Growth rate comparison
    print("\nGROWTH RATE ANALYSIS:")
    for operator, result in results.items():
        hist_data = result['historical_data']
        if len(hist_data) > 1:
            historical_growth = ((hist_data.iloc[-1] / hist_data.iloc[0]) ** (1/len(hist_data)) - 1) * 100
            
            forecast_data = result['forecast_results']
            forecast_end = forecast_data.iloc[-1]['Forecast']
            
            if pd.isna(forecast_end):
                forecast_growth_str = "N/A"
            else:
                forecast_growth = ((forecast_end / hist_data.iloc[-1]) - 1) * 100
                forecast_growth_str = f"{forecast_growth:.1f}%"
            
            print(f"  {operator}:")
            print(f"    Historical quarterly growth: {historical_growth:.2f}%")
            print(f"    Projected growth: {forecast_growth_str}")
    
    return results

if __name__ == "__main__":
    # Example usage for Zambia with multiple operators
    # Modified to specify start_period='2014-Q1' to start from 1Q 2014
    operators = ["ZamtelMobile(Zamtel)", "MTN", "Airtel"]  # Add or modify operators as needed
    
    results = forecast_multiple_operators(
        country="Zambia",
        operators=operators,
        forecast_periods=20,
        variance_method='hybrid',
        zero_method='smart_interpolation',
        start_period='2014-Q1'  # This will filter data to start from 1Q 2014
    )