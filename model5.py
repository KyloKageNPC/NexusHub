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
from sklearn.model_selection import TimeSeriesSplit  # <-- Add this import
from monitoring_dashboard import log_forecast
from monitor import log_forecast

# Add cross-validation function here
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

# Suppress warnings for cleaner output
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

def analyze_and_forecast(country, operator):
    """
    Perform full analysis and forecasting for a country-operator pair
    """
    print(f"\n{'='*50}")
    print(f"ANALYZING: {operator} in {country}")
    print(f"{'='*50}\n")
    
    # Prepare quarterly data
    ts = prepare_quarterly_data(country, operator)

    print(f"Quarterly data from {ts.index[0]} to {ts.index[-1]}")
    print(f"Number of quarters: {len(ts)}")

    # Add summary statistics here
    print(f"Mean connections: {ts.mean():,.2f}")
    print(f"Standard deviation: {ts.std():,.2f}")

    # Plot the time series
    plt.figure(figsize=(14, 7))
    ts.plot(title=f'Quarterly Telecom Connections: {operator} in {country}', 
            xlabel='Quarter', ylabel='Connections', grid=True)
    plt.tight_layout()
    plt.show()
    
    # Seasonal decomposition
    try:
        decomposition = seasonal_decompose(ts.dropna(), model='multiplicative', period=4)
        # Convert PeriodIndex to DatetimeIndex for plotting
        idx = decomposition.observed.index.to_timestamp()
        fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
        axes[0].plot(idx, decomposition.observed.values)
        axes[0].set_ylabel('Observed')
        axes[1].plot(idx, decomposition.trend.values)
        axes[1].set_ylabel('Trend')
        axes[2].plot(idx, decomposition.seasonal.values)
        axes[2].set_ylabel('Seasonal')
        axes[3].plot(idx, decomposition.resid.values)
        axes[3].set_ylabel('Residual')
        axes[3].set_xlabel('Date')
        fig.suptitle(f'Seasonal Decomposition: {operator} in {country}', fontsize=16)
        plt.tight_layout()
        plt.show()
    except ValueError as e:
        print(f"Seasonal decomposition failed: {str(e)}")
    
    # Stationarity check with ADF test
    def adf_test(timeseries):
        print("Augmented Dickey-Fuller Test Results:")
        dftest = adfuller(timeseries, autolag='AIC')
        dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', 
                                               '#Lags Used', 'Observations Used'])
        for key, value in dftest[4].items():
            dfoutput[f'Critical Value ({key})'] = value
        print(dfoutput)
    
    print("\nStationarity Check - Original Data:")
    adf_test(ts)
    
    # Auto ARIMA to find best parameters
    print("\nSearching for best SARIMA parameters...")
    auto_model = auto_arima(
        ts,
        start_p=0, d=1, start_q=0,
        start_P=0, D=1, start_Q=0,
        max_p=3, max_q=3,
        max_P=2, max_Q=2,
        m=4,  # Quarterly seasonality
        seasonal=True,
        stepwise=True,
        suppress_warnings=True,
        error_action='ignore',
        trace=True
    )
    
    print("\nBest SARIMA model identified:")
    print(auto_model.summary())
    
    # Split data into train and test (last 4 quarters for testing)
    train = ts.iloc[:-4]
    test = ts.iloc[-4:]
    
    print(f"\nTraining period: {train.index[0]} to {train.index[-1]}")
    print(f"Testing period: {test.index[0]} to {test.index[-1]}")
    
    # Fit SARIMA model with identified parameters
    order = auto_model.order
    seasonal_order = auto_model.seasonal_order

    print(f"\nFitting SARIMA{order}{seasonal_order} model...")
    model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)

    # Cross-validation to assess model robustness
    cv_rmse = cross_validate_model(ts, order, seasonal_order)
    print(f"\nCross-validated RMSE (model robustness): {cv_rmse:,.2f}")

    # Forecast next 4 quarters
    forecast = model_fit.get_forecast(steps=4)
    forecast_mean = forecast.predicted_mean
    conf_int = forecast.conf_int()
    
    # Create results DataFrame
    results = pd.DataFrame({
        'Actual': test.values,
        'Predicted': forecast_mean.values
    }, index=test.index)
    
    results['Error'] = results['Actual'] - results['Predicted']
    results['AbsoluteError'] = np.abs(results['Error'])
    results['PercentageError'] = (results['Error'] / results['Actual']) * 100
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(results['Actual'], results['Predicted']))
    mae = mean_absolute_error(results['Actual'], results['Predicted'])
    mape = mean_absolute_percentage_error(results['Actual'], results['Predicted'])
    
    print("\nForecast Results:")
    print(results)
    print("\nEvaluation Metrics:")
    print(f"RMSE: {rmse:,.2f}")
    print(f"MAE: {mae:,.2f}")
    print(f"MAPE: {mape:.2%}")
    
    # Plot results
    plt.figure(figsize=(14, 8))
    
    # Convert PeriodIndex to DatetimeIndex for plotting
    ts_idx = ts.index.to_timestamp()
    results_idx = results.index.to_timestamp()
    
    # Plot historical data
    plt.plot(ts_idx, ts.values, 'o-', label='Historical Data', alpha=0.7)
    
    # Plot forecast
    plt.plot(results_idx, results['Predicted'], 's--', color='red', 
             label='Forecast', markersize=8)
    
    # Confidence interval
    plt.fill_between(results_idx, 
                    conf_int.iloc[:, 0], 
                    conf_int.iloc[:, 1], 
                    color='pink', alpha=0.3, label='95% Confidence Interval')
    
    # Annotate actual values
    for idx, row in results.iterrows():
        plt.annotate(f"Actual: {row['Actual']/1e6:.2f}M", 
                    (idx.to_timestamp(), row['Actual']), 
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center')
    
    plt.title(f'Quarterly Connections Forecast: {operator} in {country}', fontsize=16)
    plt.xlabel('Quarter')
    plt.ylabel('Connections')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    # Prepare data for logging
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
        'cv_rmse': cv_rmse
    }

    # Log to monitoring database
    log_forecast(country, operator, metrics, forecast_data)

    return results

# Example usage
if __name__ == "__main__":
    # Ghana-MTN example
    ghana_results = analyze_and_forecast('Ghana', 'MTN')
    
    # You can add more country-operator pairs:
    # nigeria_results = analyze_and_forecast('Nigeria', 'Airtel')
    # kenya_results = analyze_and_forecast('Kenya', 'Safaricom')