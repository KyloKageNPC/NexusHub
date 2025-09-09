import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
import warnings

warnings.filterwarnings('ignore')

# --- Load CSV ---
df = pd.read_csv("cleaner.csv")

# ------------------- Data Handling -------------------

def handle_zero_values(ts, method='smart_interpolation'):
    """
    Handles zero values using different strategies.
    """
    ts_cleaned = ts.copy()
    if method == 'set_null':
        ts_cleaned = ts.replace(0, np.nan)
    elif method == 'interpolation':
        ts_cleaned = ts.replace(0, np.nan).interpolate().fillna(method='bfill').fillna(method='ffill')
    elif method == 'smart_interpolation':
        zero_positions = ts_cleaned == 0
        if zero_positions.any():
            ts_cleaned = ts_cleaned.replace(0, np.nan).interpolate()
            ts_cleaned = ts_cleaned.fillna(ts_cleaned.mean())
    return ts_cleaned

def get_quarterly_columns():
    """
    Detects all quarterly columns in the CSV.
    """
    return [col for col in df.columns if any(q in col for q in ["1Q", "2Q", "3Q", "4Q"])]

def prepare_time_series(country, operator, zero_method='smart_interpolation'):
    """
    Extract & clean a time series for the given country & operator.
    """
    op_data = df[(df['Country'] == country) & (df['Operator name'] == operator)]
    if op_data.empty:
        raise ValueError(f"No data for {operator} in {country}")
    
    # Get quarterly columns that actually have non-empty values for this operator
    q_cols = [col for col in get_quarterly_columns() if pd.notna(op_data[col].values[0])]
    
    quarters, values = [], []
    for col in q_cols:
        raw_val = op_data[col].values[0]
        
        try:
            if ' ' in col:
                q, year = col.split()
            else:
                q = col[:2]
                year = col[2:]
            
            quarter_num = int(q[0])
            year_num = int(year)
            
            quarters.append(f"{year_num}-Q{quarter_num}")
            values.append(float(raw_val))
        except:
            continue
    
    ts = pd.Series(values, index=pd.PeriodIndex(quarters, freq="Q")).sort_index()
    ts = handle_zero_values(ts, zero_method)
    ts = ts.fillna(method='ffill').fillna(method='bfill')
    return ts



# ------------------- Forecasting -------------------

def forecast_series(ts, periods=8):
    """
    Fits SARIMA via auto_arima and generates forecasts.
    """
    auto_model = auto_arima(ts, seasonal=True, m=4, stepwise=True, suppress_warnings=True)
    model = SARIMAX(ts, order=auto_model.order, seasonal_order=auto_model.seasonal_order)
    model_fit = model.fit(disp=False)
    
    forecast = model_fit.get_forecast(steps=periods)
    forecast_df = pd.DataFrame({
        "Forecast": forecast.predicted_mean,
        "Lower_Bound": forecast.conf_int().iloc[:, 0],
        "Upper_Bound": forecast.conf_int().iloc[:, 1]
    }, index=pd.period_range(start=ts.index[-1] + 1, periods=periods, freq="Q"))
    
    return forecast_df, auto_model.order, auto_model.seasonal_order

# ------------------- Plotting -------------------

def plot_forecasts(country, operators, periods=8, zero_method='smart_interpolation'):
    """
    Plots historical + forecast for multiple operators in a country.
    """
    plt.figure(figsize=(14, 7))
    
    for op in operators:
        ts = prepare_time_series(country, op, zero_method)
        forecast_df, order, seasonal_order = forecast_series(ts, periods)
        
        # Historical
        plt.plot(ts.index.to_timestamp(), ts.values, marker='o', label=f"{op} - Historical")
        
        # Forecast
        plt.plot(forecast_df.index.to_timestamp(), forecast_df["Forecast"], linestyle='--', marker='s', label=f"{op} - Forecast")
        
        # Confidence Interval
        plt.fill_between(forecast_df.index.to_timestamp(),
                         forecast_df["Lower_Bound"], forecast_df["Upper_Bound"], alpha=0.2)
        
        print(f"{op} Model: SARIMA{order}{seasonal_order}")
        print(f"{op} Latest Value: {ts.iloc[-1]:,.0f}")
        print(f"{op} Forecast End: {forecast_df['Forecast'].iloc[-1]:,.0f}")
        print()
    
    plt.title(f"Multi-Operator Forecasts: {country}", fontsize=16)
    plt.xlabel("Quarter")
    plt.ylabel("Subscribers")
    plt.legend()
    plt.grid(True)
    plt.show()

# ------------------- Example Usage -------------------

if __name__ == "__main__":
    country = "Zambia"
    selected_operators = ["ZamtelMobile(Zamtel)", "MTN", "Airtel"]
    plot_forecasts(country, selected_operators, periods=8)
