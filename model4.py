import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from pmdarima import auto_arima

# Load the data
df = pd.read_csv('cleaned.csv')

# Explore the data structure
print(df.head())
print(df.info())

print(len(df.columns))  # Should match len(new_column_names)

# For this example, let's focus on one country's total connections
country = 'Ghana'
operator = 'MTN'  # Change to 'Sirtel', etc. as needed

# Filter for the specific operator
operator_data = df[(df['Country'] == country) & (df['Operator name'] == operator)].copy()

print(f"Operator data shape: {operator_data.shape}")
print(operator_data.iloc[:, 11:29].head())

# Identify annual columns only
date_columns = operator_data.columns[11:29]
annual_columns = [col for col in date_columns if col.isdigit() and len(col) == 4]
print(f"Annual columns found: {annual_columns}")

# Prepare time series for this operator
ts = operator_data[annual_columns].iloc[0]  # Get the first (and only) row as a Series
ts = pd.DataFrame(ts)
ts.columns = ['Total_Connections']
ts.index = pd.to_datetime(ts.index, format='%Y')

# Plot the time series
plt.figure(figsize=(12,6))
plt.plot(ts)
plt.title(f'Total Telecom Connections in {country} by {operator} (2006-2023)')
plt.xlabel('Year')
plt.ylabel('Connections')
plt.grid(True)
plt.show()

# Decompose the time series
decomposition = seasonal_decompose(ts, model='additive', period=1)  # No seasonal period for annual data
decomposition.plot()
plt.show()

# Autocorrelation plot
from pandas.plotting import autocorrelation_plot
autocorrelation_plot(ts)
plt.show()


# ADF Test for stationarity
def adf_test(timeseries):
    print("Results of Dickey-Fuller Test:")
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput[f'Critical Value ({key})'] = value
    print(dfoutput)

adf_test(ts['Total_Connections'])

# Differencing if not stationary
ts_diff = ts.diff().dropna()
adf_test(ts_diff['Total_Connections'])


# Check what years you actually have
print(f"Available years: {ts.index.year.tolist()}")
print(f"Data range: {ts.index.min()} to {ts.index.max()}")

# More flexible train/test split based on available data
split_year = ts.index[-3].year  # Use last 3 years as test set
train = ts[:str(split_year-1)]
test = ts[str(split_year):]

print(f"Training period: {train.index.min().year} to {train.index.max().year}")
print(f"Test period: {test.index.min().year} to {test.index.max().year}")


# Check if we have enough data
if len(ts) < 10:
    print(f"Warning: Only {len(ts)} data points available. ARIMA may not be reliable.")

# More flexible train/test split
if len(ts) >= 10:
    split_point = len(ts) - 3  # Use last 3 years as test
    train = ts.iloc[:split_point]
    test = ts.iloc[split_point:]
else:
    # Use 80/20 split for small datasets
    split_point = int(len(ts) * 0.8)
    train = ts.iloc[:split_point]
    test = ts.iloc[split_point:]

print(f"Training data: {len(train)} points")
print(f"Test data: {len(test)} points")

# Check for NaNs in train and test sets
print("Any NaNs in train?", train.isna().any())
print("Any NaNs in test?", test.isna().any())
print("NaN rows in train:\n", train[train.isna().any(axis=1)])
print("NaN rows in test:\n", test[test.isna().any(axis=1)])

train = train.dropna()
test = test.dropna()

# If you have pmdarima installed, use auto_arima
# pip install pmdarima
try:

    auto_model = auto_arima(train['Total_Connections'],
                           start_p=0, start_q=0,
                           max_p=3, max_q=3,
                           seasonal=False,
                           stepwise=True,
                           suppress_warnings=True,
                           error_action='ignore')

    print("Auto ARIMA found best model:")
    print(auto_model.summary())

    # Use the auto model for forecasting
    forecast = auto_model.predict(n_periods=len(test))
    forecast = pd.DataFrame(forecast, index=test.index, columns=['Prediction'])

except ImportError:
    print("pmdarima not installed. Install with: pip install pmdarima")

# Fit ARIMA model
try:
    model = ARIMA(train, order=(1,1,1))
    model_fit = model.fit()
    print(model_fit.summary())
    
    # Forecast
    forecast = model_fit.forecast(steps=len(test))
    forecast = pd.DataFrame(forecast, index=test.index, columns=['Prediction'])
    
    # Plot results
    plt.figure(figsize=(12,6))
    plt.plot(train, label='Training')
    plt.plot(test, label='Actual')
    plt.plot(forecast, label='Predicted')
    plt.title(f'Telecom Connections Forecast for {country} by {operator}')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(test, forecast))
    print(f'RMSE: {rmse}')

    
except Exception as e:
    print(f"Error fitting ARIMA model: {e}")
    print("This might be due to insufficient data or other issues.")

# Print the actual vs predicted values
    print("Actual vs Predicted values:")
    print("Actual:", test['Total_Connections'].values)
    print("Predicted:", forecast['Prediction'].values)
    print("Difference:", (test['Total_Connections'] - forecast['Prediction']).values)
