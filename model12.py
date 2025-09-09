import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from math import sqrt
import warnings
warnings.filterwarnings("ignore")

# Load the dataset (replace with your actual path)
df = pd.read_csv("cleaneder.csv")

# Filter for target operators and countries
operators = {
    "MTN Nigeria": ("MTN", "Nigeria"),
    "Airtel Nigeria": ("Airtel", "Nigeria"),
    "Vodacom South Africa": ("Vodacom", "South Africa")
}

# Extract yearly subscriber data (2006-2023)
years = [str(year) for year in range(2006, 2024)]
data = []
for op_name, (op, country) in operators.items():
    subset = df[(df["Operator name"] == op) & (df["Country"] == country)]
    if not subset.empty:
        row = subset.iloc[0]
        for year in years:
            if year in row and pd.notna(row[year]):
                data.append({
                    "Operator": op_name,
                    "Year": int(year),
                    "Subscribers": row[year]
                })

subscriber_df = pd.DataFrame(data)

# Pivot for plotting
pivot_df = subscriber_df.pivot(index="Year", columns="Operator", values="Subscribers")

# Plot historical trends
plt.figure(figsize=(14, 7))
pivot_df.plot(marker="o", linestyle="-")
plt.title("Historical Subscriber Counts (2006-2023)", fontsize=16)
plt.ylabel("Subscribers (millions)", fontsize=12)
plt.xlabel("Year", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend(title="Operator", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

# ARIMA Forecasting for each operator
def forecast_subscribers(data, operator_name, steps=12):
    # Check stationarity
    result = adfuller(data)
    print(f"\n{operator_name} - ADF Test:")
    print(f"  ADF Statistic: {result[0]:.3f}")
    print(f"  p-value: {result[1]:.3f}")
    
    # Differencing if non-stationary (p-value > 0.05)
    d = 0
    if result[1] > 0.05:
        d = 1
        data_diff = data.diff().dropna()
        print("  Differencing applied (d=1)")
    else:
        data_diff = data.copy()
    
    # Plot ACF/PACF to guide ARIMA parameters
    max_lags = min(10, (len(data_diff) // 2) - 1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(data_diff, lags=max_lags, ax=ax1)
    plot_pacf(data_diff, lags=max_lags, ax=ax2, method="ywm")
    plt.suptitle(f"ACF/PACF for {operator_name}", y=1.02)
    plt.tight_layout()
    plt.show()
    
    # Fit ARIMA model (auto-selection of p/q for simplicity)
    model = ARIMA(data, order=(1, d, 1))  # Start with (1,d,1); adjust based on ACF/PACF
    model_fit = model.fit()
    print(model_fit.summary())
    
    # Forecast
    forecast = model_fit.forecast(steps=steps)
    print(f"\nForecast for next {steps} months:")
    print(forecast)
    
    # Plot historical vs forecast
    plt.figure(figsize=(12, 6))
    data.plot(label="Historical", marker="o")
    forecast.plot(label="Forecast", color="red", linestyle="--", marker="o")
    plt.title(f"{operator_name} Subscriber Forecast", fontsize=16)
    plt.ylabel("Subscribers (millions)", fontsize=12)
    plt.xlabel("Year", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.show()
    
    return forecast

# Run forecasting for each operator
forecasts = {}
for operator in pivot_df.columns:
    forecasts[operator] = forecast_subscribers(
        pivot_df[operator].dropna(),
        operator_name=operator,
        steps=6  # Forecast next 6 steps (years or quarters)
    )