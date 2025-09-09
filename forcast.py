import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import numpy as np 

# Load the dataset
df = pd.read_csv('Telco Actuals Africa.csv', skiprows=6)  # Skip metadata rows

# Example: Cameroon's total connections (including IoT)
cameroon = df[
    (df['Country'] == 'Cameroon') & 
    (df['Metric'] == 'total – including IoT') & 
    (df['Operator name'] == 'Total market')
]

# Melt yearly columns into rows
data = cameroon.melt(
    id_vars=['Country'], 
    value_vars=[col for col in cameroon.columns if col.isdigit()],  # Year columns
    var_name='ds', 
    value_name='y'
)

# Remove spaces and convert to numeric
data['y'] = pd.to_numeric(data['y'].astype(str).str.replace(' ', ''), errors='coerce')

# Convert year to datetime (end-of-year)
data['ds'] = pd.to_datetime(data['ds'] + '-12-31')  # Format: 'YYYY-12-31'
data = data.dropna()  # Remove missing values

# After preparing 'data' and before model.fit()
def fourier_terms(df, period, order):
    t = np.arange(len(df)) / period
    for i in range(1, order + 1):
        df[f'fourier_sin_{i}'] = np.sin(2 * np.pi * i * t)
        df[f'fourier_cos_{i}'] = np.cos(2 * np.pi * i * t)

fourier_terms(data, period=4, order=3)  # Example: Quarterly seasonality

model = Prophet(
    interval_width=0.95,
    yearly_seasonality=True,
    #uncertainty_samples=1000,  # Increase from default 1000
    #mcmc_samples=300,          # Enable probabilistic sampling
    #changepoint_prior_scale=0.8  # Increase flexibility
)
# Add this before fitting
data = data.sort_values('ds')
data['cap'] = data['y'].max() * 1.5  # Logistic growth cap
model = Prophet(growth='logistic')  # Instead of default linear
for i in range(1, 4):
    model.add_regressor(f'fourier_sin_{i}')
    model.add_regressor(f'fourier_cos_{i}')


# Fit the model
model.fit(data)

future_dates = pd.date_range(
    start=data['ds'].max(),
    periods=6,  # 5 future periods plus the last historical date
    freq='A-DEC'  # Annual frequency, December end
)[1:]  # Exclude the first date which is the last historical date

# Convert to DataFrame with 'ds' column
future = pd.DataFrame({'ds': future_dates})
future['cap'] = data['cap'].max()  # Use the same cap as training data   
fourier_terms(future, period=4, order=3)

forecast = model.predict(future)

# Cross-validation to evaluate the model
from prophet.diagnostics import cross_validation
df_cv = cross_validation(
    model, 
    initial='3650 days',   # Train on first 20 years
    period='365 days',   # Evaluate yearly
    horizon='1095 days'  # Forecast 3 years ahead
)
# Import performance metrics
from prophet.diagnostics import performance_metrics
df_perf = performance_metrics(df_cv)
print(df_perf.head())


# Plot the forecast
fig = model.plot(forecast)
plt.title('Cameroon Mobile Connections Forecast (Total Market)')
plt.xlabel('Year')
plt.ylabel('Connections')
plt.show()

fig2 = model.plot_components(forecast)
plt.show()