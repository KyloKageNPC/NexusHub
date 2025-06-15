import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

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

model = Prophet(
    yearly_seasonality=True,   # Capture yearly trends
    weekly_seasonality=False,  # No weekly data
    changepoint_prior_scale=0.5  # Adjust sensitivity to trend changes
)
model.fit(data)

future = model.make_future_dataframe(periods=5, freq='YE')  # Forecast 5 years(YE reprisents year end)

forecast = model.predict(future)

fig = model.plot(forecast)
plt.title('Cameroon Mobile Connections Forecast (Total Market)')
plt.xlabel('Year')
plt.ylabel('Connections')
plt.show()