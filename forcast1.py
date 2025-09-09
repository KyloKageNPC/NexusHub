import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, RegressorMixin
import plotly.graph_objects as go
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer

# Load data (assuming your CSV structure)
df = pd.read_csv('cleaned.csv')

# Identify columns that are years or quarters
id_vars = ['Country', 'Operator name']  # Add more if needed
value_vars = [col for col in df.columns if any(str(y) in col for y in range(2006, 2030)) or 'Q' in col]

# Melt the dataframe
df_long = df.melt(id_vars=id_vars, value_vars=value_vars,
                  var_name='period', value_name='connections')

# Parse year and quarter from 'period'
def parse_period(p):
    if 'Q' in p:
        q, y = p.split()
        return int(y), q
    else:
        return int(p), None

df_long[['year', 'quarter']] = df_long['period'].apply(
    lambda x: pd.Series(parse_period(x))
)

# Create time features
# Use raw string for regex and handle NaN quarters
quarter_num = df_long['quarter'].str.extract(r'(\d+)Q')[0]
quarter_num = pd.to_numeric(quarter_num, errors='coerce')

# If quarter is missing, set month to 1 (January), else calculate month from quarter
df_long['month'] = np.where(quarter_num.notna(), quarter_num * 3 - 2, 1)
df_long['date'] = pd.to_datetime(df_long['year'].astype(int).astype(str) + '-' + df_long['month'].astype(int).astype(str) + '-01')
df_long = df_long.sort_values(['Country', 'date'])

# Calculate key metrics from your CSV
df_long['penetration_growth'] = df_long.groupby('Country')['connections'].pct_change(4)  # YoY growth
df_long['market_share'] = df_long.groupby(['Country', 'date'])['connections'].transform(lambda x: x/x.sum())
df_long['hhi'] = df_long.groupby(['Country', 'date'])['market_share'].transform(lambda x: (x**2).sum() * 10000)

# Lag features
for lag in [1, 4, 8]:  # 1Q, 1Y, 2Y lags
    df_long[f'connections_lag_{lag}'] = df_long.groupby('Country')['connections'].shift(lag)

# Rolling features
df_long['rolling_avg_4q'] = df_long.groupby('Country')['connections'].rolling(4).mean().values
df_long['rolling_std_4q'] = df_long.groupby('Country')['connections'].rolling(4).std().values

# Drop NAs from feature creation
df_long = df_long.dropna(subset=['penetration_growth', 'connections_lag_4', 'rolling_avg_4q'])

# Place this after your feature engineering, before model training

class ProphetWrapper(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.model = Prophet(
            growth='logistic',
            yearly_seasonality=True,
            seasonality_mode='multiplicative'
        )
    
    def fit(self, X, y):
        train_df = pd.DataFrame({
            'ds': X['date'],
            'y': y,
            'cap': X['connections'].max() * 1.2  # Assume 20% growth potential
        })
        self.model.fit(train_df)
        return self
    
    def predict(self, X):
        future = pd.DataFrame({'ds': X['date']})
        future['cap'] = X['connections'].max() * 1.2
        forecast = self.model.predict(future)
        return forecast['yhat'].values

# Prepare features
features = [
    'connections_lag_1', 'connections_lag_4', 'connections_lag_8',
    'rolling_avg_4q', 'rolling_std_4q', 'hhi', 'penetration_growth'
]

# For training, keep date/connections/Country for Prophet, but only numeric for RF
X = df_long[features + ['date', 'connections', 'Country']]
y = df_long['connections']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# When fitting the ensemble, use only numeric features for non-Prophet models
numeric_features = features  # Only numeric features

# Update the stacking regressor to pass only numeric features to RF

# Custom wrapper to select only numeric features for RF
class FeatureSelector(BaseEstimator, RegressorMixin):
    def __init__(self, features, base_model):
        self.features = features
        self.base_model = base_model
    def fit(self, X, y):
        return self.base_model.fit(X[self.features], y)
    def predict(self, X):
        return self.base_model.predict(X[self.features])

# Initialize models
prophet = ProphetWrapper()
rf = FeatureSelector(numeric_features, RandomForestRegressor(n_estimators=100, random_state=42))

# Stacking ensemble
ensemble = StackingRegressor(
    estimators=[
        ('prophet', prophet),
        ('random_forest', rf)
    ],
    final_estimator=LinearRegression(),
    cv=5
)

# Train
ensemble.fit(X_train, y_train)

# Evaluate
print(f"Ensemble R2 Score: {ensemble.score(X_test, y_test):.3f}")

def predict_country(country, periods=8):  # 2-year forecast
    country_df = df_long[df_long['Country'] == country].copy()
    
    # Prepare future dataframe
    last_date = country_df['date'].max()
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=3),
        periods=periods,
        freq='Q'
    )
    
    # Generate features for future
    future_df = pd.DataFrame({'date': future_dates})
    for lag in [1, 4, 8]:
        future_df[f'connections_lag_{lag}'] = country_df['connections'].iloc[-lag]
    
    # Add rolling features
    future_df['rolling_avg_4q'] = country_df['connections'].iloc[-4:].mean()
    future_df['rolling_std_4q'] = country_df['connections'].iloc[-4:].std()
    
    # Assume last known HHI and growth rate
    future_df['hhi'] = country_df['hhi'].iloc[-1]
    future_df['penetration_growth'] = country_df['penetration_growth'].iloc[-1]
    future_df['connections'] = country_df['connections'].iloc[-1] * (1.05 ** np.arange(1, periods+1))
    future_df['Country'] = country  # Add country column if needed by the model

    # Predict
    # Only pass numeric features + date/connections/Country as needed
    pred_input = future_df[features + ['date', 'connections', 'Country']]
    future_df['predictions'] = ensemble.predict(pred_input)
    
    return future_df[['date', 'predictions']]

# Example usage
cameroon_forecast = predict_country('Cameroon')
print(cameroon_forecast)

def plot_forecast(country):
    history = df_long[df_long['Country'] == country]
    forecast = predict_country(country)
    
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=history['date'],
        y=history['connections'],
        name='Actual',
        line=dict(color='#1f77b4')
    ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast['date'],
        y=forecast['predictions'],
        name='Forecast',
        line=dict(color='#ff7f0e', dash='dot')
    ))
    
    # Confidence band (simplified)
    fig.add_trace(go.Scatter(
        x=forecast['date'],
        y=forecast['predictions'] * 1.05,
        fill=None,
        mode='lines',
        line=dict(width=0),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast['date'],
        y=forecast['predictions'] * 0.95,
        fill='tonexty',
        mode='lines',
        line=dict(width=0),
        fillcolor='rgba(255,127,14,0.2)',
        name='Confidence'
    ))
    
    fig.update_layout(
        title=f'{country} Connections Forecast',
        xaxis_title='Date',
        yaxis_title='Connections',
        template='plotly_white'
    )
    
    # Export for Figma (as SVG)
    fig.write_image(f"{country}_forecast.svg")
    
    return fig

plot_forecast('Ghana').show()

