import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

class ProphetTelecomForecaster:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.data = None
        self.prophet_model = None
        self.prophet_forecast = None
        
    def load_and_prepare_data(self, country='Cameroon'):
        """Load and prepare data for forecasting with robust error handling"""
        print(f"\n[DEBUG] Loading data from {self.csv_file}...")
        
        try:
            # Load with checking header
            df = pd.read_csv(self.csv_file)
            print(f"[DEBUG] Initial data shape: {df.shape}")
            
            if df.empty:
                raise ValueError("Loaded empty dataframe after skipping rows")
                
            # Print available values for debugging
            print("\n[DEBUG] Available values in data:")
            print(f"Countries: {df['Country'].unique()}")
            print(f"Metrics: {df['Metric'].unique()}")
            print(f"Operators: {df['Operator name'].unique()}")
            
            # Case-insensitive filtering with whitespace handling
            filtered_data = df[
                (df['Country'].str.strip().str.lower() == country.lower()) & 
                (df['Metric'].str.strip().str.lower().str.contains('total.*including iot', regex=True)) & 
                (df['Operator name'].str.strip().str.lower() == 'total market'.lower())
            ].copy()
            
            print(f"[DEBUG] After filtering shape: {filtered_data.shape}")
            
            if len(filtered_data) == 0:
                available = f"\nAvailable options:\nCountry: {df['Country'].unique()}\nMetric: {df['Metric'].unique()}\nOperators: {df['Operator name'].unique()}"
                raise ValueError(f"No matching data found for Country='{country}', Metric='total – including IoT', Operator='Total market'{available}")
            
            # Get only 4-digit year columns
            year_cols = [col for col in filtered_data.columns if col.strip().isdigit() and len(col.strip()) == 4]
            if not year_cols:
                raise ValueError("No valid year columns found in data")
            
            print(f"[DEBUG] Using year columns: {year_cols}")
            
            # Melt yearly columns into rows
            data = filtered_data.melt(
                id_vars=['Country'], 
                value_vars=year_cols,
                var_name='ds', 
                value_name='y'
            )
            
            # Robust numeric conversion
            print("\n[DEBUG] Sample values before cleaning:")
            print(data['y'].head())
            
            data['y'] = pd.to_numeric(
                data['y'].astype(str)
                .str.replace('[^0-9.-]', '', regex=True),
                errors='coerce'
            )
            
            # Date conversion
            data['ds'] = pd.to_datetime(
                data['ds'].str.strip() + '-12-31',
                errors='coerce'
            )
            
            # Final cleaning
            data = data.dropna().sort_values('ds')
            
            print("\n[DEBUG] Sample values after cleaning:")
            print(data.head())
            
            if len(data) < 2:
                raise ValueError(f"Only {len(data)} valid rows after cleaning. Need at least 2.")
            
            self.data = data
            print(f"\n[SUCCESS] Loaded {len(data)} data points from {data['ds'].min().year} to {data['ds'].max().year}")
            return data
            
        except Exception as e:
            print(f"\n[ERROR] Data loading failed: {str(e)}")
            raise
    
    def enhanced_prophet_model(self):
        """Create enhanced Prophet model with improved parameters"""
        if self.data is None:
            raise ValueError("No data loaded. Call load_and_prepare_data() first.")
            
        try:
            data = self.data.copy()
            print("\n[DEBUG] Preparing Prophet model...")
            
            # Add features for enhanced model
            data['custom_trend'] = np.arange(len(data))
            data['custom_trend_squared'] = data['custom_trend'] ** 2
            data['economic_cycle'] = np.sin(2 * np.pi * data['custom_trend'] / 7)  # 7-year cycle
            
            # Dynamic cap calculation
            growth_rate = data['y'].pct_change().mean()
            data['cap'] = data['y'].max() * (1 + max(0.5, growth_rate * 10))
            data['floor'] = data['y'].min() * 0.1
            
            # Configure model
            model = Prophet(
                growth='logistic',
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                seasonality_mode='multiplicative',
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0,
                holidays_prior_scale=10.0,
                interval_width=0.95,
                uncertainty_samples=1000,
                mcmc_samples=0
            )
            
            # Add custom components
            model.add_seasonality(name='business_cycle', period=2555, fourier_order=3)
            model.add_regressor('custom_trend', standardize=True)
            model.add_regressor('custom_trend_squared', standardize=True)
            model.add_regressor('economic_cycle', standardize=True)
            
            # Fit model
            print("[DEBUG] Fitting Prophet model...")
            model.fit(data)
            self.prophet_model = model
            
            # Create future dataframe
            future_periods = 5
            future = model.make_future_dataframe(periods=future_periods, freq='A')
            
            # Add future regressors
            future['custom_trend'] = np.arange(len(future))
            future['custom_trend_squared'] = future['custom_trend'] ** 2
            future['economic_cycle'] = np.sin(2 * np.pi * future['custom_trend'] / 7)
            future['cap'] = data['cap'].iloc[-1]
            future['floor'] = data['floor'].iloc[-1]
            
            # Make prediction
            forecast = model.predict(future)
            self.prophet_forecast = forecast
            
            print("[SUCCESS] Prophet model trained successfully")
            return model, forecast
            
        except Exception as e:
            print(f"\n[ERROR] Model training failed: {str(e)}")
            raise

    # [Keep the rest of your methods unchanged - evaluate_model(), plot_results(), print_forecast()]

if __name__ == "__main__":
    try:
        # Initialize with debug
        print("Starting telecom forecasting...")
        forecaster = ProphetTelecomForecaster('cleaned.csv')
        
        # Load data with debug
        print("\nLoading data...")
        data = forecaster.load_and_prepare_data('Cameroon')
        
        # Build model with debug
        print("\nBuilding model...")
        prophet_model, prophet_forecast = forecaster.enhanced_prophet_model()
        
        # Evaluate and show results
        print("\nEvaluating model...")
        performance = forecaster.evaluate_model()
        
        print("\nGenerating forecast...")
        forecaster.print_forecast()
        
        print("\nPlotting results...")
        forecaster.plot_results()
        
        print("\n[SUCCESS] Forecasting completed successfully")
        
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Pipeline failed: {str(e)}")