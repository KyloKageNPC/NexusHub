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
        """Load and prepare data for forecasting"""
        # Load the dataset
        df = pd.read_csv(self.csv_file)
        print("Columns:", df.columns.tolist())
        print("Unique countries:", df['Country'].unique() if 'Country' in df.columns else "No 'Country' column")
        print("Unique metrics:", df['Metric'].unique() if 'Metric' in df.columns else "No 'Metric' column")
        print("Unique operator names:", df['Operator name'].unique() if 'Operator name' in df.columns else "No 'Operator name' column")
        
        # Filter data for specific country
        filtered_data = df[
            (df['Country'] == country) & 
            (df['Metric'] == 'total – including IoT') & 
            (df['Operator name'] == 'Total market')
        ]
        
        # Melt yearly columns into rows
        data = filtered_data.melt(
            id_vars=['Country'], 
            value_vars=[col for col in filtered_data.columns if col.isdigit()],
            var_name='ds', 
            value_name='y'
        )
        
        # Clean and convert data
        data['y'] = pd.to_numeric(data['y'].astype(str).str.replace(' ', ''), errors='coerce')
        data['ds'] = pd.to_datetime(data['ds'] + '-12-31')
        data = data.dropna().sort_values('ds').reset_index(drop=True)
        
        self.data = data
        print(f"Data loaded for {country}: {len(data)} data points from {data['ds'].min().year} to {data['ds'].max().year}")
        return data
    
    def enhanced_prophet_model(self):
        """Create enhanced Prophet model with improved parameters"""
        data = self.data.copy()
        
        # Add external regressors (trends and patterns)
        data['custom_trend'] = np.arange(len(data))
        data['custom_trend_squared'] = data['custom_trend'] ** 2
        
        # Add economic cycle indicators (simplified)
        data['economic_cycle'] = np.sin(2 * np.pi * data['custom_trend'] / 7)  # 7-year economic cycle
        
        # Logistic growth with dynamic cap
        growth_rate = data['y'].pct_change().mean()
        data['cap'] = data['y'].max() * (1 + max(0.5, growth_rate * 10))  # Dynamic cap
        data['floor'] = data['y'].min() * 0.1  # Minimum floor
        
        # Enhanced Prophet model
        model = Prophet(
            growth='logistic',
            yearly_seasonality=True,
            weekly_seasonality=False,  # Not relevant for annual data
            daily_seasonality=False,
            seasonality_mode='multiplicative',  # Better for exponential growth
            changepoint_prior_scale=0.05,  # More flexible changepoints
            seasonality_prior_scale=10.0,  # More flexible seasonality
            holidays_prior_scale=10.0,
            interval_width=0.95,
            uncertainty_samples=1000,
            mcmc_samples=0  # Set to 300 for full Bayesian inference if needed
        )
        
        # Add custom seasonality
        model.add_seasonality(name='business_cycle', period=2555, fourier_order=3)  # ~7 year cycle
        
        # Add regressors
        model.add_regressor('custom_trend', standardize=True)
        model.add_regressor('custom_trend_squared', standardize=True)
        model.add_regressor('economic_cycle', standardize=True)
        
        # Fit model
        model.fit(data)
        self.prophet_model = model
        
        # Create future dataframe
        future_periods = 5
        future = model.make_future_dataframe(periods=future_periods, freq='A')
        
        # Add regressors to future dataframe
        future['custom_trend'] = np.arange(len(future))
        future['custom_trend_squared'] = future['custom_trend'] ** 2
        future['economic_cycle'] = np.sin(2 * np.pi * future['custom_trend'] / 7)
        future['cap'] = data['cap'].iloc[-1]
        future['floor'] = data['floor'].iloc[-1]
        
        # Make prediction
        forecast = model.predict(future)
        self.prophet_forecast = forecast
        
        return model, forecast
    
    def evaluate_model(self):
        """Evaluate Prophet model using cross-validation"""
        print("\n" + "="*50)
        print("PROPHET MODEL EVALUATION")
        print("="*50)
        
        if self.prophet_model:
            print("\nProphet Model Evaluation:")
            try:
                df_cv = cross_validation(
                    self.prophet_model, 
                    initial='2555 days',  # ~7 years
                    period='365 days',    # Evaluate yearly
                    horizon='730 days'    # Forecast 2 years ahead
                )
                df_perf = performance_metrics(df_cv)
                print("Prophet Cross-Validation Metrics:")
                print(df_perf[['horizon', 'mape', 'mae', 'rmse']].round(4))
                return df_perf
            except Exception as e:
                print(f"Cross-validation failed: {e}")
                return None
    
    def plot_results(self):
        """Plot Prophet model results"""
        if self.prophet_model and self.prophet_forecast is not None:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Prophet forecast plot
            ax1 = axes[0]
            self.prophet_model.plot(self.prophet_forecast, ax=ax1)
            ax1.set_title('Prophet Model Forecast')
            ax1.set_xlabel('Year')
            ax1.set_ylabel('Connections')
            ax1.grid(True, alpha=0.3)
            
            # Prophet components plot
            ax2 = axes[1]
            # Get components from the model
            components = self.prophet_model.predict(self.prophet_forecast)
            
            # Plot trend component
            ax2.plot(components['ds'], components['trend'], label='Trend', linewidth=2)
            if 'yearly' in components.columns:
                ax2_twin = ax2.twinx()
                ax2_twin.plot(components['ds'], components['yearly'], 
                             label='Yearly', color='orange', alpha=0.7)
                ax2_twin.set_ylabel('Yearly Component')
                ax2_twin.legend(loc='upper right')
            
            ax2.set_title('Prophet Model Trend')
            ax2.set_xlabel('Year')
            ax2.set_ylabel('Trend Component')
            ax2.legend(loc='upper left')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
    
    def print_forecast(self):
        """Print detailed Prophet forecast results"""
        print("\n" + "="*50)
        print("PROPHET FORECAST RESULTS")
        print("="*50)
        
        if self.prophet_forecast is not None:
            print("\nProphet Forecast (Next 5 Years):")
            prophet_future = self.prophet_forecast[self.prophet_forecast['ds'] > self.data['ds'].max()]
            for _, row in prophet_future.iterrows():
                print(f"{row['ds'].year}: {row['yhat']:,.0f} "
                      f"({row['yhat_lower']:,.0f} - {row['yhat_upper']:,.0f})")
            
            # Calculate growth rates
            print("\nYear-over-Year Growth Rates:")
            for i in range(1, len(prophet_future)):
                prev_val = prophet_future.iloc[i-1]['yhat']
                curr_val = prophet_future.iloc[i]['yhat']
                growth_rate = ((curr_val - prev_val) / prev_val) * 100
                year = prophet_future.iloc[i]['ds'].year
                print(f"{year}: {growth_rate:.1f}%")

# Usage
if __name__ == "__main__":
    # Initialize Prophet forecaster
    forecaster = ProphetTelecomForecaster('cleaned.csv')
    
    # Load and prepare data
    data = forecaster.load_and_prepare_data('Cameroon')
    
    # Create enhanced Prophet model
    print("\nBuilding Enhanced Prophet Model...")
    prophet_model, prophet_forecast = forecaster.enhanced_prophet_model()
    
    # Evaluate model
    performance = forecaster.evaluate_model()
    
    # Print forecasts
    forecaster.print_forecast()
    
    # Plot results
    forecaster.plot_results()
    
    # Additional Prophet-specific diagnostics
    print("\n" + "="*50)
    print("PROPHET MODEL DIAGNOSTICS")
    print("="*50)
    
    if forecaster.prophet_model:
        print("\nModel Parameters:")
        print(f"Growth: {forecaster.prophet_model.growth}")
        print(f"Changepoint Prior Scale: {forecaster.prophet_model.changepoint_prior_scale}")
        print(f"Seasonality Prior Scale: {forecaster.prophet_model.seasonality_prior_scale}")
        print(f"Number of Changepoints: {len(forecaster.prophet_model.changepoints)}")
        
        # Print changepoints
        changepoints = forecaster.prophet_model.changepoints
        if len(changepoints) > 0:
            print(f"\nDetected Changepoints:")
            for cp in changepoints[-5:]:  # Show last 5 changepoints
                print(f"  {cp.strftime('%Y-%m-%d')}")