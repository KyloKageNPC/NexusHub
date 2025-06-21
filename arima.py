import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')
import joblib
import matplotlib.dates as mdates
import sys

class ARIMATelecomForecaster:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.data = None
        self.arima_model = None
        self.arima_forecast = None
        self.best_params = None
        
    def load_and_prepare_data(self, country='Cameroon'):
        """Load and prepare data for forecasting"""
        # Load the dataset
        df = pd.read_csv(self.csv_file, skiprows=6)
        
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
    
    def check_stationarity(self, timeseries, title="Series"):
        """Check stationarity of time series using Augmented Dickey-Fuller test"""
        print(f"\nStationarity Test for {title}:")
        result = adfuller(timeseries)
        print(f'ADF Statistic: {result[0]:.6f}')
        print(f'p-value: {result[1]:.6f}')
        print(f'Critical Values:')
        for key, value in result[4].items():
            print(f'\t{key}: {value:.3f}')
        
        is_stationary = result[1] < 0.05
        print(f'Result: {"Stationary" if is_stationary else "Non-stationary"}')
        return is_stationary
    
    def plot_diagnostics(self):
        """Plot diagnostic plots for time series analysis"""
        if self.data is None:
            print("No data loaded. Please run load_and_prepare_data() first.")
            return
        
        ts = self.data.set_index('ds')['y']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Original series
        axes[0, 0].plot(ts)
        axes[0, 0].set_title('Original Time Series')
        axes[0, 0].grid(True, alpha=0.3)
        
        # First difference
        ts_diff = ts.diff().dropna()
        axes[0, 1].plot(ts_diff)
        axes[0, 1].set_title('First Difference')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Log transformation
        ts_log = np.log(ts)
        axes[0, 2].plot(ts_log)
        axes[0, 2].set_title('Log Transformation')
        axes[0, 2].grid(True, alpha=0.3)
        
        # ACF plot
        plot_acf(ts, ax=axes[1, 0], lags=min(len(ts)//2, 20))
        axes[1, 0].set_title('Autocorrelation Function')
        
        # PACF plot
        plot_pacf(ts, ax=axes[1, 1], lags=min(len(ts)//2, 20))
        axes[1, 1].set_title('Partial Autocorrelation Function')
        
        # Seasonal decomposition
        if len(ts) >= 8:  # Need at least 2 cycles for decomposition
            decomposition = seasonal_decompose(ts, model='multiplicative', period=min(4, len(ts)//2))
            axes[1, 2].plot(decomposition.trend, label='Trend')
            axes[1, 2].plot(decomposition.seasonal, label='Seasonal')
            axes[1, 2].set_title('Trend and Seasonal Components')
            axes[1, 2].legend()
        else:
            axes[1, 2].text(0.5, 0.5, 'Insufficient data\nfor decomposition', 
                           ha='center', va='center', transform=axes[1, 2].transAxes)
        
        plt.tight_layout()
        plt.show()
    
    def find_best_arima(self, max_p=3, max_d=2, max_q=3):  # Expanded ranges
        """Find best ARIMA parameters using grid search with stability checks"""
        ts = self.data.set_index('ds')['y']
        
        # Try log transformation if data is strictly positive
        if np.all(ts > 0):
            ts = np.log1p(ts)  # log1p handles small values better
        
        best_aic = np.inf
        best_params = None
        best_model = None
        results = []
        
        print("Searching for best ARIMA parameters...")
        print("Testing combinations (p, d, q):")
        
        # Try different combinations
        for d in range(max_d + 1):  # Try differencing first
            for p in range(max_p + 1):
                for q in range(max_q + 1):
                    try:
                        model = ARIMA(ts, order=(p, d, q))
                        fitted_model = model.fit(
                            method='lbfgs',  # Try different optimization method
                            maxiter=1000,    # Increase max iterations
                            cov_type='robust'  # Use robust covariance estimation
                        )
                        
                        # Calculate condition number
                        cond_number = np.linalg.cond(fitted_model.cov_params())
                        
                        # Check if model is stable and well-conditioned
                        if cond_number < 1e15:  # Less strict condition
                            aic = fitted_model.aic
                            bic = fitted_model.bic
                            
                            results.append({
                                'order': (p, d, q),
                                'aic': aic,
                                'bic': bic,
                                'cond_number': cond_number
                            })
                            
                            if aic < best_aic:
                                best_aic = aic
                                best_params = (p, d, q)
                                best_model = fitted_model
                                
                            print(f"  ARIMA{(p, d, q)} - AIC: {aic:.2f}, BIC: {bic:.2f}, Condition: {cond_number:.2e}")
                        
                    except Exception as e:
                        print(f"  ARIMA{(p, d, q)} - Failed: {str(e)}")
                        continue

        # If no model found, try simpler models
        if not best_model:
            print("\nTrying simplified models...")
            simple_combinations = [(1,1,0), (0,1,1), (1,1,1)]
            for p, d, q in simple_combinations:
                try:
                    model = ARIMA(ts, order=(p, d, q))
                    fitted_model = model.fit(method='lbfgs')
                    
                    aic = fitted_model.aic
                    if aic < best_aic:
                        best_aic = aic
                        best_params = (p, d, q)
                        best_model = fitted_model
                        print(f"  ARIMA{(p, d, q)} - AIC: {aic:.2f}")
                except:
                    continue

        if not best_model:
            raise ValueError("No stable ARIMA model found. Please check your data or try seasonal ARIMA (SARIMA).")
        
        self.best_params = best_params
        print(f"\nSelected model: ARIMA{best_params}")
        return best_model, best_params, best_aic
    
    def build_arima_model(self):
        """Build ARIMA model with automatic parameter selection"""
        data = self.data.copy()
        ts = data.set_index('ds')['y']
        
        # Check stationarity
        print("Analyzing time series properties...")
        is_stationary = self.check_stationarity(ts, "Original Series")
        
        # Check differencing if needed
        if not is_stationary:
            ts_diff = ts.diff().dropna()
            is_stationary_diff = self.check_stationarity(ts_diff, "First Difference")
            
            if not is_stationary_diff:
                ts_diff2 = ts_diff.diff().dropna()
                self.check_stationarity(ts_diff2, "Second Difference")
        
        # Find best ARIMA parameters
        print("\nFinding optimal ARIMA parameters...")
        best_model, best_params, best_aic = self.find_best_arima()
        print(f"\nSelected Model: ARIMA{best_params} with AIC: {best_aic:.2f}")
        
        # Forecast
        forecast_steps = 5
        forecast_result = best_model.forecast(steps=forecast_steps)
        conf_int = best_model.get_forecast(steps=forecast_steps).conf_int()
        
        # Create forecast dataframe
        last_date = ts.index[-1]
        future_dates = pd.date_range(start=last_date, periods=forecast_steps + 1, freq='A')[1:]
        
        arima_forecast = pd.DataFrame({
            'ds': future_dates,
            'yhat': forecast_result,
            'yhat_lower': conf_int.iloc[:, 0],
            'yhat_upper': conf_int.iloc[:, 1]
        })
        
        self.arima_model = best_model
        self.arima_forecast = arima_forecast
        
        return best_model, arima_forecast
    
    def evaluate_model(self):
        """Evaluate ARIMA model using train-test split"""
        print("\n" + "="*50)
        print("ARIMA MODEL EVALUATION")
        print("="*50)
        
        if self.arima_model and len(self.data) > 10:
            print("\nARIMA Model Evaluation:")
            # Use last 3 points for validation
            train_data = self.data[:-3]
            test_data = self.data[-3:]
            
            # Fit ARIMA on training data
            ts_train = train_data.set_index('ds')['y']
            model_temp = ARIMA(ts_train, order=self.arima_model.model.order)
            fitted_temp = model_temp.fit()
            
            # Forecast
            forecast_temp = fitted_temp.forecast(steps=3)
            actual = test_data['y'].values
            
            # Calculate metrics
            mae = mean_absolute_error(actual, forecast_temp)
            rmse = np.sqrt(mean_squared_error(actual, forecast_temp))
            mape = mean_absolute_percentage_error(actual, forecast_temp) * 100
            
            print(f"ARIMA Validation Metrics:")
            print(f"MAE: {mae:,.2f}")
            print(f"RMSE: {rmse:,.2f}")
            print(f"MAPE: {mape:.2f}%")
            
            # Residual analysis
            residuals = self.arima_model.resid
            print(f"\nResidual Analysis:")
            print(f"Mean of residuals: {residuals.mean():.4f}")
            print(f"Std of residuals: {residuals.std():.4f}")
            
            return {
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'residual_mean': residuals.mean(),
                'residual_std': residuals.std()
            }
    
    def plot_results(self):
        """Plot ARIMA model results and diagnostics"""
        if self.arima_model and self.arima_forecast is not None:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Historical data and forecast
            ax1 = axes[0, 0]
            ts = self.data.set_index('ds')['y']
            ts.plot(ax=ax1, label='Historical', color='blue', linewidth=2)
            
            # Plot ARIMA forecast
            self.arima_forecast.set_index('ds')['yhat'].plot(
                ax=ax1, label='ARIMA Forecast', color='red', linestyle='--', linewidth=2
            )
            
            # Plot confidence intervals
            # Ensure x-axis is in matplotlib date format
            x = mdates.date2num(self.arima_forecast['ds'])
            ax1.fill_between(
                x,
                self.arima_forecast['yhat_lower'],
                self.arima_forecast['yhat_upper'],
                alpha=0.3, color='red', label='Confidence Interval'
            )
            ax1.xaxis_date()  # Tell matplotlib to treat x-axis as dates
            
            ax1.set_title('ARIMA Model Forecast')
            ax1.set_xlabel('Year')
            ax1.set_ylabel('Connections')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Residuals plot
            ax2 = axes[0, 1]
            residuals = self.arima_model.resid
            residuals.plot(ax=ax2)
            ax2.set_title('Residuals')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Residuals')
            ax2.grid(True, alpha=0.3)
            
            # Residuals histogram
            ax3 = axes[1, 0]
            residuals.hist(ax=ax3, bins=10, edgecolor='black', alpha=0.7)
            ax3.set_title('Residuals Distribution')
            ax3.set_xlabel('Residuals')
            ax3.set_ylabel('Frequency')
            ax3.grid(True, alpha=0.3)
            
            # Q-Q plot of residuals
            ax4 = axes[1, 1]
            from scipy import stats
            stats.probplot(residuals, dist="norm", plot=ax4)
            ax4.set_title('Q-Q Plot of Residuals')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
    
    def print_forecast(self):
        """Print detailed ARIMA forecast results"""
        print("\n" + "="*50)
        print("ARIMA FORECAST RESULTS")
        print("="*50)
        
        if self.arima_forecast is not None:
            print(f"\nARIMA{self.best_params} Forecast (Next 5 Years):")
            for _, row in self.arima_forecast.iterrows():
                print(f"{row['ds'].year}: {row['yhat']:,.0f} "
                      f"({row['yhat_lower']:,.0f} - {row['yhat_upper']:,.0f})")
            
            # Calculate growth rates
            print("\nYear-over-Year Growth Rates:")
            last_historical = self.data['y'].iloc[-1]
            first_forecast = self.arima_forecast.iloc[0]['yhat']
            growth_rate = ((first_forecast - last_historical) / last_historical) * 100
            print(f"{self.arima_forecast.iloc[0]['ds'].year}: {growth_rate:.1f}%")
            
            for i in range(1, len(self.arima_forecast)):
                prev_val = self.arima_forecast.iloc[i-1]['yhat']
                curr_val = self.arima_forecast.iloc[i]['yhat']
                growth_rate = ((curr_val - prev_val) / prev_val) * 100
                year = self.arima_forecast.iloc[i]['ds'].year
                print(f"{year}: {growth_rate:.1f}%")

# Usage
if __name__ == "__main__":
    # Initialize ARIMA forecaster
    forecaster = ARIMATelecomForecaster('Telco Actuals Africa.csv')
    
    # Load and prepare data
    data = forecaster.load_and_prepare_data('Cameroon')
    
    # Plot diagnostic plots
    print("\nGenerating diagnostic plots...")
    forecaster.plot_diagnostics()
    
    # Build ARIMA model
    print("\nBuilding ARIMA Model...")
    arima_model, arima_forecast = forecaster.build_arima_model()
    
    # Evaluate model
    performance = forecaster.evaluate_model()
    
    # Print forecasts
    forecaster.print_forecast()
    
    # Plot results
    forecaster.plot_results()
    
    # Additional ARIMA-specific diagnostics
    print("\n" + "="*50)
    print("ARIMA MODEL DIAGNOSTICS")
    print("="*50)
    
    if forecaster.arima_model:
        print("\nARIMA Model Summary:")
        print(forecaster.arima_model.summary())
        
        # Model information
        print(f"\nModel Information:")
        print(f"Selected Order: {forecaster.best_params}")
        print(f"AIC: {forecaster.arima_model.aic:.2f}")
        print(f"BIC: {forecaster.arima_model.bic:.2f}")
        print(f"Log Likelihood: {forecaster.arima_model.llf:.2f}")

def forecast_country(country_name='Nigeria', test_points=3):
    """One-shot forecast pipeline for any country"""
    forecaster = ARIMATelecomForecaster('Telco Actuals Africa.csv')
    forecaster.load_and_prepare_data(country_name)
    forecaster.plot_diagnostics()
    forecaster.build_arima_model()
    forecaster.evaluate_model()
    forecaster.print_forecast()
    forecaster.plot_results()


if __name__ == "__main__":
    # Check for command-line argument for country
    if len(sys.argv) > 1:
        country = sys.argv[1]
        forecast_country(country)
    else:
        # Default to Cameroon if no argument is given
        forecast_country('Cameroon')