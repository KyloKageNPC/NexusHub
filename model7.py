import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
import warnings
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
from sklearn.neural_network import MLPRegressor

# For zero-inflated models
from scipy.stats import poisson, nbinom
from sklearn.mixture import GaussianMixture

warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv('cleaned.csv')

def prepare_quarterly_data_with_zeros(country, operator):
    """
    Prepare quarterly data while preserving zero values
    """
    operator_data = df[(df['Country'] == country) & (df['Operator name'] == operator)]
    
    if operator_data.empty:
        raise ValueError(f"No data found for {operator} in {country}")
    
    q_cols = [col for col in df.columns if any(q in col for q in ['1Q', '2Q', '3Q', '4Q'])]
    q_data = operator_data[q_cols].iloc[0]
    
    quarters = []
    values = []
    
    for col, val in q_data.items():
        try:
            if ' ' in col:
                q, year = col.split()
            else:
                q = col[:2]
                year = col[2:]
            
            quarter = int(q[0])
            year = int(year)
            quarters.append(f'{year}-Q{quarter}')
            
            # Handle zeros and missing values differently
            if pd.isna(val) or val == '':
                values.append(np.nan)  # Keep as NaN for missing data
            else:
                values.append(float(val))  # Keep zeros as they are real
                
        except Exception as e:
            print(f"Skipping invalid column {col}: {str(e)}")
    
    ts = pd.Series(values, index=pd.PeriodIndex(quarters, freq='Q'))
    ts = ts.sort_index()
    
    # Only fill NaN values, preserve zeros
    ts = ts.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    return ts

def analyze_zero_pattern(ts, country, operator):
    """
    Analyze the pattern of zeros in the time series
    """
    zero_count = (ts == 0).sum()
    zero_percentage = (zero_count / len(ts)) * 100
    
    print(f"\nZERO VALUE ANALYSIS for {operator} in {country}:")
    print(f"Total quarters: {len(ts)}")
    print(f"Quarters with zero connections: {zero_count}")
    print(f"Percentage of zeros: {zero_percentage:.1f}%")
    
    # Find zero streaks
    is_zero = ts == 0
    zero_streaks = []
    current_streak = 0
    
    for i, zero in enumerate(is_zero):
        if zero:
            current_streak += 1
        else:
            if current_streak > 0:
                zero_streaks.append(current_streak)
            current_streak = 0
    
    if current_streak > 0:
        zero_streaks.append(current_streak)
    
    if zero_streaks:
        print(f"Longest consecutive zero streak: {max(zero_streaks)} quarters")
        print(f"Average zero streak length: {np.mean(zero_streaks):.1f} quarters")
    
    return {
        'zero_count': zero_count,
        'zero_percentage': zero_percentage,
        'zero_streaks': zero_streaks,
        'has_many_zeros': zero_percentage > 20
    }

def create_features_for_ml(ts):
    """
    Create features for machine learning models
    """
    df_features = pd.DataFrame(index=ts.index)
    
    # Lag features
    for lag in [1, 2, 3, 4]:
        df_features[f'lag_{lag}'] = ts.shift(lag)
    
    # Rolling statistics (handle zeros appropriately)
    for window in [2, 4]:
        df_features[f'rolling_mean_{window}'] = ts.rolling(window=window).mean()
        df_features[f'rolling_std_{window}'] = ts.rolling(window=window).std()
        df_features[f'rolling_max_{window}'] = ts.rolling(window=window).max()
    
    # Seasonal features
    df_features['quarter'] = [int(str(idx).split('Q')[1]) for idx in ts.index]
    df_features['year'] = [int(str(idx).split('Q')[0]) for idx in ts.index]
    
    # Time trend
    df_features['time_trend'] = range(len(ts))
    
    # Binary indicators
    df_features['is_zero_lag1'] = (ts.shift(1) == 0).astype(int)
    df_features['is_zero_lag2'] = (ts.shift(2) == 0).astype(int)
    
    # Zero streak indicator
    is_zero = ts == 0
    zero_streak_length = []
    current_streak = 0
    
    for zero in is_zero:
        if zero:
            current_streak += 1
        else:
            current_streak = 0
        zero_streak_length.append(current_streak)
    
    df_features['zero_streak_length'] = zero_streak_length
    
    # Target variable
    df_features['target'] = ts.values
    
    # Drop rows with NaN (from lag features)
    df_features = df_features.dropna()
    
    return df_features

class ZeroInflatedRegressor:
    """
    Zero-Inflated model that combines binary classification and regression
    """
    def __init__(self, classifier=None, regressor=None):
        self.classifier = classifier or RandomForestRegressor(n_estimators=50, random_state=42)
        self.regressor = regressor or RandomForestRegressor(n_estimators=100, random_state=42)
        self.is_fitted = False
    
    def fit(self, X, y):
        # Create binary target (zero vs non-zero)
        y_binary = (y > 0).astype(int)
        
        # Fit classifier to predict zero vs non-zero
        self.classifier.fit(X, y_binary)
        
        # Fit regressor on non-zero values only
        non_zero_mask = y > 0
        if non_zero_mask.sum() > 0:
            X_non_zero = X[non_zero_mask]
            y_non_zero = y[non_zero_mask]
            self.regressor.fit(X_non_zero, y_non_zero)
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Predict probability of non-zero
        prob_non_zero = self.classifier.predict(X)
        
        # Predict values for non-zero cases
        predicted_values = self.regressor.predict(X)
        
        # Combine predictions
        final_predictions = prob_non_zero * predicted_values
        
        return final_predictions

def evaluate_models_for_zeros(country, operator):
    """
    Evaluate different modeling approaches for data with zeros
    """
    print(f"\n{'='*70}")
    print(f"EVALUATING ZERO-HANDLING MODELS: {operator} in {country}")
    print(f"{'='*70}\n")
    
    # Prepare data
    ts = prepare_quarterly_data_with_zeros(country, operator)
    zero_analysis = analyze_zero_pattern(ts, country, operator)
    
    # Create features
    df_features = create_features_for_ml(ts)
    
    if len(df_features) < 8:
        print("Not enough data points for model evaluation")
        return None
    
    # Prepare train/test split (last 25% for testing)
    test_size = max(2, len(df_features) // 4)
    train_data = df_features.iloc[:-test_size]
    test_data = df_features.iloc[-test_size:]
    
    # Features and target
    feature_cols = [col for col in df_features.columns if col != 'target']
    X_train = train_data[feature_cols]
    y_train = train_data['target']
    X_test = test_data[feature_cols]
    y_test = test_data['target']
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Models to evaluate
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'Zero-Inflated RF': ZeroInflatedRegressor(),
        'Neural Network': MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=500, random_state=42),
        'Ridge Regression': Ridge(alpha=1.0)
    }
    
    results = {}
    predictions = {}
    
    for name, model in models.items():
        try:
            print(f"Training {name}...")
            
            # Use scaled features for Neural Network and Ridge
            if name in ['Neural Network', 'Ridge Regression']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Ensure non-negative predictions
            y_pred = np.maximum(y_pred, 0)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            # Handle MAPE carefully with zeros
            non_zero_mask = y_test != 0
            if non_zero_mask.sum() > 0:
                mape = mean_absolute_percentage_error(y_test[non_zero_mask], y_pred[non_zero_mask])
            else:
                mape = np.inf
            
            # Zero prediction accuracy
            zero_accuracy = np.mean((y_test == 0) == (y_pred < 0.5))
            
            results[name] = {
                'RMSE': rmse,
                'MAE': mae,
                'MAPE': mape,
                'Zero_Accuracy': zero_accuracy,
                'Model': model
            }
            
            predictions[name] = y_pred
            
        except Exception as e:
            print(f"Error with {name}: {str(e)}")
            continue
    
    # Display results
    results_df = pd.DataFrame(results).T
    print("\nMODEL COMPARISON RESULTS:")
    print(results_df[['RMSE', 'MAE', 'MAPE', 'Zero_Accuracy']].round(4))
    
    # Find best model
    best_model_name = results_df['RMSE'].idxmin()
    best_model = results[best_model_name]['Model']
    
    print(f"\nBest Model: {best_model_name}")
    
    # Detailed analysis of best model
    best_predictions = predictions[best_model_name]
    
    # Create detailed results
    detailed_results = pd.DataFrame({
        'Actual': y_test.values,
        'Predicted': best_predictions,
        'Error': y_test.values - best_predictions,
        'Absolute_Error': np.abs(y_test.values - best_predictions)
    }, index=y_test.index)
    
    print(f"\nDETAILED RESULTS FOR {best_model_name}:")
    print(detailed_results)
    
    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Time series plot
    ts_idx = ts.index.to_timestamp()
    test_idx = test_data.index.to_timestamp()
    
    axes[0,0].plot(ts_idx, ts.values, 'o-', label='Historical Data', alpha=0.7)
    axes[0,0].plot(test_idx, y_test.values, 'go', label='Actual Test', markersize=8)
    axes[0,0].plot(test_idx, best_predictions, 'rs', label=f'Predicted ({best_model_name})', markersize=8)
    axes[0,0].set_title(f'Time Series Forecast: {operator} in {country}')
    axes[0,0].set_ylabel('Connections')
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    # Actual vs Predicted
    axes[0,1].scatter(y_test.values, best_predictions, alpha=0.7)
    min_val = min(min(y_test.values), min(best_predictions))
    max_val = max(max(y_test.values), max(best_predictions))
    axes[0,1].plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    axes[0,1].set_xlabel('Actual')
    axes[0,1].set_ylabel('Predicted')
    axes[0,1].set_title('Actual vs Predicted')
    axes[0,1].legend()
    axes[0,1].grid(True)
    
    # Model comparison
    model_names = list(results.keys())
    rmse_values = [results[name]['RMSE'] for name in model_names]
    bars = axes[1,0].bar(model_names, rmse_values, color='skyblue')
    axes[1,0].set_title('Model Comparison (RMSE)')
    axes[1,0].set_ylabel('RMSE')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Highlight best model
    best_idx = model_names.index(best_model_name)
    bars[best_idx].set_color('red')
    
    # Residuals
    residuals = y_test.values - best_predictions
    axes[1,1].scatter(best_predictions, residuals, alpha=0.7)
    axes[1,1].axhline(y=0, color='r', linestyle='--')
    axes[1,1].set_xlabel('Predicted')
    axes[1,1].set_ylabel('Residuals')
    axes[1,1].set_title('Residual Plot')
    axes[1,1].grid(True)
    
    plt.suptitle(f'Zero-Handling Model Analysis: {operator} in {country}', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    return {
        'best_model': best_model,
        'best_model_name': best_model_name,
        'results': results_df,
        'detailed_results': detailed_results,
        'zero_analysis': zero_analysis,
        'scaler': scaler,
        'feature_columns': feature_cols
    }

def forecast_with_best_model(country, operator, steps=4):
    """
    Generate forecasts using the best model for zero-handling
    """
    analysis_results = evaluate_models_for_zeros(country, operator)
    
    if analysis_results is None:
        return None
    
    print(f"\n{'='*60}")
    print(f"GENERATING FORECASTS: {operator} in {country}")
    print(f"Using: {analysis_results['best_model_name']}")
    print(f"{'='*60}\n")
    
    # Prepare full dataset for forecasting
    ts = prepare_quarterly_data_with_zeros(country, operator)
    df_features = create_features_for_ml(ts)
    
    best_model = analysis_results['best_model']
    scaler = analysis_results['scaler']
    feature_cols = analysis_results['feature_columns']
    
    # Generate future features (this is simplified - in practice you'd need more sophisticated feature engineering)
    last_values = df_features.iloc[-steps:][feature_cols]
    
    # Make predictions
    if analysis_results['best_model_name'] in ['Neural Network', 'Ridge Regression']:
        last_values_scaled = scaler.transform(last_values)
        future_predictions = best_model.predict(last_values_scaled)
    else:
        future_predictions = best_model.predict(last_values)
    
    future_predictions = np.maximum(future_predictions, 0)  # Ensure non-negative
    
    # Create future index
    last_period = ts.index[-1]
    future_periods = []
    for i in range(1, steps + 1):
        future_periods.append(last_period + i)
    
    forecast_series = pd.Series(future_predictions, index=future_periods)
    
    print("FORECAST RESULTS:")
    for period, value in forecast_series.items():
        print(f"{period}: {value:,.0f} connections")
    
    return forecast_series, analysis_results

# Example usage
if __name__ == "__main__":
    # Analyze a problematic operator with zeros
    print("Analyzing operator with zero values...")
    
    # Example with Glo (you mentioned they have zeros in 2024)
    try:
        forecast, analysis = forecast_with_best_model('Ghana', 'MTN')  # Adjust country as needed
        print(f"\nAnalysis completed successfully!")
        print(f"Zero percentage in data: {analysis['zero_analysis']['zero_percentage']:.1f}%")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Trying with Ghana MTN for demonstration...")
        
        # Fallback to your working example
        forecast, analysis = forecast_with_best_model('Ghana', 'MTN')
        
    # You can also run the evaluation separately
    # results = evaluate_models_for_zeros('Ghana', 'MTN')