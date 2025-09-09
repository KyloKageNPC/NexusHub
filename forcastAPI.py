# app.py
from flask import Flask, request, jsonify
from model8 import prepare_quarterly_data_enhanced, apply_variance_reduction_techniques
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
import numpy as np
import pandas as pd

app = Flask(__name__)

@app.route('/forecast', methods=['POST'])
def forecast():
    try:
        data = request.get_json()
        country = data.get('country')
        operator = data.get('operator')
        zero_method = data.get('zero_method', 'smart_interpolation')
        variance_method = data.get('variance_method', 'hybrid')
        forecast_steps = int(data.get('steps', 20))  # 5 years by default

        # Prepare and transform data
        ts_cleaned, zero_info = prepare_quarterly_data_enhanced(country, operator, zero_method)
        ts_transformed, transform_info = apply_variance_reduction_techniques(ts_cleaned, variance_method)

        # Fit model
        auto_model = auto_arima(
            ts_transformed,
            start_p=0, d=1, start_q=0,
            start_P=0, D=1, start_Q=0,
            max_p=3, max_q=3, max_P=2, max_Q=2,
            m=4, seasonal=True, stepwise=True,
            suppress_warnings=True, error_action='ignore', trace=False
        )

        model = SARIMAX(ts_transformed, order=auto_model.order, seasonal_order=auto_model.seasonal_order)
        model_fit = model.fit(disp=False)

        forecast_result = model_fit.get_forecast(steps=forecast_steps)
        forecast_transformed = forecast_result.predicted_mean

        # Inverse transform
        inverse_func = transform_info['inverse_func']
        forecast_original = inverse_func(forecast_transformed)

        # Sanitize
        forecast_original = np.maximum(forecast_original, 0.1).tolist()
        forecast_index = [str(ts_cleaned.index[-1] + i + 1) for i in range(forecast_steps)]

        return jsonify({
            'forecast': forecast_original,
            'periods': forecast_index,
            'info': {
                'zero_handling': zero_method,
                'variance_method': variance_method,
                'steps': forecast_steps
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
