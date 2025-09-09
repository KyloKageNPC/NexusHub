import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime
import sqlite3
import json
import os
from statsmodels.tsa.statespace.sarimax import SARIMAXResults

# Initialize Dash app
app = dash.Dash(__name__, title='Telecom Forecast Monitor')
server = app.server

# Database setup
def init_db():
    conn = sqlite3.connect('monitoring.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS forecasts
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  country TEXT,
                  operator TEXT,
                  timestamp DATETIME,
                  metrics TEXT,
                  forecast_data TEXT)''')
    conn.commit()
    conn.close()

init_db()

# Layout components
def create_metric_card(title, value, delta=None):
    return html.Div(
        className="metric-card",
        children=[
            html.H3(title),
            html.H2(value),
            html.Small(f"{delta} from last run" if delta else "")
        ],
        style={
            'border': '1px solid #ddd',
            'padding': '15px',
            'margin': '10px',
            'borderRadius': '5px',
            'backgroundColor': '#f9f9f9'
        }
    )

app.layout = html.Div([
    html.H1("Telecom Connections Forecast Monitoring", style={'textAlign': 'center'}),
    
    # Filters
    html.Div([
        dcc.Dropdown(
            id='country-selector',
            options=[{'label': c, 'value': c} for c in ['Ghana', 'Nigeria', 'Kenya']],
            value='Ghana',
            style={'width': '30%', 'display': 'inline-block'}
        ),
        dcc.Dropdown(
            id='operator-selector',
            options=[{'label': o, 'value': o} for o in ['MTN', 'Airtel', 'Safaricom']],
            value='MTN',
            style={'width': '30%', 'display': 'inline-block', 'marginLeft': '10px'}
        ),
        dcc.DatePickerRange(
            id='date-range',
            min_date_allowed=datetime(2020, 1, 1),
            max_date_allowed=datetime.today(),
            start_date=datetime(2023, 1, 1),
            end_date=datetime.today()
        )
    ], style={'padding': '20px'}),
    
    # Metrics Row
    html.Div(id='metrics-row', className="row", style={'display': 'flex'}),
    
    # Main Charts
    dcc.Graph(id='forecast-chart'),
    dcc.Graph(id='error-trend-chart'),
    
    # Data Tables
    html.H3("Recent Forecasts"),
    dash_table.DataTable(
        id='forecast-table',
        columns=[
            {'name': 'Country', 'id': 'country'},
            {'name': 'Operator', 'id': 'operator'},
            {'name': 'Date', 'id': 'timestamp'},
            {'name': 'RMSE', 'id': 'rmse'},
            {'name': 'MAPE', 'id': 'mape'},
            {'name': 'Status', 'id': 'status'}
        ],
        style_table={'overflowX': 'auto'}
    ),
    
    # Hidden div to store intermediate values
    html.Div(id='intermediate-value', style={'display': 'none'})
])

# Callbacks
@app.callback(
    Output('intermediate-value', 'children'),
    [Input('country-selector', 'value'),
     Input('operator-selector', 'value')]
)
def load_data(country, operator):
    conn = sqlite3.connect('monitoring.db')
    query = f"""SELECT * FROM forecasts 
                WHERE country='{country}' AND operator='{operator}'
                ORDER BY timestamp DESC LIMIT 100"""
    df = pd.read_sql(query, conn)
    conn.close()
    return df.to_json(date_format='iso', orient='split')

@app.callback(
    [Output('metrics-row', 'children'),
     Output('forecast-chart', 'figure'),
     Output('error-trend-chart', 'figure'),
     Output('forecast-table', 'data')],
    [Input('intermediate-value', 'children')]
)
def update_dashboard(json_data):
    df = pd.read_json(json_data, orient='split')
    
    if df.empty:
        return [], {}, {}, []
    
    # Process metrics
    latest = df.iloc[0]
    metrics = json.loads(latest['metrics'])
    
    # Metric cards
    cards = [
        create_metric_card("Current RMSE", f"{metrics['rmse']:,.2f}"),
        create_metric_card("Mean Absolute % Error", f"{metrics['mape']:.2%}"),
        create_metric_card(
            "Last Forecast Date",
            latest['timestamp'].strftime('%Y-%m-%d') if hasattr(latest['timestamp'], 'strftime') else str(latest['timestamp'])[:10]
        ),
        create_metric_card("Data Points", f"{len(json.loads(latest['forecast_data']))}")
    ]
    
    # Forecast chart
    forecast_data = json.loads(latest['forecast_data'])
    forecast_df = pd.DataFrame(forecast_data)
    forecast_df['date'] = pd.to_datetime(forecast_df['date'])
    
    forecast_fig = px.line(forecast_df, x='date', y=['actual', 'forecast'],
                          title=f"Actual vs Forecast: {latest['country']} - {latest['operator']}")
    forecast_fig.update_layout(hovermode="x unified")
    
    # Error trend chart
    error_df = df.copy()
    error_df['metrics'] = error_df['metrics'].apply(json.loads)
    error_df['rmse'] = error_df['metrics'].apply(lambda x: x['rmse'])
    error_df['mape'] = error_df['metrics'].apply(lambda x: x['mape'])
    error_df['timestamp'] = pd.to_datetime(error_df['timestamp'])
    
    error_fig = px.line(error_df, x='timestamp', y=['rmse', 'mape'],
                       title="Error Metrics Over Time",
                       labels={'value': 'Metric Value', 'variable': 'Metric'})
    
    # Table data
    table_data = df.head(10).copy()
    table_data['metrics'] = table_data['metrics'].apply(json.loads)
    table_data['rmse'] = table_data['metrics'].apply(lambda x: f"{x['rmse']:,.2f}")
    table_data['mape'] = table_data['metrics'].apply(lambda x: f"{x['mape']:.2%}")
    table_data['status'] = table_data['metrics'].apply(
        lambda x: "✅ Good" if x['mape'] < 0.1 else "⚠️ Warning" if x['mape'] < 0.2 else "❌ Alert")
    
    return cards, forecast_fig, error_fig, table_data.to_dict('records')

# Helper function to log new forecasts
def log_forecast(country, operator, metrics, forecast_data):
    conn = sqlite3.connect('monitoring.db')
    c = conn.cursor()
    c.execute("""INSERT INTO forecasts 
                 (country, operator, timestamp, metrics, forecast_data) 
                 VALUES (?, ?, ?, ?, ?)""",
              (country, operator, datetime.now().isoformat(),
               json.dumps(metrics), json.dumps(forecast_data)))
    conn.commit()
    conn.close()

if __name__ == '__main__':
    app.run(debug=True)