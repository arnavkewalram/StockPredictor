import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime, timedelta

class StockDashboard:
    def __init__(self, data: pd.DataFrame, predictions: Dict[str, np.ndarray], 
                 metrics: Dict[str, Dict[str, float]]):
        self.data = data
        self.predictions = predictions
        self.metrics = metrics
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        
    def create_layout(self):
        """Create the dashboard layout"""
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col(html.H1("Stock Price Predictor Dashboard", className="text-center my-4"))
            ]),
            
            dbc.Row([
                dbc.Col([
                    html.H3("Stock Price Prediction"),
                    dcc.Graph(id='price-prediction-graph')
                ], width=8),
                
                dbc.Col([
                    html.H3("Model Performance"),
                    dcc.Graph(id='model-performance-graph')
                ], width=4)
            ]),
            
            dbc.Row([
                dbc.Col([
                    html.H3("Technical Indicators"),
                    dcc.Graph(id='technical-indicators-graph')
                ], width=6),
                
                dbc.Col([
                    html.H3("News Sentiment Analysis"),
                    dcc.Graph(id='sentiment-graph')
                ], width=6)
            ]),
            
            dbc.Row([
                dbc.Col([
                    html.H3("Risk Analysis"),
                    dcc.Graph(id='risk-analysis-graph')
                ], width=12)
            ]),
            
            dbc.Row([
                dbc.Col([
                    html.H3("Model Metrics"),
                    html.Div(id='model-metrics-table')
                ], width=12)
            ])
        ])
        
        # Register callbacks
        self._register_callbacks()
        
    def _register_callbacks(self):
        """Register dashboard callbacks"""
        @self.app.callback(
            Output('price-prediction-graph', 'figure'),
            [Input('price-prediction-graph', 'relayoutData')]
        )
        def update_price_prediction(_):
            fig = go.Figure()
            
            # Add historical data
            fig.add_trace(go.Scatter(
                x=self.data.index,
                y=self.data['Close'],
                name='Historical Price',
                line=dict(color='blue')
            ))
            
            # Add predictions for each model
            for model_name, pred in self.predictions.items():
                future_dates = pd.date_range(
                    start=self.data.index[-1],
                    periods=len(pred),
                    freq='D'
                )
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=pred.flatten(),
                    name=f'{model_name} Prediction',
                    line=dict(dash='dash')
                ))
            
            fig.update_layout(
                title='Stock Price Predictions',
                xaxis_title='Date',
                yaxis_title='Price',
                hovermode='x unified'
            )
            
            return fig
        
        @self.app.callback(
            Output('model-performance-graph', 'figure'),
            [Input('model-performance-graph', 'relayoutData')]
        )
        def update_model_performance(_):
            metrics_df = pd.DataFrame(self.metrics).T
            fig = px.bar(
                metrics_df,
                title='Model Performance Metrics',
                labels={'value': 'Score', 'index': 'Model'},
                barmode='group'
            )
            
            return fig
        
        @self.app.callback(
            Output('technical-indicators-graph', 'figure'),
            [Input('technical-indicators-graph', 'relayoutData')]
        )
        def update_technical_indicators(_):
            fig = go.Figure()
            
            # Add RSI
            fig.add_trace(go.Scatter(
                x=self.data.index,
                y=self.data['RSI'],
                name='RSI',
                line=dict(color='purple')
            ))
            
            # Add MACD
            fig.add_trace(go.Scatter(
                x=self.data.index,
                y=self.data['MACD'],
                name='MACD',
                line=dict(color='orange')
            ))
            
            # Add Bollinger Bands
            fig.add_trace(go.Scatter(
                x=self.data.index,
                y=self.data['BB_Upper'],
                name='BB Upper',
                line=dict(color='gray', dash='dash')
            ))
            fig.add_trace(go.Scatter(
                x=self.data.index,
                y=self.data['BB_Lower'],
                name='BB Lower',
                line=dict(color='gray', dash='dash')
            ))
            
            fig.update_layout(
                title='Technical Indicators',
                xaxis_title='Date',
                yaxis_title='Value',
                hovermode='x unified'
            )
            
            return fig
        
        @self.app.callback(
            Output('risk-analysis-graph', 'figure'),
            [Input('risk-analysis-graph', 'relayoutData')]
        )
        def update_risk_analysis(_):
            # Calculate volatility
            returns = self.data['Close'].pct_change()
            volatility = returns.rolling(window=20).std() * np.sqrt(252)
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=self.data.index,
                y=volatility,
                name='Volatility',
                line=dict(color='red')
            ))
            
            fig.update_layout(
                title='Risk Analysis',
                xaxis_title='Date',
                yaxis_title='Volatility',
                hovermode='x unified'
            )
            
            return fig
        
        @self.app.callback(
            Output('model-metrics-table', 'children'),
            [Input('model-metrics-table', 'children')]
        )
        def update_model_metrics(_):
            metrics_df = pd.DataFrame(self.metrics).T
            return dbc.Table.from_dataframe(
                metrics_df,
                striped=True,
                bordered=True,
                hover=True
            )
    
    def run(self, debug: bool = True):
        """Run the dashboard"""
        self.create_layout()
        self.app.run_server(debug=debug) 