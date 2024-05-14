# Portfolio-optimization-Python
Portfolio Optimization with Python and Streamlit
This project showcases a portfolio optimization application built using Python and Streamlit. The application leverages financial data from Yahoo Finance, along with powerful data visualization tools from Plotly, to calculate and display optimal investment portfolios.

Features:
Data Retrieval: Uses the yfinance library to download historical price data for a list of selected assets.
Portfolio Calculation: Computes daily returns, the covariance matrix, expected returns, and volatilities. Simulates random portfolio weights to identify the optimal Sharpe ratio portfolio.
Visualization: Generates various plots to visualize the expected returns, volatilities, and the efficient frontier. The plots are saved as PNG files for easy sharing and analysis.
Streamlit Integration: Provides an interactive web application interface where users can view the portfolio analysis results.
Key Components:
Data Collection: Downloads adjusted close prices for selected assets from Yahoo Finance.
Portfolio Optimization: Uses Monte Carlo simulation to generate random portfolio weights, then calculates the expected returns and volatilities for these portfolios.
Sharpe Ratio Optimization: Identifies the portfolio with the highest Sharpe ratio, indicating the best risk-adjusted return.
Interactive Visualizations: Utilizes Plotly to create and save detailed visualizations, including bar charts of expected returns and volatilities, scatter plots for risk-return analysis, and pie charts for portfolio weights.
