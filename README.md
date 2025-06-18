# Quantitative Market Risk Dashboard

This project is a **Streamlit app** that provides a quantitative market risk analysis framework for a portfolio of assets. It calculates various risk metrics such as Value at Risk (VaR), Conditional VaR (CVaR), performs backtesting, and stress testing with historical market events.

## Features

- Download historical stock data from Yahoo Finance
- Calculate daily log returns
- Compute risk metrics:
  - Historical VaR
  - Parametric VaR (Gaussian)
  - Monte Carlo VaR
  - Historical CVaR
- Backtesting with Kupiec test
- Stress testing with predefined market crash scenarios
- Interactive dashboard with visualization of return distributions and risk metrics

## Technologies

- Python 3.x
- Streamlit for interactive web app
- yfinance for financial data
- numpy, pandas for data manipulation
- matplotlib, seaborn for visualization
- scipy for statistical tests

