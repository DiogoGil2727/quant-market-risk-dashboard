# Quantitative Market Risk Framework

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, chi2
import streamlit as st

# ------------------------
# INITIAL CONFIGURATION
# ------------------------
assets = ["AAPL", "MSFT", "GOOGL"]
weights = np.array([0.4, 0.3, 0.3])
start_date = "2020-01-01"
end_date = "2024-12-31"
confidence_level = 0.95
portfolio_value = 1_000_000

# ------------------------
# DATA COLLECTION
# ------------------------
def get_data(assets, start, end):
    raw_data = yf.download(assets, start=start, end=end, auto_adjust=True)
    if isinstance(raw_data.columns, pd.MultiIndex):
        adj_close = raw_data["Close"]
    else:
        adj_close = raw_data.to_frame(name=assets[0])
    return adj_close.dropna()

# ------------------------
# RETURNS CALCULATION
# ------------------------
def compute_returns(data):
    return np.log(data / data.shift(1)).dropna()

# ------------------------
# RISK METRICS
# ------------------------
def historical_var(returns, cl=0.95):
    return -np.percentile(returns, (1 - cl) * 100)

def parametric_var(returns, cl=0.95):
    mu = returns.mean()
    sigma = returns.std()
    z = norm.ppf(1 - cl)
    return -(mu + z * sigma)

def monte_carlo_var(mu, sigma, value, cl=0.95, sims=10000):
    sim_returns = np.random.normal(mu, sigma, sims)
    losses = value * sim_returns
    return -np.percentile(losses, (1 - cl) * 100)

def cvar_historical(returns, cl=0.95):
    var = historical_var(returns, cl)
    losses = -returns[returns <= -var]
    return losses.mean()

# ------------------------
# BACKTESTING (Kupiec Test)
# ------------------------
def kupiec_test(returns, var, cl=0.95):
    breaches = returns < -var
    x = breaches.sum()
    n = len(returns)
    p_hat = x / n
    p = 1 - cl
    lr = -2 * (np.log(((1 - p) ** (n - x) * p ** x)) - np.log(((1 - p_hat) ** (n - x) * p_hat ** x)))
    p_value = 1 - chi2.cdf(lr, df=1)
    return x, lr, p_value

# ------------------------
# STRESS TESTING
# ------------------------
def stress_test(prices, drop_percent):
    stressed_prices = prices * (1 - drop_percent)
    returns = compute_returns(stressed_prices)
    portfolio_returns = returns.dot(weights[:len(returns.columns)])
    return portfolio_returns

# Specific stress events
stress_events = {
    "COVID-19 Crash (Mar 2020)": 0.30,
    "2008 Financial Crisis": 0.50,
    "Ukraine Invasion": 0.15
}

# ------------------------
# VISUALIZATIONS
# ------------------------
def plot_distribution(returns):
    plt.figure(figsize=(10, 4))
    sns.histplot(returns, bins=50, kde=True, color="blue")
    plt.title("Portfolio Returns Distribution")
    plt.xlabel("Daily Return")
    plt.ylabel("Frequency")
    plt.grid(True)

# ------------------------
# STREAMLIT DASHBOARD
# ------------------------
def main():
    st.title("Quantitative Market Risk Dashboard")
    data = get_data(assets, start_date, end_date)
    returns = compute_returns(data)
    portfolio_returns = returns.dot(weights[:len(returns.columns)])

    st.subheader("Risk Metrics")
    hist_var = historical_var(portfolio_returns, confidence_level)
    param_var = parametric_var(portfolio_returns, confidence_level)
    mc_var = monte_carlo_var(portfolio_returns.mean(), portfolio_returns.std(), portfolio_value, confidence_level)
    cvar = cvar_historical(portfolio_returns, confidence_level)

    st.write(f"**Historical VaR (95%)**: {hist_var:.4f}")
    st.write(f"**Parametric VaR (95%)**: {param_var:.4f}")
    st.write(f"**Monte Carlo VaR (95%)**: {mc_var:.2f} USD")
    st.write(f"**Historical CVaR (95%)**: {cvar:.4f}")

    st.subheader("VaR Backtesting")
    breaches, lr_stat, p_val = kupiec_test(portfolio_returns, hist_var, confidence_level)
    st.write(f"Number of breaches: {breaches}")
    st.write(f"LR statistic: {lr_stat:.4f}")
    st.write(f"p-value: {p_val:.4f}")

    st.subheader("Stress Testing")
    for name, drop in stress_events.items():
        stressed = stress_test(data, drop)
        var_stress = historical_var(stressed, confidence_level)
        st.write(f"{name} ({int(drop*100)}% drop) → VaR: {var_stress:.4f}")

    st.subheader("Returns Distribution")
    plot_distribution(portfolio_returns)
    st.pyplot(plt.gcf())  
    plt.clf()            

    st.markdown("""
    ## 📘 Documentation
    
    **Data:** Obtained from Yahoo Finance via `yfinance`, automatically adjusted.

    **Model:**
    - Calculates VaR using historical, parametric, and Monte Carlo simulation methods.
    - CVaR as the average loss beyond the VaR threshold.
    - Backtesting using the Kupiec test.
    - Stress testing with real historical events (COVID-19, 2008 crisis).

    **Interpretation:**
    - VaR shows the maximum expected loss for a given confidence level.
    - CVaR highlights risk in extreme tail scenarios.

    **Limitations:**
    - Parametric models assume normality.
    - Tail events and systemic risk may not be fully captured.
    - Simplified models for educational purposes.
    """)

if __name__ == "__main__":
    main()
