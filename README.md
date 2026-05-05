# 🛡️ Project Aegis: Institutional-Grade Statistical Arbitrage Engine

A Python-based algorithmic trading proof-of-concept (PoC) demonstrating advanced statistical arbitrage, mean-reversion modeling, and execution friction in the cryptocurrency markets.

## 🧠 The Concept
Most retail pairs-trading algorithms fail because they rely on static OLS regression and ignore market regime changes and transaction costs. **Project Aegis** was built to solve this using institutional-grade mathematics. It models the market as a state-space using an **Adaptive Kalman Filter**, tracks synthetic cointegration baskets, and protects capital using a **CUSUM (Cumulative Sum) Watchdog** for structural break detection.

## ⚙️ Core Architecture

The engine consists of three advanced layers:

1. **The Kalman Engine (Adaptive State Estimation):** 
   Tracks the dynamic hedge ratio ($\beta$) between assets (e.g., ARB vs. OP, or an N-dimensional basket). It uses an innovation-based adaptive measurement noise ($R_t$) to slow down during market chaos.
2. **The CUSUM Watchdog (Regime Jump Detection):**
   Monitors the standardized innovations. If a fundamental structural break occurs (e.g., a token unlock or network upgrade), the CUSUM accumulator triggers a massive covariance reset, preventing the bot from aggressively trading against a permanently shifted mean.
3. **The Signal Layer (EV-Based Fee Hurdle):**
   Uses empirical quantiles (not Gaussian $\sigma$ thresholds, due to crypto's fat tails) to generate signals. Crucially, it calculates the Expected Value (EV) in dollars and implements a strict **Fee Hurdle**: it only enters a trade if the expected mean-reversion profit is at least 1.5x - 2.0x the total multi-leg transaction costs.

## 📊 Why it is NOT running live (The Quant Reality)
Despite achieving a high simulated Sharpe Ratio (> 1.5) and minimal drawdowns in backtesting, this project is open-sourced as a research tool rather than a live bot. The quantitative analysis proved the **Basket Friction Paradox**:
* To achieve true cointegration, you need a synthetic basket (e.g., 4 assets).
* 4 assets require 8 individual market transactions for a full round-trip.
* With standard exchange Taker fees (0.1%), the execution friction consumes the theoretical Alpha.
* **Conclusion:** This strategy requires High-Frequency Trading (HFT) infrastructure and Maker (Limit) orders to be viable. Naive retail execution will slowly bleed capital.

## 🛠️ Files in this Repository
* `kalman_engine.py`: The core Adaptive Kalman Filter with the CUSUM jump detection mathematics.
* `aegis_screener.py`: The Quant Funnel. Scans the top 40 Binance coins, runs vectorized correlation matrices, and tests for Cointegration (ADF), Mean-Reversion (Hurst Exponent), and Half-Life.
* `basket_engine.py`: The N-Dimensional Multivariate Kalman Filter for synthetic basket trading.
* `aegis_live_paper_bot.py`: A daemonized live paper-trading script with API integration and local ledger logging.

---

## 🚀 How to Run

### Prerequisites
* **Python 3.9** or higher
* Internet connection (for Binance API data fetching)
* No Binance API keys are required for the Screener or the Paper Trading bot (they use public market data endpoints).

### 1. Installation

Clone the repository and navigate into the directory:
```bash
git clone [https://github.com/Zsoli28/Aegis.git
cd Aegis
```

Create a virtual environment to keep dependencies clean (Recommended):
```bash
# On Windows:
python -m venv venv
venv\Scripts\activate

# On macOS and Linux:
python3 -m venv venv
source venv/bin/activate
```

Install the required quantitative and data science libraries:
```bash
pip install -r requirements.txt
```

### 2. Executing the Modules

The project is modular. You can run different parts of the engine depending on your goals:

#### A) Find Cointegrating Pairs (The Quant Funnel)
To scan the top Binance assets and find pairs that pass the strict institutional criteria (ADF, Hurst, Half-Life, Fee Hurdle):
```bash
python aegis_screener.py
```
*Output:* A table of elite, tradeable pairs or a message indicating that no pairs passed the strict fee hurdle today.

#### B) Run the Live Paper Trading Daemon
To start the live bot that tracks ARB/OP (or any configured pair), calculates real-time Z-scores, handles CUSUM jump detection, and logs simulated trades:
```bash
python aegis_live_paper_bot.py
```
*Output:* The bot will warm up using the last 48 hours of data, then print real-time updates every 5 minutes. Trades are logged locally to `aegis_trades.csv`.

#### C) Run Historical Backtests
If you want to test the N-Dimensional Basket strategy or the standard pair strategy on historical data:
```bash
python basket_engine.py
# or
python kalman_engine.py
```
*Output:* Terminal logs of the total simulated Gross/Net PnL, Sharpe Ratio, Max Drawdown, and a matplotlib chart of the equity curve.

---
*Disclaimer: This repository is for educational and research purposes only. It is not financial advice.*
