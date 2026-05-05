# 🛡️ Project Aegis: Institutional-Grade Statistical Arbitrage Engine
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)

A Python-based algorithmic trading proof-of-concept (PoC) demonstrating advanced statistical arbitrage, mean-reversion modeling, and execution friction in the cryptocurrency markets.

## 🧠 The Concept
Most retail pairs-trading algorithms fail because they rely on static OLS regression and ignore market regime changes and transaction costs. **Project Aegis** was built to solve this using institutional-grade mathematics. It models the market as a state-space using an **Adaptive Kalman Filter**, tracks synthetic cointegration baskets, and protects capital using a **CUSUM (Cumulative Sum) Watchdog** for structural break detection.

## ⚙️ Core Architecture

The engine consists of three advanced layers:

1. **The Kalman Engine (Adaptive State Estimation):** 
   Tracks the dynamic hedge ratio ($\beta$) between assets. It uses an innovation-based adaptive measurement noise ($R_t$) to slow down during market chaos.
2. **The CUSUM Watchdog (Regime Jump Detection):**
   Monitors the standardized innovations. If a fundamental structural break occurs, the CUSUM accumulator triggers a massive covariance reset, preventing the bot from aggressively trading against a permanently shifted mean.
3. **The Signal Layer (EV-Based Fee Hurdle):**
   Uses empirical quantiles (not Gaussian $\sigma$ thresholds, due to crypto's fat tails) to generate signals. Crucially, it calculates the Expected Value (EV) in dollars and implements a strict **Fee Hurdle**: it only enters a trade if the expected mean-reversion profit is at least 1.5x - 2.0x the total multi-leg transaction costs.

## 📈 Performance & 1-Year Backtest
To validate the model's robustness, the core engine was run across a 1-year period (over 105,000 candles on 5-minute intervals) through various market regimes. 

The strategy exhibited sniper-like precision, only executing trades when the EV strictly covered the double-leg exchange fees. The maximum drawdown remained virtually flat at **-0.55%**, proving the extreme safety of the CUSUM watchdog.

![1-Year Backtest Results](https://github.com/Zsoli28/Aegis/blob/main/fig/backtest_1year.png?raw=true)

## 📊 Why it is NOT running live (The Quant Reality)
Despite achieving a high simulated Sharpe Ratio and minimal drawdowns in backtesting, this project is open-sourced as a research tool rather than a live retail bot. The quantitative analysis proved the **Execution Friction Paradox**:
* Standard exchange Taker fees (0.1%) mean a full round-trip consumes a massive 0.4% of deployed capital per 2-asset pair (and 0.8% for a 4-asset basket).
* The algorithm perfectly filters out noise and finds true cointegration, but finding spreads large enough to safely cover the fee hurdle is extremely rare (yielding few trades per year).
* **Conclusion:** This strategy requires High-Frequency Trading (HFT) infrastructure and Maker (Limit) orders to be viably scaled. Naive retail market-order execution will slowly bleed capital.

## 🛠️ Files in this Repository

* `kalman_engine.py`  
  The core mathematical engine containing the `AdaptiveKalmanWithJumpDetection` class (Joseph form updates, CUSUM jump detection, and adaptive $R_t$).
* `aegisv1.2backtest.py`  
  The robust 1-year historical backtesting suite. Downloads over 100k candles, simulates exact fee deductions, and calculates institutional metrics (Sharpe, Max Drawdown).
* `aegis_screener.py`  
  The Quant Funnel. Scans the top Binance coins, runs vectorized correlation matrices, and tests for Cointegration (ADF), Mean-Reversion (Hurst Exponent), and Half-Life.
* `basket_engine.py`  
  The N-Dimensional Multivariate Kalman Filter built for synthetic basket trading (e.g., Target vs. 3 correlated assets to filter idiosyncratic risk).
* `aegis_live_paper_bot.py`  
  A daemonized live paper-trading script with API integration. Warms up the matrices, calculates real-time Z-scores, handles jump detection, and logs simulated trades to a local ledger.

---

## 🚀 How to Run

### Prerequisites
* **Python 3.9+**
* Internet connection (for Binance API data fetching)

### 1. Installation

Clone the repository and install the dependencies:
```bash
git clone [https://github.com/Zsoli28/Project-Aegis.git
cd Aegis

# Create virtual environment
python -m venv venv
# Windows: venv\Scripts\activate
# Mac/Linux: source venv/bin/activate

# Install required packages
pip install numpy pandas requests statsmodels scipy matplotlib
```

### 2. Executing the Modules

**A) Run the 1-Year Backtest:**
```bash
python aegisv1.2backtest.py
```

**B) Find Cointegrating Pairs (The Screener):**
```bash
python aegis_screener.py
```

**C) Run the Live Paper Trading Daemon:**
```bash
python aegis_live_paper_bot.py
```

---
*Disclaimer: This repository is for educational and research purposes only. It is not financial advice.*
