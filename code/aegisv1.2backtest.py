import numpy as np
import pandas as pd
import requests
import time
from collections import deque
import matplotlib.pyplot as plt

# ==============================================================================
# 1. ROBUSZTUS ADATBESZERZÉS – HOSSZABB IDŐSZAKRA OPTIMALIZÁLVA
#    A limit paraméterrel most már könnyen kérhetsz 1+ év adatot is.
# ==============================================================================
def fetch_binance_klines_paginated(symbol, interval='5m', limit=105120):
    """
    Limit = 105120 gyertya kb. 1 év 5 perces adat.
    A Binance 1000-esével adja, ezért ez akár 106 API hívást is jelenthet.
    """
    print(f"Letöltés: {symbol} (~{limit} gyertya, ez eltarthat pár percig)...")
    url = "https://api.binance.com/api/v3/klines"
    all_data = []
    end_time = int(time.time() * 1000)
    requests_count = 0

    while len(all_data) < limit:
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': min(1000, limit - len(all_data)),
            'endTime': end_time
        }
        try:
            res = requests.get(url, params=params)
            requests_count += 1
            if res.status_code != 200:
                print(f"  API hiba (status {res.status_code}), várok 5 mp-et...")
                time.sleep(5)
                continue
            data = res.json()
            if not isinstance(data, list) or not data:
                print(f"  Nincs több adat, leállás {len(all_data)} sorral.")
                break
            all_data = data + all_data
            end_time = data[0][0] - 1
            # Haladásjelző minden 5000 gyertya után
            if len(all_data) % 5000 == 0:
                print(f"  {symbol}: {len(all_data)}/{limit} letöltve ({requests_count} kérés)")
            time.sleep(0.15)  # Kíméljük az API-t
        except Exception as e:
            print(f"  Hiba: {e}, újrapróbálkozás...")
            time.sleep(5)

    columns = ['timestamp', 'open', 'high', 'low', 'close', 'v',
               'ct', 'qav', 'nt', 'tbb', 'tbq', 'i']
    df = pd.DataFrame(all_data, columns=columns)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[['close']].astype(float).rename(columns={'close': symbol})
    print(f"  {symbol} kész: {len(df)} sor.")
    return df

def build_portfolio(limit=105120):
    """
    Letölti az ARB és OP párt, összefésüli és tisztítja.
    Alapértelmezett limit 1 év (~105120 db 5 perces).
    """
    df_a = fetch_binance_klines_paginated('ARBUSDT', '5m', limit)
    df_b = fetch_binance_klines_paginated('OPUSDT', '5m', limit)
    portfolio = pd.merge(df_a, df_b, left_index=True, right_index=True, how='outer')
    portfolio.ffill(inplace=True)
    portfolio.dropna(inplace=True)
    print(f"Portfólió összeállítva: {len(portfolio)} közös adatsor.\n")
    return portfolio

# ==============================================================================
# 2. KALMAN MOTOR – VÁLTOZATLAN
# ==============================================================================
class AdaptiveKalmanWithJumpDetection:
    def __init__(self, q=1e-5, r_init=1.0, h=5.0, k=0.5, r_window=100,
                 n_blackout=13, z_window=30, p_reset_scale=1000.0):
        self.x = np.zeros(2)
        self.P = np.eye(2)
        self.Q = np.eye(2) * q
        self.R = r_init
        self.r_window = r_window
        self.innovations = deque(maxlen=r_window)
        self.warmup_complete = False
        self.cusum_pos = 0.0
        self.cusum_neg = 0.0
        self.h = h
        self.k = k
        self.blackout_counter = 0
        self.n_blackout = n_blackout
        self.p_reset_scale = p_reset_scale
        self.jump_history = []
        self.z_window = z_window
        self.spread_history = deque(maxlen=z_window * 3)

    def update(self, y, x_val):
        H = np.array([1.0, x_val])
        x_pred = self.x.copy()
        P_pred = self.P + self.Q

        y_pred = H @ x_pred
        nu = y - y_pred
        self.innovations.append(nu)

        if len(self.innovations) >= self.r_window:
            self.warmup_complete = True
            self.R = float(np.var(self.innovations)) + 1e-5

        S = float(H @ P_pred @ H.T) + self.R
        S = max(S, 1e-10)

        jump_detected = False
        if self.warmup_complete:
            std_nu = nu / np.sqrt(S)
            self.cusum_pos = max(0.0, self.cusum_pos + std_nu - self.k)
            self.cusum_neg = max(0.0, self.cusum_neg - std_nu - self.k)
            jump_detected = (self.cusum_pos > self.h) or (self.cusum_neg > self.h)

        if jump_detected:
            self.P = np.eye(2) * self.p_reset_scale
            P_pred = self.P + self.Q
            S = float(H @ P_pred @ H.T) + self.R
            S = max(S, 1e-10)
            self.cusum_pos = 0.0
            self.cusum_neg = 0.0
            self.blackout_counter = self.n_blackout
            self.spread_history.clear()

        self.jump_history.append(jump_detected)

        K = P_pred @ H.T / S
        self.x = x_pred + K * nu
        I = np.eye(2)
        IKH = I - np.outer(K, H)
        self.P = IKH @ P_pred @ IKH.T + np.outer(K, K) * self.R

        beta_hat = self.x[1]
        spread = y - (self.x[0] + beta_hat * x_val)
        self.spread_history.append(spread)

        is_trading_halted = self.blackout_counter > 0
        if is_trading_halted:
            self.blackout_counter -= 1

        return beta_hat, spread, is_trading_halted, list(self.spread_history)

# ==============================================================================
# 3. BACKTEST – V1.1 LOGIKA (HOSSZABB ADATON)
# ==============================================================================
print("\n>>> PROJECT AEGIS V1.2 – HOSSZÍTOTT BACKTEST <<<")
# 1 év alapértelmezés, de átírhatod pl. 200000-re 2 évhez
df = build_portfolio(limit=105120)

kf = AdaptiveKalmanWithJumpDetection(q=1e-5, h=5.0, k=0.5, r_window=50,
                                     n_blackout=15, z_window=30)

position = 0
entry_y, entry_x, entry_beta = 0, 0, 0
cumulative_pnl = 0
total_fees_paid = 0
capital = 1000
pnl_history = []
trade_count = 0

FEE_RATE = 0.001
TRADE_SIZE = 1000

print(f"Adatok betöltve: {len(df)} sor. Backtest indítása...\n")

for i in range(len(df)):
    y = df['ARBUSDT'].iloc[i]
    x = df['OPUSDT'].iloc[i]

    beta, spread, halted, spread_hist = kf.update(y, x)

    if len(spread_hist) >= kf.z_window + 1 and not halted:
        clean_history = spread_hist[:-1]

        upper_q = np.percentile(clean_history, 95)
        lower_q = np.percentile(clean_history, 5)
        mean_q = np.mean(clean_history)

        qty_y = TRADE_SIZE / y
        expected_dollar_profit = abs(spread - mean_q) * qty_y
        round_trip_fee = TRADE_SIZE * FEE_RATE * 4
        MIN_EV = round_trip_fee * 2.0

        if position == 0:
            if expected_dollar_profit > MIN_EV:
                if spread > upper_q:
                    position = -1
                    entry_y, entry_x, entry_beta = y, x, beta
                    fee = round_trip_fee / 2
                    cumulative_pnl -= fee
                    total_fees_paid += fee

                elif spread < lower_q:
                    position = 1
                    entry_y, entry_x, entry_beta = y, x, beta
                    fee = round_trip_fee / 2
                    cumulative_pnl -= fee
                    total_fees_paid += fee

        elif position == 1 and spread >= mean_q:
            qty_y_trade = TRADE_SIZE / entry_y
            qty_x_trade = qty_y_trade * entry_beta
            pnl = (y - entry_y) * qty_y_trade - (x - entry_x) * qty_x_trade
            cumulative_pnl += pnl
            fee = round_trip_fee / 2
            cumulative_pnl -= fee
            total_fees_paid += fee
            position = 0
            trade_count += 1

        elif position == -1 and spread <= mean_q:
            qty_y_trade = TRADE_SIZE / entry_y
            qty_x_trade = qty_y_trade * entry_beta
            pnl = (entry_y - y) * qty_y_trade - (entry_x - x) * qty_x_trade
            cumulative_pnl += pnl
            fee = round_trip_fee / 2
            cumulative_pnl -= fee
            total_fees_paid += fee
            position = 0
            trade_count += 1

    pnl_history.append(cumulative_pnl)

df['PnL'] = pnl_history

# ==============================================================================
# 4. INTÉZMÉNYI METRIKÁK
# ==============================================================================
print("=== VÉGEREDMÉNY (v1.2 – Hosszú Backtest) ===")
days_simulated = len(df) * 5 / 60 / 24

returns = pd.Series(pnl_history).diff().dropna()
sharpe = 0.0
if returns.std() > 0:
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252 * 288)

max_drawdown = (pd.Series(pnl_history) - pd.Series(pnl_history).cummax()).min()

print(f"Vizsgált időszak: {len(df)} gyertya (~{days_simulated:.1f} nap, ~{days_simulated/365:.2f} év)")
print(f"Kezdőtőke: ${capital}")
print(f"Befejezett Trade-ek: {trade_count} db")
print(f"-----------------------------------")
print(f"Bruttó Profit (Gross PnL): ${cumulative_pnl + total_fees_paid:,.2f}")
print(f"Kifizetett Jutalékok: ${total_fees_paid:,.2f}")
print(f"Valós Nettó Profit (True Net PnL): ${cumulative_pnl:,.2f}")
print(f"-----------------------------------")
print(f"Annualized Sharpe Ratio: {sharpe:.3f}")
print(f"Max Drawdown: ${max_drawdown:,.2f}")
print(f"Évesített hozam (becsült): {((cumulative_pnl/capital)/(days_simulated/365))*100:.2f}%")

# Rajzolás
plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [2, 1]})

ax1.plot(df.index, df['ARBUSDT'], label='ARB', color='cyan', alpha=0.7)
ax1.set_title(f'ARB Árfolyam ({len(df)} gyertya)')
ax1.legend()

ax2.plot(df.index, df['PnL'], label='Cumulative PnL ($)', color='lime', linewidth=2)
ax2.set_title(f'Kereskedési Profit (Sharpe: {sharpe:.2f}, Trade: {trade_count})')
ax2.axhline(0, color='white', linestyle='--', alpha=0.5)
ax2.legend()

plt.tight_layout()
plt.show()