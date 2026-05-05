import numpy as np
import pandas as pd
import requests
import time
import datetime
import csv
import os
from collections import deque

print("""
=================================================
    PROJECT AEGIS - LIVE PAPER TRADING NODE
    Motor: Adaptive Kalman Filter + CUSUM
    Páros: ARB/USDT vs OP/USDT
    Idősík: 5m | Fee Hurdle: Aktív
=================================================
""")

# --- 1. A KALMAN MOTOR ---
class AdaptiveKalmanWithJumpDetection:
    def __init__(self, q=1e-5, r_init=1.0, h=5.0, k=0.5, r_window=50, n_blackout=15, z_window=30, p_reset_scale=1000.0):
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
            
        return beta_hat, spread, is_trading_halted, list(self.spread_history), jump_detected

# --- 2. API ÉS SEGÉDFÜGGVÉNYEK ---
def get_klines(symbol, limit=1):
    """Letölti az utolsó lezárt gyertyát."""
    url = "https://api.binance.com/api/v3/klines"
    params = {'symbol': symbol, 'interval': '5m', 'limit': limit}
    for attempt in range(3): # Hibatűrés
        try:
            res = requests.get(url, params=params, timeout=10)
            data = res.json()
            if data and len(data) > 0:
                return data
        except Exception as e:
            print(f"[!] API Hiba ({symbol}): {e}. Újrapróbálkozás...")
            time.sleep(5)
    return None

def log_trade(action, price_y, price_x, beta, spread, pnl=0.0, fee=0.0):
    """Beírja a kötést a naplófájlba (Ledger)."""
    file_exists = os.path.isfile('aegis_trades.csv')
    with open('aegis_trades.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Timestamp', 'Action', 'ARB_Price', 'OP_Price', 'Beta', 'Spread', 'Net_PnL', 'Fee'])
        writer.writerow([datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), action, price_y, price_x, beta, spread, pnl, fee])

# --- 3. INICIALIZÁLÁS (WARM-UP) ---
print("[*] Kalman motor bemelegítése az elmúlt 48 óra adataival...")
kf = AdaptiveKalmanWithJumpDetection()

hist_arb = get_klines('ARBUSDT', limit=500)
hist_op = get_klines('OPUSDT', limit=500)

if not hist_arb or not hist_op:
    print("[!] Hiba a kezdő adatok letöltésekor. Kilépés.")
    exit()

for i in range(len(hist_arb) - 1): # Az utolsó gyertya még nincs lezárva!
    y_hist = float(hist_arb[i][4])
    x_hist = float(hist_op[i][4])
    kf.update(y_hist, x_hist)

print("[+] Bemelegítés kész. A mátrixok készen állnak az éles adatokra.")

# --- 4. ÉLES PAPER TRADING CIKLUS ---
position = 0 # 1: Long, -1: Short
entry_y, entry_x, entry_beta = 0, 0, 0
FEE_RATE = 0.001
TRADE_SIZE = 1000
paper_pnl = 0.0

print("[*] Bot élesítve. Várakozás a következő 5 perces gyertya zárására...\n")

while True:
    try:
        # Kiszámoljuk, mikor van a következő 5 perces zárás
        now = datetime.datetime.now()
        minutes_to_wait = 5 - (now.minute % 5)
        next_run = now + datetime.timedelta(minutes=minutes_to_wait)
        next_run = next_run.replace(second=2, microsecond=0) # +2 mp biztonsági rátartás a Binance API miatt
        
        sleep_seconds = (next_run - now).total_seconds()
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
            
        # Gyertyák letöltése
        arb_data = get_klines('ARBUSDT', limit=2)
        op_data = get_klines('OPUSDT', limit=2)
        
        # Az utolsó LEZÁRT gyertya a limit=2 listában a [0] indexű (a [1] a jelenleg futó)
        y = float(arb_data[0][4])
        x = float(op_data[0][4])
        
        # Kalman Frissítés
        beta, spread, halted, spread_hist, jumped = kf.update(y, x)
        
        # Képernyőfrissítés
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status = "HALTED" if halted else "ACTIVE"
        if jumped: print(f"\n[!!!] {timestamp} - STRUKTURÁLIS TÖRÉS DETEKTÁLVA. CUSUM Reset. Blackout aktív.")
        print(f"[{timestamp}] ARB: {y:.4f} | OP: {x:.4f} | Spread: {spread:.4f} | Beta: {beta:.4f} | Status: {status}")
        
        # --- Szignál és Kereskedés Logika ---
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
                        fee_paid = round_trip_fee / 2
                        paper_pnl -= fee_paid
                        print(f"   [+] NYITÁS: SHORT SPREAD | Várható EV: ${expected_dollar_profit:.2f}")
                        log_trade("OPEN_SHORT", y, x, beta, spread, fee=-fee_paid)
                        
                    elif spread < lower_q:
                        position = 1
                        entry_y, entry_x, entry_beta = y, x, beta
                        fee_paid = round_trip_fee / 2
                        paper_pnl -= fee_paid
                        print(f"   [+] NYITÁS: LONG SPREAD | Várható EV: ${expected_dollar_profit:.2f}")
                        log_trade("OPEN_LONG", y, x, beta, spread, fee=-fee_paid)
                        
            elif position == 1 and spread >= mean_q:
                qty_y_trade = TRADE_SIZE / entry_y
                qty_x_trade = qty_y_trade * entry_beta
                gross_pnl = (y - entry_y) * qty_y_trade - (x - entry_x) * qty_x_trade
                fee_paid = round_trip_fee / 2
                net_pnl = gross_pnl - fee_paid
                paper_pnl += net_pnl
                position = 0
                print(f"   [-] ZÁRÁS: LONG SPREAD | Nettó PnL: ${net_pnl:.2f} | Összesített: ${paper_pnl:.2f}")
                log_trade("CLOSE_LONG", y, x, beta, spread, pnl=net_pnl, fee=-fee_paid)
                
            elif position == -1 and spread <= mean_q:
                qty_y_trade = TRADE_SIZE / entry_y
                qty_x_trade = qty_y_trade * entry_beta
                gross_pnl = (entry_y - y) * qty_y_trade - (entry_x - x) * qty_x_trade
                fee_paid = round_trip_fee / 2
                net_pnl = gross_pnl - fee_paid
                paper_pnl += net_pnl
                position = 0
                print(f"   [-] ZÁRÁS: SHORT SPREAD | Nettó PnL: ${net_pnl:.2f} | Összesített: ${paper_pnl:.2f}")
                log_trade("CLOSE_SHORT", y, x, beta, spread, pnl=net_pnl, fee=-fee_paid)

    except Exception as e:
        print(f"[!] Váratlan hiba a fő ciklusban: {e}")
        time.sleep(10) # Hiba esetén várunk kicsit, mielőtt újra próbálkoznánk