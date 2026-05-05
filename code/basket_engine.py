import numpy as np
import pandas as pd
import requests
import time
from collections import deque
import matplotlib.pyplot as plt

def fetch_data(symbol, interval='5m', limit=5000):
    print(f"Letöltés: {symbol}...")
    url = "https://api.binance.com/api/v3/klines"
    all_data = []
    end_time = int(time.time() * 1000)
    
    while len(all_data) < limit:
        params = {'symbol': symbol, 'interval': interval, 'limit': min(1000, limit - len(all_data)), 'endTime': end_time}
        try:
            res = requests.get(url, params=params).json()
            if not res or type(res) != list: break
            all_data = res + all_data
            end_time = res[0][0] - 1
            time.sleep(0.1)
        except Exception as e:
            print(f"Hiba: {e}")
            break
            
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'v', 'ct', 'qav', 'nt', 'tbb', 'tbq', 'i'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df[['close']].astype(float).rename(columns={'close': symbol})

# --- A TÖBBVÁLTOZÓS MOTOR (N-Dimensional Kalman Filter) ---
class MultivariateKalmanWithJumpDetection:
    def __init__(self, n_features, q=1e-5, r_init=1.0, h=5.0, k=0.5, r_window=100, n_blackout=15, z_window=30, p_reset_scale=1000.0):
        # n_features: A kosárban lévő eszközök száma (pl. OP, MATIC, ETH = 3)
        self.n = n_features + 1 # +1 az Alpha (metszet) miatt
        
        # Állapotvektor mátrix: [Alpha, Beta1, Beta2, ... BetaN]^T
        self.x = np.zeros((self.n, 1))
        self.P = np.eye(self.n) * 1.0
        self.Q = np.eye(self.n) * q
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

    def update(self, y, x_vals):
        """
        y: Függő változó (Célpont, pl. ARB árfolyam)
        x_vals: Lista vagy tömb a független változókból (Kosár, pl. [OP_ár, MATIC_ár, ETH_ár])
        """
        # H (Megfigyelés) mátrix: [1.0, x_1, x_2, ..., x_n]
        H = np.array([[1.0] + list(x_vals)]) 
        
        # 1. PREDICT
        x_pred = self.x.copy()
        P_pred = self.P + self.Q
        
        # 2. INNOVATION
        y_pred = (H @ x_pred)[0, 0]
        nu = y - y_pred
        self.innovations.append(nu)
        
        if len(self.innovations) >= self.r_window:
            self.warmup_complete = True
            self.R = float(np.var(self.innovations)) + 1e-5
            
        S = (H @ P_pred @ H.T)[0, 0] + self.R
        S = max(S, 1e-10)
        
        # 3. CUSUM JUMP DETECTION
        jump_detected = False
        if self.warmup_complete:
            std_nu = nu / np.sqrt(S)
            self.cusum_pos = max(0.0, self.cusum_pos + std_nu - self.k)
            self.cusum_neg = max(0.0, self.cusum_neg - std_nu - self.k)
            jump_detected = (self.cusum_pos > self.h) or (self.cusum_neg > self.h)
            
        if jump_detected:
            self.P = np.eye(self.n) * self.p_reset_scale
            P_pred = self.P + self.Q
            S = (H @ P_pred @ H.T)[0, 0] + self.R
            S = max(S, 1e-10)
            self.cusum_pos = 0.0
            self.cusum_neg = 0.0
            self.blackout_counter = self.n_blackout
            self.spread_history.clear()
            
        self.jump_history.append(jump_detected)
        
        # 4. KALMAN UPDATE (Joseph Form)
        K = (P_pred @ H.T) / S
        self.x = x_pred + K * nu
        
        I = np.eye(self.n)
        IKH = I - (K @ H)
        self.P = IKH @ P_pred @ IKH.T + (K @ K.T) * self.R
        
        # 5. SIGNAL (SPREAD) CALCULATION
        # A spread = y - (Alpha + Beta1*X1 + Beta2*X2 + ...)
        current_alpha = self.x[0, 0]
        current_betas = self.x[1:, 0]
        spread = y - (current_alpha + np.dot(current_betas, x_vals))
        
        self.spread_history.append(spread)
        
        is_trading_halted = self.blackout_counter > 0
        if is_trading_halted:
            self.blackout_counter -= 1
            
        return current_betas, spread, is_trading_halted, list(self.spread_history)

# --- TESZT KÖRNYEZET ---
if __name__ == "__main__":
    print(">>> MULTIVARIATE BASKET TRADING INITIALIZED <<<")
    
    # 1. Adatgyűjtés
    TARGET = 'ARBUSDT'
    BASKET = ['OPUSDT', 'MATICUSDT', 'ETHUSDT']
    LIMIT = 10000 # ~34 nap 5 perces gyertyákon
    
    df_target = fetch_data(TARGET, limit=LIMIT)
    df_basket = [fetch_data(sym, limit=LIMIT) for sym in BASKET]
    
    df = pd.concat([df_target] + df_basket, axis=1).ffill().dropna()
    print(f"Közös adatsor összeállítva: {len(df)} gyertya.")
    
    # 2. Inicializáljuk a Többváltozós Motort (N=3 a 3 független coin miatt)
    kf = MultivariateKalmanWithJumpDetection(n_features=len(BASKET), q=1e-5, h=5.0, k=0.5, z_window=30)
    
    pnl_history = []
    cumulative_pnl = 0
    trade_count = 0
    position = 0 # 1: Long Spread, -1: Short Spread
    
    entry_y = 0
    entry_x_vals = []
    entry_betas = []
    
    FEE_RATE = 0.0002
    TRADE_SIZE = 1000
    
    for i in range(len(df)):
        y = df[TARGET].iloc[i]
        x_vals = df[BASKET].iloc[i].values
        
        betas, spread, halted, spread_hist = kf.update(y, x_vals)
        
        if len(spread_hist) >= kf.z_window + 1 and not halted:
            clean_hist = spread_hist[:-1]
            upper_q = np.percentile(clean_hist, 95)
            lower_q = np.percentile(clean_hist, 5)
            mean_q = np.mean(clean_hist)
            
            qty_y = TRADE_SIZE / y
            expected_dollar_profit = abs(spread - mean_q) * qty_y
            
            # FIGYELEM: A Fee nagyobb, mert 4 coinnal kereskedünk egyszerre!
            # Round trip = (1 Célpont + 3 Kosár coin) * Nyitás/Zárás = 8 tranzakció
            num_assets = 1 + len(BASKET)
            round_trip_fee = TRADE_SIZE * FEE_RATE * num_assets * 2
            MIN_EV = round_trip_fee * 1.5 # Itt 1.5x szorzót kérünk
            
            if position == 0 and expected_dollar_profit > MIN_EV:
                if spread > upper_q:
                    position = -1
                    entry_y = y
                    entry_x_vals = x_vals
                    entry_betas = betas
                    cumulative_pnl -= round_trip_fee / 2
                elif spread < lower_q:
                    position = 1
                    entry_y = y
                    entry_x_vals = x_vals
                    entry_betas = betas
                    cumulative_pnl -= round_trip_fee / 2
                    
            elif position != 0:
                # Kiszállás (Mean Reversion)
                if (position == 1 and spread >= mean_q) or (position == -1 and spread <= mean_q):
                    # Profit számítása egy szintetikus kosáron
                    qty_y_trade = TRADE_SIZE / entry_y
                    
                    # Célpont láb profitja
                    if position == 1: pnl_y = (y - entry_y) * qty_y_trade
                    else:             pnl_y = (entry_y - y) * qty_y_trade
                    
                    # Kosár lábak profitja (Ellentétes irányú)
                    pnl_basket = 0
                    for j in range(len(BASKET)):
                        qty_x_trade = qty_y_trade * entry_betas[j]
                        if position == 1: pnl_basket -= (x_vals[j] - entry_x_vals[j]) * qty_x_trade
                        else:             pnl_basket -= (entry_x_vals[j] - x_vals[j]) * qty_x_trade
                        
                    cumulative_pnl += (pnl_y + pnl_basket)
                    cumulative_pnl -= round_trip_fee / 2
                    position = 0
                    trade_count += 1
                    
        pnl_history.append(cumulative_pnl)
        
    df['PnL'] = pnl_history
    print(f"\nBefejezett Basket Trade-ek: {trade_count} db")
    print(f"Nettó Profit: ${cumulative_pnl:.2f}")
    print(f"Jutalékok kifizetve: ~${trade_count * round_trip_fee:.2f}")