import numpy as np
import pandas as pd
import requests
import time
import itertools
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings("ignore")

# --- 1. MATEMATIKAI MOTOROK (Claude specifikációi) ---
def calculate_half_life(spread):
    dz = np.diff(spread)
    z_lag = spread[:-1]
    X = sm.add_constant(z_lag)
    res = sm.OLS(dz, X).fit()
    b = res.params[1]
    if b >= 0: return np.inf
    return -np.log(2) / np.log(1 + b)

def calculate_hurst_exponent(ts, max_lag=20):
    lags = range(2, max_lag)
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0

# --- 2. ADATBÁZIS ÉPÍTŐ (Top Coinok letöltése) ---
def fetch_top_usdt_pairs(limit=30):
    print("Binance Top likvid USDT párosok lekérése...")
    url = "https://api.binance.com/api/v3/ticker/24hr"
    res = requests.get(url).json()
    
    # Csak USDT párosok, kivéve a stabilcoinokat (USDC, FDUSD stb.)
    usdt_pairs = [x for x in res if x['symbol'].endswith('USDT') and 'USD' not in x['symbol'][:-4]]
    # Rendezés 24 órás volumen alapján (kiszűrjük a shitcoin zajt)
    usdt_pairs.sort(key=lambda x: float(x['quoteVolume']), reverse=True)
    
    top_symbols = [x['symbol'] for x in usdt_pairs[:limit]]
    print(f"Top {limit} coin kiválasztva. (pl. {top_symbols[:5]}...)")
    return top_symbols

def download_bulk_data(symbols, interval='5m', limit=2000):
    print(f"Adatok letöltése {len(symbols)} coinra (ez eltarthat 1-2 percig)...")
    df_dict = {}
    
    for sym in symbols:
        url = f"https://api.binance.com/api/v3/klines"
        params = {'symbol': sym, 'interval': interval, 'limit': limit}
        try:
            res = requests.get(url, params=params).json()
            if type(res) == list and len(res) > 0:
                closes = [float(candle[4]) for candle in res]
                timestamps = [candle[0] for candle in res]
                df_dict[sym] = pd.Series(closes, index=pd.to_datetime(timestamps, unit='ms'))
        except Exception as e:
            continue
        time.sleep(0.1) # Rate limit védelem
        
    df = pd.DataFrame(df_dict).dropna(axis=1) # Eldobjuk, amiből hiányzik adat
    print(f"Sikeresen letöltve: {df.shape[1]} coin, {df.shape[0]} gyertya/coin.")
    return df

# --- 3. A KVANTITATÍV TÖLCSÉR ---
def run_screener():
    top_symbols = fetch_top_usdt_pairs(limit=40) # Top 40 coin
    df = download_bulk_data(top_symbols, interval='5m', limit=2000)
    
    # Készítünk egy normalizált (százalékos) dataframe-et a korrelációhoz
    df_pct = df.pct_change().dropna()
    
    print("\n[KAPU 1] Vektorizált Korrelációs Szűrés...")
    corr_matrix = df_pct.corr()
    
    candidate_pairs = []
    # Végigmegyünk a mátrix felső háromszögén (hogy ne legyen duplikáció)
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            coin_a = corr_matrix.columns[i]
            coin_b = corr_matrix.columns[j]
            correlation = corr_matrix.iloc[i, j]
            
            # Ha a két coin hozama legalább 75%-ban együtt mozog
            if correlation > 0.75:
                candidate_pairs.append((coin_a, coin_b, correlation))
                
    # Rendezzük korreláció szerint csökkenőbe
    candidate_pairs.sort(key=lambda x: x[2], reverse=True)
    print(f"Fennmaradt erősen korreláló párosok: {len(candidate_pairs)} db.")
    
    print("\n[KAPU 2] A 'Claude Battery' (ADF, Hurst, Half-Life) futtatása...")
    elite_pairs = []
    
    for coin_a, coin_b, corr in candidate_pairs:
        y = df[coin_a].values
        x = df[coin_b].values
        
        # 1. OLS Hedge Ratio becslés
        X = sm.add_constant(x)
        res = sm.OLS(y, X).fit()
        beta = res.params[1]
        spread = y - (beta * x)
        
        # 2. Fee Hurdle: Van elég elárazás a jutalékok (0.4%) kitermeléséhez?
        # A spread átlagos szórása el kell érje a coin árának 0.8%-át
        spread_volatility_pct = np.std(spread) / np.mean(y)
        if spread_volatility_pct < 0.008:
            continue # Túl szűk a spread, a Binance megeszi a profitot
            
        # 3. ADF Teszt (Stationarity)
        try:
            adf_p = adfuller(spread, autolag='AIC')[1]
            if adf_p > 0.05: continue
            
            # 4. Hurst Exponens (Mean Reversion)
            hurst = calculate_hurst_exponent(spread)
            if hurst > 0.40: continue
            
            # 5. OU Half-Life
            hl = calculate_half_life(spread)
            if hl < 10 or hl > 150: continue
            
            # Ha idáig eljutott, ez egy Szent Grál páros!
            elite_pairs.append({
                'Pair': f"{coin_a} / {coin_b}",
                'Corr': round(corr, 3),
                'ADF_p': round(adf_p, 4),
                'Hurst': round(hurst, 3),
                'HalfLife': round(hl, 1),
                'Volatility': f"{spread_volatility_pct*100:.2f}%"
            })
        except:
            continue
            
    # Eredmények formázása
    print("\n" + "="*70)
    print("🚀 AZ ELIT FLOTTA (KERESKEDHETŐ PÁROSOK) 🚀")
    print("="*70)
    if not elite_pairs:
        print("A mai napon NINCS olyan páros, ami átment volna az intézményi szűrőkön.")
    else:
        results_df = pd.DataFrame(elite_pairs)
        results_df.sort_values(by='Hurst', inplace=True) # Hurst (Mean Reversion) alapján rendezve
        print(results_df.to_string(index=False))
        
if __name__ == "__main__":
    run_screener()