# cointegration_scanner.py
import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint, adfuller
from datetime import datetime, timedelta
import os
import smtplib
from email.mime.text import MIMEText
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. LISTA DE SÍMBOLOS (LÍQUIDOS)
# ==========================================
forex = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "USDCAD=X", "AUDUSD=X", "NZDUSD=X",
    "EURGBP=X", "EURJPY=X", "EURCHF=X", "EURCAD=X", "EURAUD=X", "EURNZD=X",
    "GBPJPY=X", "GBPCHF=X", "GBPCAD=X", "GBPAUD=X", "GBPNZD=X",
    "AUDJPY=X", "AUDCHF=X", "AUDCAD=X", "AUDNZD=X",
    "CADJPY=X", "CADCHF=X", "CHFJPY=X",
    "NZDJPY=X", "NZDCHF=X", "NZDCAD=X"
]
metals = ["GC=F", "SI=F", "PL=F", "PA=F", "HG=F"]
all_symbols = forex + metals

print(f"Símbolos a analizar: {len(all_symbols)}")
print(f"  Forex: {len(forex)}")
print(f"  Metales: {len(metals)}")

# ==========================================
# 2. CONFIGURACIÓN TEMPORAL
# ==========================================
days_lookback = 365
end_date = datetime.now()
start_date = end_date - timedelta(days=days_lookback)
p_value_threshold = 0.05

print(f"\nPeríodo: {start_date.date()} a {end_date.date()} ({days_lookback} días)")
print(f"Umbral p-valor: {p_value_threshold}")
print("Descargando datos históricos...")

# ==========================================
# 3. DESCARGA DE DATOS (ticker por ticker)
# ==========================================
data_dict = {}
for i, sym in enumerate(all_symbols):
    try:
        ticker = yf.Ticker(sym)
        hist = ticker.history(start=start_date, end=end_date)
        if not hist.empty and 'Close' in hist.columns:
            series = hist['Close'].dropna()
            if len(series) > 0:
                data_dict[sym] = series
                print(f"[{i+1:2d}/{len(all_symbols)}] ✓ {sym} ({len(series)} días)")
            else:
                print(f"[{i+1:2d}/{len(all_symbols)}] ✗ {sym} (sin datos)")
        else:
            print(f"[{i+1:2d}/{len(all_symbols)}] ✗ {sym}")
    except:
        print(f"[{i+1:2d}/{len(all_symbols)}] ✗ {sym} (error)")

if not data_dict:
    print("No se descargaron datos.")
    exit()

data = pd.DataFrame(data_dict)
print(f"Datos brutos: {data.shape[1]} activos, {data.shape[0]} días.")

# Eliminar activos con menos del 40% de datos
min_req = int(len(data) * 0.4)
data.dropna(axis=1, thresh=min_req, inplace=True)
print(f"Tras limpieza: {data.shape[1]} activos.")

if data.shape[1] < 2:
    print("No hay suficientes activos. Salir.")
    exit()

# ==========================================
# 4. ANÁLISIS DE COINTEGRACIÓN
# ==========================================
tickers = list(data.columns)
n = len(tickers)
print(f"\nAnalizando {n*(n-1)//2} pares...")

cointegrated = []

for i in range(n):
    for j in range(i+1, n):
        sym1 = tickers[i]
        sym2 = tickers[j]
        s1 = data[sym1].dropna()
        s2 = data[sym2].dropna()
        common = s1.index.intersection(s2.index)
        if len(common) < 50:
            continue
        y = s1.loc[common]
        x = s2.loc[common]
        try:
            score, p_value, _ = coint(y, x)
            if p_value < p_value_threshold:
                # Regresión para beta
                X = np.vstack([x, np.ones(len(x))]).T
                beta, alpha = np.linalg.lstsq(X, y, rcond=None)[0]
                spread = y - beta * x
                adf_stat, adf_p, _, _, _, _ = adfuller(spread)
                cointegrated.append({
                    'symbol_y': sym1,
                    'symbol_x': sym2,
                    'p_value': p_value,
                    'beta': beta,
                    'adf_pvalue': adf_p,
                    'spread_mean': spread.mean(),
                    'spread_std': spread.std()
                })
        except:
            continue

if not cointegrated:
    print("No se encontraron pares cointegrados.")
    best = None
else:
    cointegrated.sort(key=lambda x: x['p_value'])
    best = cointegrated[0]
    # Mostrar mejores resultados en consola
    df = pd.DataFrame(cointegrated[:10])
    print("\nMEJORES 10 PARES:")
    print(df[['symbol_y', 'symbol_x', 'p_value', 'beta']].round(6).to_string(index=False))

# ==========================================
# 5. ENVÍO DE CORREO (si hay credenciales)
# ==========================================
def send_email(subject, body):
    email_user = os.environ.get("EMAIL_USER")
    email_pass = os.environ.get("EMAIL_PASSWORD")
    if not email_user or not email_pass:
        print("No hay credenciales de correo, no se envía alerta.")
        return
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = email_user
    msg["To"] = email_user
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(email_user, email_pass)
        server.send_message(msg)
        server.quit()
        print("Correo enviado.")
    except Exception as e:
        print(f"Error de correo: {e}")

if best is not None:
    subject = f"PAR COINTEGRADO: {best['symbol_y']} / {best['symbol_x']}"
    body = f"""
    Mejor par detectado:
    Activo Y: {best['symbol_y']}
    Activo X: {best['symbol_x']}
    Hedge Ratio (β): {best['beta']:.6f}
    p-valor: {best['p_value']:.6f}
    ADF p-valor: {best['adf_pvalue']:.6f}
    Spread Medio: {best['spread_mean']:.6f}
    Spread Std: {best['spread_std']:.6f}
    
    Parámetros para EA:
    InpSymbolY = "{best['symbol_y']}";
    InpSymbolX = "{best['symbol_x']}";
    InpHedgeRatio = {best['beta']:.6f};
    """
    print("\n" + body)
    send_email(subject, body)
else:
    send_email("Cointegración - Sin pares", "No se encontraron pares cointegrados con p<0.05.")
