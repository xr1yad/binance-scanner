import os
import time
import math
import random
import requests
import pandas as pd

# =========================
# Settings (from env)
# =========================
TIMEFRAME = os.getenv("TIMEFRAME", "15m")
TOP_N = int(os.getenv("TOP_N", "60"))
USE_SWEEP = os.getenv("USE_SWEEP", "false").lower() == "true"

SMA_LEN = int(os.getenv("SMA_LEN", "200"))
PIVOT_LEN = int(os.getenv("PIVOT_LEN", "3"))
MACD_FAST = int(os.getenv("MACD_FAST", "12"))
MACD_SLOW = int(os.getenv("MACD_SLOW", "26"))
MACD_SIG  = int(os.getenv("MACD_SIG", "9"))

BOT_TOKEN = os.getenv("BOT_TOKEN", "")
CHAT_ID = os.getenv("CHAT_ID", "")

RECENT_WINDOW_SEC = int(os.getenv("RECENT_WINDOW_SEC", "240"))

VOL_MA_LEN = int(os.getenv("VOL_MA_LEN", "20"))
VOL_SPIKE_MULT = float(os.getenv("VOL_SPIKE_MULT", "2.5"))

# =========================
# OKX hosts
# =========================
OKX_HOSTS = [
    "https://www.okx.com",
    "https://my.okx.com",
    "https://app.okx.com",
]

OKX_BAR_MAP = {
    "1m": "1m",
    "3m": "3m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1H",
    "2h": "2H",
    "4h": "4H",
    "6h": "6H",
    "12h": "12H",
    "1d": "1D",
}

COMMON_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json",
}

# =========================
# Telegram
# =========================
def tg_send(text):
    if not BOT_TOKEN or not CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": CHAT_ID, "text": text}, timeout=30)

# =========================
# Helpers
# =========================
def http_get_json(path, params):
    for host in OKX_HOSTS:
        try:
            r = requests.get(host + path, params=params, headers=COMMON_HEADERS, timeout=30)
            if r.status_code == 200:
                return r.json()
        except:
            pass
    raise RuntimeError("OKX API failed")

def get_top_usdt_symbols(top_n):
    data = http_get_json("/api/v5/market/tickers", {"instType": "SPOT"})
    rows = []
    for x in data["data"]:
        if x["instId"].endswith("-USDT"):
            rows.append((x["instId"], float(x.get("volCcy24h", 0))))
    rows.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in rows[:top_n]]

def fetch_klines(symbol, tf, limit=300):
    data = http_get_json("/api/v5/market/candles", {
        "instId": symbol,
        "bar": OKX_BAR_MAP[tf],
        "limit": limit
    })["data"]

    df = pd.DataFrame(data, columns=[
        "ts","open","high","low","close","volume",
        "volCcy","volCcyQuote","confirm"
    ])
    df = df.astype({"open":float,"high":float,"low":float,"close":float,"volume":float})
    df["open_time"] = pd.to_datetime(df["ts"].astype(int), unit="ms")
    return df.sort_values("open_time").reset_index(drop=True)

# =========================
# Indicators
# =========================
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def macd(close):
    m = ema(close, MACD_FAST) - ema(close, MACD_SLOW)
    s = ema(m, MACD_SIG)
    return m, s, m - s

# =========================
# Strategy
# =========================
def evaluate(df):
    if len(df) < SMA_LEN + 10:
        return None

    df["sma"] = df["close"].rolling(SMA_LEN).mean()
    df["trend"] = df["close"] > df["sma"]

    m, s, h = macd(df["close"])
    df["macdBull"] = (m > s) & (h > 0)

    idx = df.index[df["confirm"] == "1"]
    if len(idx) == 0:
        return None
    i = idx[-1]

    if df.at[i, "trend"] and df.at[i, "macdBull"]:
        now = pd.Timestamp.utcnow()
        age = abs((now - df.at[i, "open_time"]).total_seconds())
        if age <= RECENT_WINDOW_SEC:
            return df.at[i, "close"]

    return None

# =========================
# Main
# =========================
def main():
    syms = get_top_usdt_symbols(TOP_N)
    alerts = []

    for s in syms:
        try:
            df = fetch_klines(s, TIMEFRAME)
            price = evaluate(df)
            if price:
                alerts.append(f"🟢 BUY | {s} | TF {TIMEFRAME} | Price {price}")
        except:
            pass
        time.sleep(0.2)

    if alerts:
        tg_send("📡 BUY Alerts\n\n" + "\n".join(alerts))
    else:
        print("No alerts")

if __name__ == "__main__":
    main()



