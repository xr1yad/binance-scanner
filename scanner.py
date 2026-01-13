import os
import time
import math
import requests
import pandas as pd

# =========================
# Settings (from env)
# =========================
TIMEFRAME = os.getenv("TIMEFRAME", "1h")      # فريم الساعة
TOP_N = int(os.getenv("TOP_N", "60"))         # عدد العملات
USE_SWEEP = os.getenv("USE_SWEEP", "true").lower() == "true"

SMA_LEN = int(os.getenv("SMA_LEN", "200"))
PIVOT_LEN = int(os.getenv("PIVOT_LEN", "3"))
MACD_FAST = int(os.getenv("MACD_FAST", "12"))
MACD_SLOW = int(os.getenv("MACD_SLOW", "26"))
MACD_SIG  = int(os.getenv("MACD_SIG", "9"))

ENABLE_SELL = os.getenv("ENABLE_SELL", "true").lower() == "true"
ENABLE_EXIT = os.getenv("ENABLE_EXIT", "true").lower() == "true"
ENABLE_EXIT_MACD_WEAK = os.getenv("ENABLE_EXIT_MACD_WEAK", "true").lower() == "true"
HIST_WEAK_BARS = int(os.getenv("HIST_WEAK_BARS", "2"))
ENABLE_EXIT_STRUCT = os.getenv("ENABLE_EXIT_STRUCT", "true").lower() == "true"

BOT_TOKEN = os.getenv("BOT_TOKEN", "")
CHAT_ID = os.getenv("CHAT_ID", "")

BINANCE_BASE = "https://api.binance.com"


# =========================
# Telegram
# =========================
def tg_send(text: str):
    if not BOT_TOKEN or not CHAT_ID:
        print("Missing BOT_TOKEN / CHAT_ID")
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text}
    r = requests.post(url, json=payload, timeout=30)
    r.raise_for_status()


# =========================
# Binance helpers
# =========================
def get_top_usdt_symbols(top_n: int):
    """
    Safely fetch top USDT symbols by 24h quote volume.
    If Binance returns an unexpected response (string/error), we fallback to a safe list
    so the bot never crashes.
    """
    url = f"{BINANCE_BASE}/api/v3/ticker/24hr"
    r = requests.get(url, timeout=30)

    # Try JSON
    try:
        data = r.json()
    except Exception:
        print("Binance response not JSON. Status:", r.status_code)
        print("Body:", r.text[:500])
        data = None

    # Must be a list of dicts
    if not isinstance(data, list):
        print("Unexpected Binance response type:", type(data))
        print("Status:", r.status_code)
        try:
            print("Body:", str(data)[:500])
        except Exception:
            pass

        # Fallback list (strong, liquid pairs)
        fallback = [
            "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","ADAUSDT","DOGEUSDT","TRXUSDT",
            "LINKUSDT","AVAXUSDT","DOTUSDT","MATICUSDT","ATOMUSDT","OPUSDT","ARBUSDT","NEARUSDT",
            "LTCUSDT","BCHUSDT","INJUSDT","APTUSDT","SUIUSDT","SHIBUSDT","PEPEUSDT"
        ]
        return fallback[:max(10, min(top_n, len(fallback)))]

    rows = []
    for x in data:
        if not isinstance(x, dict):
            continue
        sym = x.get("symbol", "")
        if not sym.endswith("USDT"):
            continue

        # exclude leveraged tokens
        if any(t in sym for t in ["UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT"]):
            continue

        try:
            qv = float(x.get("quoteVolume", "0"))
        except Exception:
            qv = 0.0

        rows.append((sym, qv))

    rows.sort(key=lambda z: z[1], reverse=True)
    return [s for s, _ in rows[:top_n]]


def fetch_klines(symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
    url = f"{BINANCE_BASE}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(url, params=params, timeout=30)

    try:
        data = r.json()
    except Exception:
        raise RuntimeError(f"Binance klines not JSON for {symbol}. Status={r.status_code} Body={r.text[:200]}")

    if not isinstance(data, list) or len(data) == 0:
        raise RuntimeError(f"Binance klines invalid for {symbol}. Status={r.status_code} Data={str(data)[:200]}")

    df = pd.DataFrame(data, columns=[
        "open_time","open","high","low","close","volume","close_time",
        "qav","num_trades","tbbav","tbqav","ignore"
    ])

    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

    return df.dropna().reset_index(drop=True)


# =========================
# Indicators
# =========================
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def macd(close: pd.Series, fast: int, slow: int, sig: int):
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, sig)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def compute_pivots(df: pd.DataFrame, n: int):
    highs = df["high"].values
    lows  = df["low"].values
    ph = [math.nan] * len(df)
    pl = [math.nan] * len(df)

    for i in range(n, len(df) - n):
        window_h = highs[i - n:i + n + 1]
        window_l = lows[i - n:i + n + 1]
        if highs[i] == max(window_h):
            ph[i] = highs[i]
        if lows[i] == min(window_l):
            pl[i] = lows[i]

    return pd.Series(ph), pd.Series(pl)


# =========================
# Strategy logic
# =========================
def evaluate_signals(df: pd.DataFrame):
    if len(df) < max(SMA_LEN + 10, 300):
        return None

    df = df.copy()
    df["sma200"] = df["close"].rolling(SMA_LEN).mean()

    macd_line, signal_line, hist = macd(df["close"], MACD_FAST, MACD_SLOW, MACD_SIG)
    df["macd"] = macd_line
    df["signal"] = signal_line
    df["hist"] = hist

    df["macdBull"] = (df["macd"] > df["signal"]) & (df["hist"] > 0)
    df["macdBear"] = (df["macd"] < df["signal"]) & (df["hist"] < 0)

    df["trendBull"] = df["close"] > df["sma200"]
    df["trendBear"] = df["close"] < df["sma200"]

    ph, pl = compute_pivots(df, PIVOT_LEN)
    df["ph"] = ph
    df["pl"] = pl

    lastSwingHigh = math.nan
    lastSwingLow  = math.nan
    structureTrend = 0

    bosBull = [False] * len(df)
    bosBear = [False] * len(df)
    chochBull = [False] * len(df)
    chochBear = [False] * len(df)
    sweepBull = [False] * len(df)
    sweepBear = [False] * len(df)

    for i in range(len(df)):
        if not math.isnan(df.at[i, "ph"]):
            lastSwingHigh = float(df.at[i, "ph"])
        if not math.isnan(df.at[i, "pl"]):
            lastSwingLow = float(df.at[i, "pl"])

        c  = float(df.at[i, "close"])
        lo = float(df.at[i, "low"])
        hi = float(df.at[i, "high"])

        bBull = (not math.isnan(lastSwingHigh)) and (c > lastSwingHigh)
        bBear = (not math.isnan(lastSwingLow))  and (c < lastSwingLow)

        cBull = (structureTrend == -1) and bBull
        cBear = (structureTrend ==  1) and bBear

        bosBull[i] = bBull
        bosBear[i] = bBear
        chochBull[i] = cBull
        chochBear[i] = cBear

        if bBull:
            structureTrend = 1
        if bBear:
            structureTrend = -1

        if not math.isnan(lastSwingLow):
            sweepBull[i] = (lo < lastSwingLow) and (c > lastSwingLow)
        if not math.isnan(lastSwingHigh):
            sweepBear[i] = (hi > lastSwingHigh) and (c < lastSwingHigh)

    df["smcBullEvent"] = (pd.Series(bosBull) | pd.Series(chochBull))
    df["smcBearEvent"] = (pd.Series(bosBear) | pd.Series(chochBear))

    if USE_SWEEP:
        df["buySignal"]  = df["trendBull"] & df["macdBull"] & df["smcBullEvent"] & pd.Series(sweepBull)
        df["sellSignal"] = df["trendBear"] & df["macdBear"] & df["smcBearEvent"] & pd.Series(sweepBear)
    else:
        df["buySignal"]  = df["trendBull"] & df["macdBull"] & df["smcBullEvent"]
        df["sellSignal"] = df["trendBear"] & df["macdBear"] & df["smcBearEvent"]

    df["buyTrigger"]  = df["buySignal"]  & (~df["buySignal"].shift(1).fillna(False))
    df["sellTrigger"] = df["sellSignal"] & (~df["sellSignal"].shift(1).fillna(False))

    # Histogram weakening (exit)
    df["histWeak"] = False
    for i in range(HIST_WEAK_BARS, len(df)):
        ok = True
        for k in range(HIST_WEAK_BARS):
            if not (df.at[i - k, "hist"] < df.at[i - k - 1, "hist"]):
                ok = False
                break
        df.at[i, "histWeak"] = ok

    df["exitMacdWeak"] = ENABLE_EXIT_MACD_WEAK and True
    df["exitMacdWeak"] = (df["trendBull"] & df["histWeak"]) if ENABLE_EXIT_MACD_WEAK else False

    # Last swing low series for structure exit
    last_low_series = []
    last_low = math.nan
    for i in range(len(df)):
        if not math.isnan(df.at[i, "pl"]):
            last_low = float(df.at[i, "pl"])
        last_low_series.append(last_low)

    df["lastSwingLow"] = last_low_series
    df["exitStructure"] = (ENABLE_EXIT_STRUCT and True)
    df["exitStructure"] = ((~pd.isna(df["lastSwingLow"])) & (df["close"] < df["lastSwingLow"])) if ENABLE_EXIT_STRUCT else False

    df["exitLongSignal"] = df["exitMacdWeak"] | df["exitStructure"]
    df["exitLongTrigger"] = df["exitLongSignal"] & (~df["exitLongSignal"].shift(1).fillna(False))

    # Use last CLOSED candle
    idx = len(df) - 2
    if idx < 0:
        return None

    return {
        "buy":  bool(df.at[idx, "buyTrigger"]),
        "sell": bool(df.at[idx, "sellTrigger"]),
        "exit": bool(df.at[idx, "exitLongTrigger"]),
        "price": float(df.at[idx, "close"]),
        "time": str(df.at[idx, "close_time"]),
    }


# =========================
# Main
# =========================
def main():
    symbols = get_top_usdt_symbols(TOP_N)
    alerts = []

    print(f"Scanning {len(symbols)} symbols on {TIMEFRAME} ...")

    for sym in symbols:
        try:
            df = fetch_klines(sym, TIMEFRAME, limit=500)
            sig = evaluate_signals(df)
            if not sig:
                continue

            if sig["buy"]:
                alerts.append(f"🟢 BUY | {sym} | TF {TIMEFRAME} | Price {sig['price']:.8g} | Close {sig['time']}")

            if ENABLE_SELL and sig["sell"]:
                alerts.append(f"🔴 SELL | {sym} | TF {TIMEFRAME} | Price {sig['price']:.8g} | Close {sig['time']}")

            if ENABLE_EXIT and sig["exit"]:
                alerts.append(f"🟡 EXIT | {sym} | TF {TIMEFRAME} | Price {sig['price']:.8g} | Close {sig['time']}")

        except Exception as e:
            print(f"{sym} error: {e}")

        # reduce API pressure
        time.sleep(0.2)

    if alerts:
        msg = "📡 1H Scanner Alerts\n" + "\n".join(alerts[:30])
        tg_send(msg)
        print("Sent:", len(alerts))
    else:
        print("No alerts.")


if __name__ == "__main__":
    main()

