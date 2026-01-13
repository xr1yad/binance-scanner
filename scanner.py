import os
import time
import math
import random
import requests
import pandas as pd

# =========================
# Settings (from env)
# =========================
TIMEFRAME = os.getenv("TIMEFRAME", "1h")
TOP_N = int(os.getenv("TOP_N", "60"))
USE_SWEEP = os.getenv("USE_SWEEP", "false").lower() == "true"

SMA_LEN = int(os.getenv("SMA_LEN", "200"))
PIVOT_LEN = int(os.getenv("PIVOT_LEN", "3"))
MACD_FAST = int(os.getenv("MACD_FAST", "12"))
MACD_SLOW = int(os.getenv("MACD_SLOW", "26"))
MACD_SIG  = int(os.getenv("MACD_SIG", "9"))

ENABLE_SELL = os.getenv("ENABLE_SELL", "false").lower() == "true"
ENABLE_EXIT = os.getenv("ENABLE_EXIT", "true").lower() == "true"
ENABLE_EXIT_MACD_WEAK = os.getenv("ENABLE_EXIT_MACD_WEAK", "true").lower() == "true"
HIST_WEAK_BARS = int(os.getenv("HIST_WEAK_BARS", "2"))
ENABLE_EXIT_STRUCT = os.getenv("ENABLE_EXIT_STRUCT", "true").lower() == "true"

BOT_TOKEN = os.getenv("BOT_TOKEN", "")
CHAT_ID = os.getenv("CHAT_ID", "")

# Bybit hosts (نجرّب أكثر من واحد)
BYBIT_HOSTS = [
    "https://api.bybit.com",
    "https://api.bytick.com",   # بديل أحيانًا يشتغل إذا الرئيسي مقفل
]

BYBIT_INTERVAL_MAP = {
    "1m": "1",
    "3m": "3",
    "5m": "5",
    "15m": "15",
    "30m": "30",
    "1h": "60",
    "2h": "120",
    "4h": "240",
    "6h": "360",
    "12h": "720",
    "1d": "D",
    "1w": "W",
    "1M": "M",
}

# Headers لتجاوز بعض WAF
COMMON_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}

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
# HTTP helper (multi-host)
# =========================
def http_get_json_multi_host(path: str, params: dict):
    """
    Try multiple Bybit hosts until one returns JSON with HTTP 200.
    If all fail, return the last (status, text) for debugging.
    """
    last_err = None
    for base in BYBIT_HOSTS:
        url = f"{base}{path}"
        try:
            # jitter صغير
            time.sleep(0.05 + random.random() * 0.05)
            r = requests.get(url, params=params, headers=COMMON_HEADERS, timeout=30)

            # إذا مو 200 أو رجع HTML
            ct = (r.headers.get("content-type") or "").lower()
            if r.status_code != 200:
                last_err = (base, r.status_code, r.text[:300])
                continue
            if "text/html" in ct:
                last_err = (base, r.status_code, r.text[:300])
                continue

            # حاول JSON
            try:
                data = r.json()
            except Exception:
                last_err = (base, r.status_code, r.text[:300])
                continue

            return base, r.status_code, data

        except Exception as e:
            last_err = (base, -1, str(e)[:300])
            continue

    # كلهم فشلوا
    base, status, body = last_err if last_err else ("", -1, "unknown")
    raise RuntimeError(f"All Bybit hosts failed. LastHost={base} Status={status} Body={body}")

# =========================
# Bybit functions
# =========================
def get_top_usdt_symbols(top_n: int):
    """
    /v5/market/tickers?category=spot
    """
    host, status, data = http_get_json_multi_host("/v5/market/tickers", {"category": "spot"})
    if not isinstance(data, dict) or data.get("retCode") != 0:
        raise RuntimeError(f"Bybit tickers error. Host={host} Status={status} Data={str(data)[:300]}")

    items = (data.get("result") or {}).get("list", [])
    if not isinstance(items, list) or not items:
        raise RuntimeError(f"Bybit tickers empty. Host={host} Status={status} Data={str(data)[:300]}")

    rows = []
    for x in items:
        if not isinstance(x, dict):
            continue
        sym = x.get("symbol", "")
        if not sym.endswith("USDT"):
            continue
        try:
            turnover = float(x.get("turnover24h", "0"))
        except Exception:
            turnover = 0.0
        rows.append((sym, turnover))

    rows.sort(key=lambda z: z[1], reverse=True)
    syms = [s for s, _ in rows[:top_n]]
    print(f"Tickers OK from host: {host}. Symbols: {len(syms)}")
    return syms

def fetch_klines(symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
    interval = BYBIT_INTERVAL_MAP.get(timeframe)
    if interval is None:
        raise ValueError(f"Unsupported TIMEFRAME: {timeframe}")

    host, status, data = http_get_json_multi_host("/v5/market/kline", {
        "category": "spot",
        "symbol": symbol,
        "interval": interval,
        "limit": str(limit),
    })

    if not isinstance(data, dict) or data.get("retCode") != 0:
        raise RuntimeError(f"Bybit kline error {symbol}. Host={host} Status={status} Data={str(data)[:200]}")

    rows = (data.get("result") or {}).get("list", [])
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"Bybit kline empty {symbol}. Host={host} Status={status} Data={str(data)[:200]}")

    # rows newest-first -> sort ascending by startTime
    parsed = []
    for r in rows:
        if not isinstance(r, list) or len(r) < 6:
            continue
        parsed.append(r)

    df = pd.DataFrame(parsed, columns=["start_ms", "open", "high", "low", "close", "volume", "turnover"])
    df["start_ms"] = pd.to_numeric(df["start_ms"], errors="coerce")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna().sort_values("start_ms").reset_index(drop=True)
    df["open_time"] = pd.to_datetime(df["start_ms"], unit="ms")

    if interval.isdigit():
        minutes = int(interval)
        df["close_time"] = df["open_time"] + pd.to_timedelta(minutes, unit="m")
    else:
        df["close_time"] = df["open_time"]

    return df[["open_time", "open", "high", "low", "close", "volume", "close_time"]]

# =========================
# Indicators / Strategy
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
    lows = df["low"].values
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
    lastSwingLow = math.nan
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

        c = float(df.at[i, "close"])
        lo = float(df.at[i, "low"])
        hi = float(df.at[i, "high"])

        bBull = (not math.isnan(lastSwingHigh)) and (c > lastSwingHigh)
        bBear = (not math.isnan(lastSwingLow)) and (c < lastSwingLow)

        cBull = (structureTrend == -1) and bBull
        cBear = (structureTrend == 1) and bBear

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

    df["buyTrigger"] = df["buySignal"] & (~df["buySignal"].shift(1).fillna(False))
    df["sellTrigger"] = df["sellSignal"] & (~df["sellSignal"].shift(1).fillna(False))

    df["histWeak"] = False
    for i in range(HIST_WEAK_BARS, len(df)):
        ok = True
        for k in range(HIST_WEAK_BARS):
            if not (df.at[i - k, "hist"] < df.at[i - k - 1, "hist"]):
                ok = False
                break
        df.at[i, "histWeak"] = ok

    if ENABLE_EXIT_MACD_WEAK:
        df["exitMacdWeak"] = df["trendBull"] & df["histWeak"]
    else:
        df["exitMacdWeak"] = False

    last_low_series = []
    last_low = math.nan
    for i in range(len(df)):
        if not math.isnan(df.at[i, "pl"]):
            last_low = float(df.at[i, "pl"])
        last_low_series.append(last_low)
    df["lastSwingLow"] = last_low_series

    if ENABLE_EXIT_STRUCT:
        df["exitStructure"] = (~pd.isna(df["lastSwingLow"])) & (df["close"] < df["lastSwingLow"])
    else:
        df["exitStructure"] = False

    df["exitLongSignal"] = df["exitMacdWeak"] | df["exitStructure"]
    df["exitLongTrigger"] = df["exitLongSignal"] & (~df["exitLongSignal"].shift(1).fillna(False))

    idx = len(df) - 2
    if idx < 0:
        return None

    return {
        "buy": bool(df.at[idx, "buyTrigger"]),
        "sell": bool(df.at[idx, "sellTrigger"]),
        "exit": bool(df.at[idx, "exitLongTrigger"]),
        "price": float(df.at[idx, "close"]),
        "time": str(df.at[idx, "close_time"]),
    }

def main():
    try:
        symbols = get_top_usdt_symbols(TOP_N)
    except Exception as e:
        # لا نوقف البوت: نرسل لك سبب المشكلة
        err = f"❌ Bybit API blocked from runner.\n{e}"
        print(err)
        tg_send(err)
        return

    alerts = []
    print(f"Scanning {len(symbols)} symbols on {TIMEFRAME} (Bybit) ...")

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

        time.sleep(0.25)

    if alerts:
        msg = "📡 1H Scanner Alerts (Bybit)\n" + "\n".join(alerts[:30])
        tg_send(msg)
        print("Sent:", len(alerts))
    else:
        print("No alerts.")

if __name__ == "__main__":
    main()
