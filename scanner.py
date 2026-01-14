import os
import time
import math
import random

import requests
import pandas as pd

# =========================
# Settings (from env)
# =========================
TIMEFRAME = os.getenv("TIMEFRAME", "1h")  # examples: 15m, 1h, 4h, 1d
TOP_N = int(os.getenv("TOP_N", "60"))
USE_SWEEP = os.getenv("USE_SWEEP", "false").lower() == "true"

SMA_LEN = int(os.getenv("SMA_LEN", "200"))
PIVOT_LEN = int(os.getenv("PIVOT_LEN", "3"))

MACD_FAST = int(os.getenv("MACD_FAST", "12"))
MACD_SLOW = int(os.getenv("MACD_SLOW", "26"))
MACD_SIG = int(os.getenv("MACD_SIG", "9"))

BOT_TOKEN = os.getenv("BOT_TOKEN", "")
CHAT_ID = os.getenv("CHAT_ID", "")

# يرسل فقط إذا كانت الشمعة "أغلقت للتو" (لتجنب تكرار التنبيهات مع GitHub Actions)
RECENT_WINDOW_SEC = int(os.getenv("RECENT_WINDOW_SEC", "240"))  # 4 دقائق افتراضي

# Pump / Volume spike settings
VOL_MA_LEN = int(os.getenv("VOL_MA_LEN", "20"))        # متوسط الفوليوم
VOL_SPIKE_MULT = float(os.getenv("VOL_SPIKE_MULT", "2.5"))  # كم مرة أعلى من المتوسط

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
    "1w": "1W",
    "1M": "1M",
}

COMMON_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
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
    last_err = None
    for base in OKX_HOSTS:
        url = f"{base}{path}"
        try:
            time.sleep(0.05 + random.random() * 0.05)
            r = requests.get(url, params=params, headers=COMMON_HEADERS, timeout=30)
            ct = (r.headers.get("content-type") or "").lower()

            if r.status_code != 200 or "text/html" in ct:
                last_err = (base, r.status_code, r.text[:300])
                continue

            try:
                data = r.json()
            except Exception:
                last_err = (base, r.status_code, r.text[:300])
                continue

            return base, r.status_code, data

        except Exception as e:
            last_err = (base, -1, str(e)[:300])
            continue

    base, status, body = last_err if last_err else ("", -1, "unknown")
    raise RuntimeError(f"All OKX hosts failed. LastHost={base} Status={status} Body={body}")

# =========================
# OKX functions
# =========================
def get_top_usdt_symbols(top_n: int):
    host, status, data = http_get_json_multi_host("/api/v5/market/tickers", {"instType": "SPOT"})
    if not isinstance(data, dict) or data.get("code") != "0":
        raise RuntimeError(f"OKX tickers error. Host={host} Status={status} Data={str(data)[:300]}")

    items = data.get("data", [])
    if not isinstance(items, list) or not items:
        raise RuntimeError(f"OKX tickers empty. Host={host} Status={status} Data={str(data)[:300]}")

    rows = []
    for x in items:
        if not isinstance(x, dict):
            continue
        inst = x.get("instId", "")
        if not inst.endswith("-USDT"):
            continue
        try:
            turnover = float(x.get("volCcy24h", "0"))
        except Exception:
            turnover = 0.0
        rows.append((inst, turnover))

    rows.sort(key=lambda z: z[1], reverse=True)
    syms = [s for s, _ in rows[:top_n]]
    print(f"Tickers OK from host: {host}. Symbols: {len(syms)}")
    return syms


def _infer_close_time(open_time: pd.Timestamp, bar: str) -> pd.Timestamp:
    try:
        if bar.endswith("m"):
            return open_time + pd.to_timedelta(int(bar[:-1]), unit="m")
        if bar.endswith("H"):
            return open_time + pd.to_timedelta(int(bar[:-1]), unit="h")
        if bar.endswith("D"):
            return open_time + pd.to_timedelta(int(bar[:-1]), unit="d")
        if bar.endswith("W"):
            return open_time + pd.to_timedelta(int(bar[:-1]), unit="w")
        if bar.endswith("M"):
            return open_time
    except Exception:
        pass
    return open_time


def fetch_klines(symbol: str, timeframe: str, limit: int = 300) -> pd.DataFrame:
    bar = OKX_BAR_MAP.get(timeframe)
    if bar is None:
        raise ValueError(f"Unsupported TIMEFRAME: {timeframe}. Supported: {list(OKX_BAR_MAP.keys())}")

    host, status, data = http_get_json_multi_host(
        "/api/v5/market/candles",
        {"instId": symbol, "bar": bar, "limit": str(min(int(limit), 300))},
    )

    if not isinstance(data, dict) or data.get("code") != "0":
        raise RuntimeError(f"OKX candles error {symbol}. Host={host} Status={status} Data={str(data)[:200]}")

    rows = data.get("data", [])
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"OKX candles empty {symbol}. Host={host} Status={status} Data={str(data)[:200]}")

    parsed = []
    for r in rows:
        if not isinstance(r, list) or len(r) < 6:
            continue
        parsed.append(r)

    df = pd.DataFrame(
        parsed,
        columns=[
            "start_ms", "open", "high", "low", "close", "volume",
            "volCcy", "volCcyQuote", "confirm"
        ][:len(parsed[0])]
    )

    df["start_ms"] = pd.to_numeric(df["start_ms"], errors="coerce")
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna().sort_values("start_ms").reset_index(drop=True)

    # tz-naive
    df["open_time"] = pd.to_datetime(df["start_ms"], unit="ms", utc=True).dt.tz_convert(None)
    df["close_time"] = df["open_time"].apply(lambda t: _infer_close_time(t, bar))

    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time"]
    if "confirm" in df.columns:
        cols.append("confirm")
    return df[cols]

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
        df["buySignal"] = df["trendBull"] & df["macdBull"] & df["smcBullEvent"] & pd.Series(sweepBull)
    else:
        df["buySignal"] = df["trendBull"] & df["macdBull"] & df["smcBullEvent"]

    df["buyTrigger"] = df["buySignal"] & (~df["buySignal"].shift(1).fillna(False))

    # ===== Volume spike (Pump potential) =====
    df["volMA"] = df["volume"].rolling(VOL_MA_LEN).mean()
    df["volRatio"] = df["volume"] / df["volMA"]

    # ===== Choose last confirmed candle =====
    if "confirm" in df.columns:
        confirmed_idx = df.index[df["confirm"].astype(str) == "1"]
        if len(confirmed_idx) == 0:
            return None
        idx = int(confirmed_idx[-1])
    else:
        idx = len(df) - 2

    if idx < 0:
        return None

    # ===== Recent-close filter (no duplicates) =====
    now = pd.Timestamp.utcnow().tz_localize(None)
    ct = pd.to_datetime(df.at[idx, "close_time"], errors="coerce")
    if pd.isna(ct):
        return None

    ct = pd.Timestamp(ct).tz_localize(None)
    age_sec = abs((now - ct).total_seconds())
    is_recent_close = age_sec <= RECENT_WINDOW_SEC

    buy_now = bool(df.at[idx, "buyTrigger"]) and is_recent_close

    # Pump potential if buy + big volume spike
    vol_ratio = df.at[idx, "volRatio"]
    try:
        vol_ratio_f = float(vol_ratio)
    except Exception:
        vol_ratio_f = float("nan")

    pump = False
    if buy_now and (not math.isnan(vol_ratio_f)) and vol_ratio_f >= VOL_SPIKE_MULT:
        pump = True

    return {
        "buy": buy_now,
        "pump": pump,
        "vol_ratio": vol_ratio_f,
        "price": float(df.at[idx, "close"]),
        "time": str(df.at[idx, "close_time"]),
    }

# =========================
# Main
# =========================
def main():
    try:
        symbols = get_top_usdt_symbols(TOP_N)
    except Exception as e:
        err = f"❌ OKX API blocked from runner.\n{e}"
        print(err)
        tg_send(err)
        return

    alerts = []
    print(f"Scanning {len(symbols)} symbols on {TIMEFRAME} (OKX) ...")
    print(
        f"Recent-window (sec): {RECENT_WINDOW_SEC} | "
        f"VOL_MA_LEN={VOL_MA_LEN} | VOL_SPIKE_MULT={VOL_SPIKE_MULT}"
    )

    for sym in symbols:
        try:
            df = fetch_klines(sym, TIMEFRAME, limit=300)
            sig = evaluate_signals(df)
            if not sig:
                continue

            # ✅ BUY ONLY
            if sig["buy"]:
                if sig["pump"]:
                    alerts.append(
                        f"🟢 BUY | {sym} | TF {TIMEFRAME} | Price {sig['price']:.8g} | Close {sig['time']}\n"
                        f"🚀 PUMP محتمل | Volume Spike x{sig['vol_ratio']:.2f} (>= {VOL_SPIKE_MULT})"
                    )
                else:
                    alerts.append(
                        f"🟢 BUY | {sym} | TF {TIMEFRAME} | Price {sig['price']:.8g} | Close {sig['time']}"
                    )

        except Exception as e:
            print(f"{sym} error: {e}")

        time.sleep(0.25)

    if alerts:
        msg = f"📡 BUY Alerts Only (OKX) | TF {TIMEFRAME}\n" + "\n\n".join(alerts[:20])
        tg_send(msg)
        print("Sent BUY alerts:", len(alerts))
    else:
        print("No alerts.")


if __name__ == "__main__":
    main()
