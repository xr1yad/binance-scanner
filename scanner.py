import os
import time
import math
import random
import requests
import pandas as pd

# =========================
# Settings (from env)
# =========================
TIMEFRAME = os.getenv("TIMEFRAME", "15m")        # HTF signal timeframe (15m)
MONITOR_TF = os.getenv("MONITOR_TF", "1m")       # LTF used for early detection (1m)
SCAN_EVERY_MIN = int(os.getenv("SCAN_EVERY_MIN", "5"))  # matches cron */5

TOP_N = int(os.getenv("TOP_N", "60"))
USE_SWEEP = os.getenv("USE_SWEEP", "false").lower() == "true"

SMA_LEN = int(os.getenv("SMA_LEN", "200"))
PIVOT_LEN = int(os.getenv("PIVOT_LEN", "3"))
MACD_FAST = int(os.getenv("MACD_FAST", "12"))
MACD_SLOW = int(os.getenv("MACD_SLOW", "26"))
MACD_SIG  = int(os.getenv("MACD_SIG", "9"))

BOT_TOKEN = os.getenv("BOT_TOKEN", "")
CHAT_ID = os.getenv("CHAT_ID", "")

# Early detection window: compare "now snapshot" vs "past snapshot"
EARLY_LOOKBACK_MIN = int(os.getenv("EARLY_LOOKBACK_MIN", str(SCAN_EVERY_MIN)))

# Pump / Volume spike settings (on HTF candle)
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
    "1w": "1W",
    "1M": "1M",
}

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

def fetch_klines(symbol: str, timeframe: str, limit: int = 600) -> pd.DataFrame:
    bar = OKX_BAR_MAP.get(timeframe)
    if bar is None:
        raise ValueError(f"Unsupported TIMEFRAME: {timeframe}. Supported: {list(OKX_BAR_MAP.keys())}")

    host, status, data = http_get_json_multi_host("/api/v5/market/candles", {
        "instId": symbol,
        "bar": bar,
        "limit": str(min(int(limit), 300)),  # OKX max 300
    })

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

    df = pd.DataFrame(parsed, columns=[
        "start_ms", "open", "high", "low", "close", "volume",
        "volCcy", "volCcyQuote", "confirm"
    ][:len(parsed[0])])

    df["start_ms"] = pd.to_numeric(df["start_ms"], errors="coerce")
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna().sort_values("start_ms").reset_index(drop=True)

    # tz-naive (UTC)
    df["open_time"] = pd.to_datetime(df["start_ms"], unit="ms", utc=True).dt.tz_convert(None)
    df["close_time"] = df["open_time"].apply(lambda t: _infer_close_time(t, bar))

    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time"]
    if "confirm" in df.columns:
        cols.append("confirm")
    return df[cols]

# =========================
# Build HTF candles from 1m (forming candle supported)
# =========================
def _floor_time(ts: pd.Timestamp, minutes: int) -> pd.Timestamp:
    # floor to N-minute boundary (UTC naive)
    ts = pd.Timestamp(ts).tz_localize(None)
    minute = (ts.minute // minutes) * minutes
    return ts.replace(minute=minute, second=0, microsecond=0)

def build_htf_from_1m(df1m: pd.DataFrame, htf_minutes: int, cutoff: pd.Timestamp) -> pd.DataFrame:
    """
    Build HTF candles up to cutoff time (inclusive of last available 1m <= cutoff).
    The last HTF candle will be 'forming' based on available 1m data.
    """
    cutoff = pd.Timestamp(cutoff).tz_localize(None)

    # keep 1m rows with open_time <= cutoff
    d = df1m[df1m["open_time"] <= cutoff].copy()
    if d.empty:
        return pd.DataFrame()

    d = d.sort_values("open_time").reset_index(drop=True)

    # assign bucket open
    d["bucket_open"] = d["open_time"].apply(lambda t: _floor_time(t, htf_minutes))

    # aggregate OHLCV
    g = d.groupby("bucket_open", as_index=False).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        last_1m_time=("open_time", "max"),
    )

    g = g.sort_values("bucket_open").reset_index(drop=True)

    g["open_time"] = g["bucket_open"]
    g["close_time"] = g["open_time"] + pd.to_timedelta(htf_minutes, unit="m")

    # mark confirm: only candles fully closed before cutoff are confirmed
    g["confirm"] = (g["close_time"] <= cutoff).astype(int).astype(str)

    return g[["open_time", "open", "high", "low", "close", "volume", "close_time", "confirm"]]

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

def compute_buy_signal_on_htf(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns df with buySignal computed for ALL candles (including the forming last candle).
    """
    df = df.copy()
    df["sma200"] = df["close"].rolling(SMA_LEN).mean()

    macd_line, signal_line, hist = macd(df["close"], MACD_FAST, MACD_SLOW, MACD_SIG)
    df["macd"] = macd_line
    df["signal"] = signal_line
    df["hist"] = hist

    df["macdBull"] = (df["macd"] > df["signal"]) & (df["hist"] > 0)
    df["trendBull"] = df["close"] > df["sma200"]

    ph, pl = compute_pivots(df, PIVOT_LEN)
    df["ph"] = ph
    df["pl"] = pl

    lastSwingHigh = math.nan
    lastSwingLow = math.nan
    structureTrend = 0

    bosBull = [False] * len(df)
    chochBull = [False] * len(df)
    sweepBull = [False] * len(df)

    for i in range(len(df)):
        if not math.isnan(df.at[i, "ph"]):
            lastSwingHigh = float(df.at[i, "ph"])
        if not math.isnan(df.at[i, "pl"]):
            lastSwingLow = float(df.at[i, "pl"])

        c = float(df.at[i, "close"])
        lo = float(df.at[i, "low"])

        bBull = (not math.isnan(lastSwingHigh)) and (c > lastSwingHigh)
        cBull = (structureTrend == -1) and bBull

        bosBull[i] = bBull
        chochBull[i] = cBull

        if bBull:
            structureTrend = 1

        if not math.isnan(lastSwingLow):
            sweepBull[i] = (lo < lastSwingLow) and (c > lastSwingLow)

    df["smcBullEvent"] = (pd.Series(bosBull) | pd.Series(chochBull))

    if USE_SWEEP:
        df["buySignal"] = df["trendBull"] & df["macdBull"] & df["smcBullEvent"] & pd.Series(sweepBull)
    else:
        df["buySignal"] = df["trendBull"] & df["macdBull"] & df["smcBullEvent"]

    # volume spike on HTF
    df["volMA"] = df["volume"].rolling(VOL_MA_LEN).mean()
    df["volRatio"] = df["volume"] / df["volMA"]

    return df

def early_buy_trigger(df1m: pd.DataFrame, htf_minutes: int) -> dict | None:
    """
    Compare current snapshot vs past snapshot (EARLY_LOOKBACK_MIN).
    Trigger when buySignal becomes True within the last lookback window.
    """
    if df1m is None or df1m.empty:
        return None

    df1m = df1m.sort_values("open_time").reset_index(drop=True)
    latest = pd.Timestamp(df1m["open_time"].iloc[-1]).tz_localize(None)

    now_cutoff = latest
    past_cutoff = latest - pd.to_timedelta(max(1, EARLY_LOOKBACK_MIN), unit="m")

    htf_now = build_htf_from_1m(df1m, htf_minutes, now_cutoff)
    htf_past = build_htf_from_1m(df1m, htf_minutes, past_cutoff)

    # need enough data for sma/macd/pivots
    min_need = max(SMA_LEN + 10, 250)
    if len(htf_now) < min_need or len(htf_past) < min_need:
        return None

    htf_now = compute_buy_signal_on_htf(htf_now)
    htf_past = compute_buy_signal_on_htf(htf_past)

    now_sig = bool(htf_now["buySignal"].iloc[-1])
    past_sig = bool(htf_past["buySignal"].iloc[-1])

    if (not past_sig) and now_sig:
        # Pump?
        vol_ratio = htf_now["volRatio"].iloc[-1]
        try:
            vol_ratio_f = float(vol_ratio)
        except Exception:
            vol_ratio_f = float("nan")

        pump = False
        if (not math.isnan(vol_ratio_f)) and vol_ratio_f >= VOL_SPIKE_MULT:
            pump = True

        last_price = float(htf_now["close"].iloc[-1])
        bucket_open = str(htf_now["open_time"].iloc[-1])
        bucket_close = str(htf_now["close_time"].iloc[-1])
        live_time = str(latest)

        return {
            "buy": True,
            "pump": pump,
            "vol_ratio": vol_ratio_f,
            "price": last_price,
            "bucket_open": bucket_open,
            "bucket_close": bucket_close,
            "live_time": live_time,
        }

    return None

def parse_htf_minutes(tf: str) -> int:
    # supports 15m, 30m, 1h ...
    tf = tf.strip().lower()
    if tf.endswith("m"):
        return int(tf[:-1])
    if tf.endswith("h"):
        return int(tf[:-1]) * 60
    raise ValueError(f"Unsupported TIMEFRAME for early mode: {tf}")

def main():
    try:
        symbols = get_top_usdt_symbols(TOP_N)
    except Exception as e:
        err = f"❌ OKX API blocked from runner.\n{e}"
        print(err)
        tg_send(err)
        return

    # We use MONITOR_TF data (1m) to build TIMEFRAME candles (15m)
    htf_minutes = parse_htf_minutes(TIMEFRAME)

    alerts = []
    print(f"Scanning {len(symbols)} symbols | HTF={TIMEFRAME} (early) via LTF={MONITOR_TF}")
    print(f"EARLY_LOOKBACK_MIN={EARLY_LOOKBACK_MIN} | VOL_MA_LEN={VOL_MA_LEN} | VOL_SPIKE_MULT={VOL_SPIKE_MULT}")

    for sym in symbols:
        try:
            # fetch 1m (OKX max 300 per call) – 300 minutes = 5 hours, enough for lookback but SMA200 needs more HTF candles.
            # For SMA200 on 15m you need 200 candles => 3000 minutes => not possible in one call.
            # So we fetch HTF directly for history, and overlay the forming candle from 1m.
            # ---- Hybrid approach ----
            htf_hist = fetch_klines(sym, TIMEFRAME, limit=300)   # confirmed HTF history (max OKX)
            ltf = fetch_klines(sym, MONITOR_TF, limit=300)       # recent 1m for forming candle

            if htf_hist.empty or ltf.empty:
                continue

            # Build forming HTF candles from 1m up to latest
            latest_ltf = pd.Timestamp(ltf["open_time"].iloc[-1]).tz_localize(None)
            htf_from_1m = build_htf_from_1m(ltf, htf_minutes, latest_ltf)
            if htf_from_1m.empty:
                continue

            # Merge: take HTF history (confirmed) + replace/append last bucket with forming candle
            htf_all = htf_hist.copy().sort_values("open_time").reset_index(drop=True)

            forming_bucket_open = htf_from_1m["open_time"].iloc[-1]
            forming_row = htf_from_1m.iloc[-1:].copy()

            # remove any same bucket from hist then append forming
            htf_all = htf_all[htf_all["open_time"] != forming_bucket_open]
            htf_all = pd.concat([htf_all, forming_row], ignore_index=True).sort_values("open_time").reset_index(drop=True)

            # Now we can compute early trigger by comparing snapshots:
            # snapshot_now: using current ltf cutoff
            # snapshot_past: using ltf cutoff - EARLY_LOOKBACK_MIN
            sig = early_buy_trigger(ltf, htf_minutes)
            if not sig:
                continue

            # ✅ BUY ONLY
            if sig["buy"]:
                if sig["pump"]:
                    alerts.append(
                        f"🟢 BUY EARLY | {sym} | HTF {TIMEFRAME} (forming) | Price {sig['price']:.8g}\n"
                        f"🕒 LTF time {sig['live_time']} | HTF bucket {sig['bucket_open']} → {sig['bucket_close']}\n"
                        f"🚀 PUMP محتمل | Volume Spike x{sig['vol_ratio']:.2f} (>= {VOL_SPIKE_MULT})"
                    )
                else:
                    alerts.append(
                        f"🟢 BUY EARLY | {sym} | HTF {TIMEFRAME} (forming) | Price {sig['price']:.8g}\n"
                        f"🕒 LTF time {sig['live_time']} | HTF bucket {sig['bucket_open']} → {sig['bucket_close']}"
                    )

        except Exception as e:
            print(f"{sym} error: {e}")

        time.sleep(0.25)

    if alerts:
        msg = f"📡 BUY Alerts (EARLY) | HTF {TIMEFRAME} via LTF {MONITOR_TF}\n" + "\n\n".join(alerts[:20])
        tg_send(msg)
        print("Sent BUY alerts:", len(alerts))
    else:
        print("No alerts.")

if __name__ == "__main__":
    main()


