import os
import json
import time
import math
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List

import numpy as np
import pandas as pd
import requests

BINANCE_KLINES = "https://api.binance.com/api/v3/klines"


# -------------------------
# Helpers: indicators
# -------------------------

def wma(series: pd.Series, length: int) -> pd.Series:
    if length <= 0:
        return pd.Series(np.nan, index=series.index)
    weights = np.arange(1, length + 1, dtype=float)
    return series.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def hma(series: pd.Series, length: int) -> pd.Series:
    # HMA(n) = WMA( 2*WMA(price, n/2) - WMA(price, n), sqrt(n) )
    if length <= 0:
        return pd.Series(np.nan, index=series.index)
    half = max(1, length // 2)
    sqrt_n = max(1, int(math.sqrt(length)))
    wma_full = wma(series, length)
    wma_half = wma(series, half)
    raw = 2 * wma_half - wma_full
    return wma(raw, sqrt_n)

def rsi(series: pd.Series, length: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def atr_wilder(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1/length, adjust=False, min_periods=length).mean()

def atr_sma_tr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.rolling(length).mean()

def macd(close: pd.Series, fast: int, slow: int, signal: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False).mean()
    hist = line - sig
    return line, sig, hist


# -------------------------
# Pine translation core
# -------------------------

@dataclass
class SignalResult:
    symbol: str
    timeframe: str
    ts_close: int  # candle close time (ms)
    side: str      # "BUY" or "SELL"
    entry: float
    sl: float
    tp1: float
    tp2: float
    tp3: float


def fetch_klines(symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
    r = requests.get(BINANCE_KLINES, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    # Binance kline fields:
    # 0 open time, 1 open, 2 high, 3 low, 4 close, 5 volume, 6 close time, ...
    df = pd.DataFrame(data, columns=[
        "open_time","open","high","low","close","volume","close_time",
        "qav","num_trades","tbbav","tbqav","ignore"
    ])
    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)
    df["open_time"] = df["open_time"].astype(np.int64)
    df["close_time"] = df["close_time"].astype(np.int64)
    return df


def compute_signals(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    multiplier: float = 2.0,
    atr_len: int = 14,
    atr_method: str = "Method 1",   # "Method 1" = Wilder ATR, "Method 2" = SMA(TR)
    stoploss_pct: float = 2.0,      # Pine stopLossVal (0 disables)
) -> Optional[SignalResult]:
    # Use last CLOSED candle like Pine alert.freq_once_per_bar_close
    if len(df) < max(atr_len * 3, 60):
        return None

    o = df["open"]
    h = df["high"]
    l = df["low"]
    c = df["close"]

    src = (h + l) / 2.0  # hl2

    # src1 = ta.hma(open, 5)[1]
    src1 = hma(o, 5).shift(1)
    # src2 = ta.hma(close, 12)
    src2 = hma(c, 12)

    momm1 = src1.diff()
    momm2 = src2.diff()

    # f1(m, n) => m >= n ? m : 0.0
    # f2(m, n) => m >= n ? 0.0 : -m
    m1 = np.where(momm1 >= momm2, momm1, 0.0)
    m2 = np.where(momm1 >= momm2, 0.0, -momm1)

    sm1 = pd.Series(m1, index=df.index).rolling(1).sum()
    sm2 = pd.Series(m2, index=df.index).rolling(1).sum()

    # percent(sm1-sm2, sm1+sm2) => 100 * nom / div
    denom = (sm1 + sm2).replace(0, np.nan)
    cmo_calc = 100.0 * (sm1 - sm2) / denom

    # hpivot/lpivot logic (كما في كودك)
    hh = h.rolling(2).max()
    hh_dev = hh.rolling(2).std(ddof=0)  # ta.dev
    hpivot = hh.where((hh_dev.fillna(0) == 0), np.nan).ffill()

    ll = l.rolling(2).min()
    ll_dev = ll.rolling(2).std(ddof=0)
    lpivot = ll.where((ll_dev.fillna(0) == 0), np.nan).ffill()

    rsi_calc = rsi(c, 9)

    # sup/res موجودة بالكود لكنها غير مستخدمة في الإشارات النهائية
    # sup = (rsi_calc < 25) & (cmo_calc > 50) & lpivot.notna()
    # res = (rsi_calc > 75) & (cmo_calc < -50) & hpivot.notna()

    # ATR
    if atr_method == "Method 2":
        atr_val = atr_sma_tr(h, l, c, atr_len)
    else:
        atr_val = atr_wilder(h, l, c, atr_len)

    up = src - (multiplier * atr_val)
    dn = src + (multiplier * atr_val)

    # trailing logic:
    up1 = up.shift(1)
    dn1 = dn.shift(1)

    up_tr = up.copy()
    dn_tr = dn.copy()

    for i in range(1, len(df)):
        # up := close[1] > up1 ? max(up, up1) : up
        if c.iloc[i-1] > up1.iloc[i]:
            up_tr.iloc[i] = max(up.iloc[i], up1.iloc[i])
        else:
            up_tr.iloc[i] = up.iloc[i]

        # dn := close[1] < dn1 ? min(dn, dn1) : dn
        if c.iloc[i-1] < dn1.iloc[i]:
            dn_tr.iloc[i] = min(dn.iloc[i], dn1.iloc[i])
        else:
            dn_tr.iloc[i] = dn.iloc[i]

    # trend switching
    trend = pd.Series(1, index=df.index, dtype=int)
    for i in range(1, len(df)):
        prev = trend.iloc[i-1]
        if prev == -1 and c.iloc[i] > dn_tr.iloc[i-1]:
            trend.iloc[i] = 1
        elif prev == 1 and c.iloc[i] < up_tr.iloc[i-1]:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = prev

    buy_signal = (trend == 1) & (trend.shift(1) == -1)
    sell_signal = (trend == -1) & (trend.shift(1) == 1)

    # pos logic + SL/TP reset
    pos = pd.Series(0.0, index=df.index)
    for i in range(1, len(df)):
        if buy_signal.iloc[i]:
            pos.iloc[i] = 1
        elif sell_signal.iloc[i]:
            pos.iloc[i] = -1
        else:
            pos.iloc[i] = pos.iloc[i-1]

    long_cond = buy_signal & (pos.shift(1) != 1)
    short_cond = sell_signal & (pos.shift(1) != -1)

    # valuewhen(longCond, close, 0) => last close where long_cond true
    entry_long = c.where(long_cond).ffill()
    entry_short = c.where(short_cond).ffill()

    sl = (stoploss_pct / 100.0) if stoploss_pct > 0 else 99999.0

    stop_long = entry_long * (1 - sl)
    stop_short = entry_short * (1 + sl)

    tp_long_1 = entry_long * (1 + sl)
    tp_long_2 = entry_long * (1 + sl * 2)
    tp_long_3 = entry_long * (1 + sl * 3)

    tp_short_1 = entry_short * (1 - sl)
    tp_short_2 = entry_short * (1 - sl * 2)
    tp_short_3 = entry_short * (1 - sl * 3)

    # Reset pos to 0 when SL/TP3 hit (كما بالكود)
    # Note: هذا جزء "إدارة صفقة" وليس شرط الدخول نفسه،
    # لكنه مهم عشان pos يرجع 0 لو سكربتك يعتمد عليها بعدين.
    pos2 = pos.copy()
    for i in range(1, len(df)):
        prev_pos = pos2.iloc[i-1]

        long_sl = (l.iloc[i] < stop_long.iloc[i]) and (prev_pos == 1)
        short_sl = (h.iloc[i] > stop_short.iloc[i]) and (prev_pos == -1)

        long_tp_final = (h.iloc[i] > tp_long_3.iloc[i]) and (prev_pos == 1)
        short_tp_final = (l.iloc[i] < tp_short_3.iloc[i]) and (prev_pos == -1)

        if long_sl or short_sl or long_tp_final or short_tp_final:
            pos2.iloc[i] = 0.0
        else:
            pos2.iloc[i] = prev_pos if not (buy_signal.iloc[i] or sell_signal.iloc[i]) else pos2.iloc[i]

    # نستخدم آخر شمعة مغلقة = قبل آخر صف في Binance (الأخير غالباً مغلق، لكن أضمن: نأخذ -2)
    idx = df.index[-2]
    close_time = int(df.loc[idx, "close_time"])

    if bool(long_cond.loc[idx]):
        e = float(entry_long.loc[idx])
        return SignalResult(
            symbol=symbol, timeframe=timeframe, ts_close=close_time,
            side="BUY", entry=e,
            sl=float(stop_long.loc[idx]),
            tp1=float(tp_long_1.loc[idx]), tp2=float(tp_long_2.loc[idx]), tp3=float(tp_long_3.loc[idx]),
        )

    if bool(short_cond.loc[idx]):
        e = float(entry_short.loc[idx])
        return SignalResult(
            symbol=symbol, timeframe=timeframe, ts_close=close_time,
            side="SELL", entry=e,
            sl=float(stop_short.loc[idx]),
            tp1=float(tp_short_1.loc[idx]), tp2=float(tp_short_2.loc[idx]), tp3=float(tp_short_3.loc[idx]),
        )

    return None


# -------------------------
# Telegram + State
# -------------------------

def load_state(path: str = "state.json") -> Dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"sent": {}}

def save_state(state: Dict, path: str = "state.json") -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

def tg_send(bot_token: str, chat_id: str, text: str) -> None:
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "disable_web_page_preview": True}
    r = requests.post(url, json=payload, timeout=30)
    r.raise_for_status()

def make_key(sig: SignalResult) -> str:
    return f"{sig.symbol}|{sig.timeframe}|{sig.ts_close}|{sig.side}"

def format_message(sig: SignalResult) -> str:
    # قريب من نص تنبيهاتك في Pine
    return (
        f"{sig.symbol} {sig.side} ALERT!\n"
        f"TF: {sig.timeframe}\n"
        f"Entry: {sig.entry:.8f}\n"
        f"TP1: {sig.tp1:.8f}\n"
        f"TP2: {sig.tp2:.8f}\n"
        f"TP3: {sig.tp3:.8f}\n"
        f"SL : {sig.sl:.8f}\n"
        f"Candle close (ms): {sig.ts_close}"
    )


def main():
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not bot_token or not chat_id:
        raise SystemExit("Missing TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID env vars")

    symbols_raw = os.getenv("SYMBOLS", "BTCUSDT,ETHUSDT").strip()
    timeframe = os.getenv("TIMEFRAME", "1h").strip()
    limit = int(os.getenv("LIMIT", "500"))

    multiplier = float(os.getenv("MULTIPLIER", "2"))
    atr_len = int(os.getenv("ATR_LEN", "14"))
    atr_method = os.getenv("ATR_METHOD", "Method 1").strip()  # "Method 1" or "Method 2"
    stoploss_pct = float(os.getenv("STOPLOSS_PCT", "2.0"))

    symbols = [s.strip().upper() for s in symbols_raw.split(",") if s.strip()]

    state = load_state()
    sent = state.get("sent", {})

    for sym in symbols:
        try:
            df = fetch_klines(sym, timeframe, limit=limit)
            sig = compute_signals(
                df=df, symbol=sym, timeframe=timeframe,
                multiplier=multiplier, atr_len=atr_len,
                atr_method=atr_method, stoploss_pct=stoploss_pct,
            )
            if sig is None:
                continue

            key = make_key(sig)
            if sent.get(key):
                continue  # already sent this exact signal

            msg = format_message(sig)
            tg_send(bot_token, chat_id, msg)

            sent[key] = int(time.time())
            state["sent"] = sent
            save_state(state)

        except Exception as e:
            # لا نوقف السكربت بالكامل بسبب رمز واحد
            err_msg = f"[ERROR] {sym} {timeframe}: {type(e).__name__}: {e}"
            print(err_msg)

    print("Done")


if __name__ == "__main__":
    main()
