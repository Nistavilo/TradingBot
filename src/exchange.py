from typing import Optional, List
import pandas as pd
import ccxt
import time

def make_exchange(name: str):
    name = name.lower()
    if not hasattr(ccxt, name):
        raise ValueError(f"Bilinmeyen borsa: {name}")
    ex = getattr(ccxt, name)({
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    })
    return ex

def fetch_ohlcv_df(exchange, symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df.set_index("ts", inplace=True)
    df.sort_index(inplace=True)
    return df

def fetch_ohlcv_df_paginated(exchange, symbol: str, timeframe: str, limit_total: int = 1000, since_ms: Optional[int] = None, pause_sec: float = 0.2) -> pd.DataFrame:
    per_call_limit = min(1000, limit_total)
    all_rows: List[List[float]] = []
    tf_ms = int(exchange.parse_timeframe(timeframe) * 1000)
    since = since_ms

    while len(all_rows) < limit_total:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=per_call_limit, since=since)
        if not batch:
            break
        if all_rows and batch[0][0] <= all_rows[-1][0]:
            since = all_rows[-1][0] + tf_ms
            continue
        all_rows.extend(batch)
        if len(batch) < per_call_limit:
            break
        since = batch[-1][0] + tf_ms
        time.sleep(pause_sec)

    if not all_rows:
        return pd.DataFrame(columns=["open","high","low","close","volume"])

    all_rows = all_rows[-limit_total:]

    df = pd.DataFrame(all_rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df.set_index("ts", inplace=True)
    df.sort_index(inplace=True)
    return df

def append_latest_candle(df: pd.DataFrame, exchange, symbol: str, timeframe: str) -> pd.DataFrame:
    last = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=2)
    if not last:
        return df
    ts, o, h, l, c, v = last[-1]
    ts = pd.to_datetime(ts, unit="ms", utc=True)

    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    for col in ohlcv_cols:
        if col not in df.columns:
            df[col] = pd.Series(dtype="float64")

    if ts in df.index:
        df.loc[ts, ohlcv_cols] = [o, h, l, c, v]
    else:
        df.loc[ts, ohlcv_cols] = [o, h, l, c, v]

    return df.sort_index()