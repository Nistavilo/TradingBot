import argparse
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Iterable
from pathlib import Path
import time
import json
import itertools

import numpy as np
import pandas as pd

from src.config import Settings
from src.exchange import make_exchange, fetch_ohlcv_df
from src.indicators import compute_indicators
from src.storage import symbol_to_filename


@dataclass
class Trade:
    entry_ts: pd.Timestamp
    entry_price: float
    exit_ts: Optional[pd.Timestamp]
    exit_price: Optional[float]
    pnl_pct: Optional[float]
    initial_stop: Optional[float] = None
    risk_distance: Optional[float] = None
    r_multiple: Optional[float] = None


# ------------------ Utility / Metrics ------------------ #

def periods_per_year_from_timeframe(timeframe: str, df: pd.DataFrame) -> float:
    mapping = {
        "1m": 525600, "3m": 175200, "5m": 105120, "15m": 35040, "30m": 17520,
        "1h": 8760, "2h": 4380, "4h": 2190, "6h": 1460, "12h": 730,
        "1d": 365, "3d": 121.7, "1w": 52, "1M": 12
    }
    if timeframe in mapping:
        return mapping[timeframe]
    if len(df) >= 2:
        med_delta = (df.index.to_series().diff().median())
        if pd.isna(med_delta) or med_delta is pd.NaT:
            return 365.0
        secs = med_delta / np.timedelta64(1, "s")
        if secs <= 0:
            return 365.0
        return (365.0 * 24 * 3600) / secs
    return 365.0


def fetch_ohlcv_df_paginated_local(exchange, symbol: str, timeframe: str, limit_total: int = 1000,
                                   since_ms: Optional[int] = None, pause_sec: float = 0.25) -> pd.DataFrame:
    per_call_limit = min(1000, max(1, limit_total))
    all_rows: List[List[float]] = []
    tf_ms = int(exchange.parse_timeframe(timeframe) * 1000)

    now_ms = exchange.milliseconds() if hasattr(exchange, "milliseconds") else int(time.time() * 1000)
    start_ms = since_ms if since_ms is not None else max(0, now_ms - (limit_total + 5) * tf_ms)

    cursor = start_ms
    last_appended = -1

    while len(all_rows) < limit_total:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=per_call_limit, since=cursor)
        if not batch:
            break

        appended = 0
        for row in batch:
            ts = row[0]
            if ts > last_appended:
                all_rows.append(row)
                last_appended = ts
                appended += 1

        if appended == 0 or len(batch) < per_call_limit:
            break

        cursor = batch[-1][0] + tf_ms
        time.sleep(pause_sec)

    if not all_rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    all_rows = all_rows[-limit_total:]

    df = pd.DataFrame(all_rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df.set_index("ts", inplace=True)
    df.sort_index(inplace=True)
    return df


def merge_trend(df: pd.DataFrame, htf_df: pd.DataFrame, trend_ma: int, ma_type: str = "sma") -> pd.DataFrame:
    htf = htf_df.copy()
    if ma_type.lower() == "ema":
        htf["htf_ma"] = htf["close"].ewm(span=trend_ma, adjust=False).mean()
    else:
        htf["htf_ma"] = htf["close"].rolling(window=trend_ma, min_periods=trend_ma).mean()

    a = df.reset_index().rename(columns={"ts": "ts"})
    b = htf.reset_index().rename(columns={"ts": "ts", "close": "htf_close"})

    merged = pd.merge_asof(
        a.sort_values("ts"),
        b[["ts", "htf_close", "htf_ma"]].sort_values("ts"),
        on="ts", direction="backward",
    ).set_index("ts")

    merged.index = pd.to_datetime(merged.index, utc=True)
    return merged


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()


def compute_stoch(df: pd.DataFrame, period: int = 14, k_smooth: int = 3, d_smooth: int = 3) -> pd.DataFrame:
    ll = df["low"].rolling(period, min_periods=period).min()
    hh = df["high"].rolling(period, min_periods=period).max()
    k = 100 * (df["close"] - ll) / (hh - ll).replace(0, np.nan)
    k = k.rolling(k_smooth, min_periods=1).mean()
    d = k.rolling(d_smooth, min_periods=1).mean()
    return pd.DataFrame({"stoch_k": k, "stoch_d": d})


def compute_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return pd.DataFrame({"macd": macd, "macd_signal": macd_signal, "macd_hist": macd_hist})


def compute_metrics(equity: List[float], df_index: pd.Index, timeframe: str) -> dict:
    if not equity:
        return {}
    eq = np.array(equity, dtype=float)
    start, end = eq[0], eq[-1]
    total_ret = (end / start) - 1.0

    peaks = np.maximum.accumulate(eq)
    dd = (eq / peaks) - 1.0
    max_dd = dd.min() if len(dd) else 0.0

    rets = np.diff(eq) / eq[:-1] if len(eq) > 1 else np.array([])
    pp_year = periods_per_year_from_timeframe(timeframe, pd.DataFrame(index=df_index))
    sharpe = 0.0
    if len(rets) > 1:
        mu = rets.mean()
        sd = rets.std(ddof=1) + 1e-12
        sharpe = (mu / sd) * np.sqrt(pp_year)

    if len(df_index) >= 2:
        days = (df_index[-1] - df_index[0]).total_seconds() / (3600 * 24)
        cagr = (end / start) ** (365.0 / max(1e-9, days)) - 1.0 if days > 0 else np.nan
    else:
        cagr = np.nan

    return {
        "start_equity": float(start),
        "end_equity": float(end),
        "total_return_pct": total_ret * 100.0,
        "max_drawdown_pct": max_dd * 100.0,
        "sharpe": sharpe,
        "cagr_pct": (cagr * 100.0) if pd.notna(cagr) else np.nan,
    }


def extra_trade_metrics(trades: List[Trade]) -> dict:
    closed = [t for t in trades if t.exit_ts and t.pnl_pct is not None]
    if not closed:
        return {}
    rets = np.array([t.pnl_pct for t in closed], dtype=float)
    wins = rets[rets > 0]
    losses = rets[rets <= 0]
    profit_factor = float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else np.nan
    downside = rets[rets < 0]
    sortino = float(rets.mean() / (downside.std(ddof=1) + 1e-12)) if len(downside) > 0 else np.nan
    expectancy = float(rets.mean() * 100.0)
    r_values = [t.r_multiple for t in closed if t.r_multiple is not None]
    avg_r = float(np.mean(r_values)) if r_values else np.nan
    return {
        "profit_factor": profit_factor,
        "sortino": sortino,
        "expectancy_pct": expectancy,
        "avg_r_multiple": avg_r
    }


# ------------------ Core Backtest ------------------ #

def prepare_base_dataframe(symbol: str,
                           timeframe: str,
                           limit: int,
                           short: int,
                           long: int,
                           rsi_period: int,
                           use_trend: bool,
                           trend_timeframe: str,
                           trend_sma: int,
                           trend_ma_type: str,
                           dynamic_atr: bool,
                           atr_period: int,
                           atr_period_fast: int,
                           atr_period_slow: int,
                           atr_switch_pct: float,
                           stoch_period: int,
                           stoch_k_smooth: int,
                           stoch_d_smooth: int,
                           use_stoch: bool,
                           use_macd: bool,
                           csv_in: Optional[str]) -> pd.DataFrame:
    """
    Tek seferlik veri + temel indikatörler hazırlığı (grid / walk-forward için).
    """
    if csv_in:
        df = pd.read_csv(csv_in)
        if "timestamp" not in df.columns:
            raise ValueError("CSV must contain 'timestamp' column.")
        df["ts"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("ts").sort_index()
        need_ind = any(col not in df.columns for col in ["sma_short", "sma_long", "rsi"])
        if need_ind or (short != Settings.SMA_SHORT or long != Settings.SMA_LONG or rsi_period != Settings.RSI_PERIOD):
            base = df[["open", "high", "low", "close", "volume"]].copy()
            df = compute_indicators(base, short, long, rsi_period)
    else:
        ex = make_exchange(Settings.EXCHANGE)
        if limit > 1000:
            df = fetch_ohlcv_df_paginated_local(ex, symbol, timeframe, limit_total=limit)
        else:
            df = fetch_ohlcv_df(ex, symbol, timeframe, limit=limit)
        df = compute_indicators(df, short, long, rsi_period)

    df = df.dropna(subset=["close"]).copy().sort_index()

    # Prev columns
    df["rsi_prev"] = df["rsi"].shift(1)
    df["sma_short_prev"] = df["sma_short"].shift(1)
    df["sma_long_prev"] = df["sma_long"].shift(1)

    # Stoch (her ihtimale karşı hesaplanıyor; use_stoch flag koşumda kontrol)
    stoch = compute_stoch(df, stoch_period, stoch_k_smooth, stoch_d_smooth)
    df = pd.concat([df, stoch], axis=1)
    df["stoch_k_prev"] = df["stoch_k"].shift(1)
    df["stoch_d_prev"] = df["stoch_d"].shift(1)

    # MACD
    macd = compute_macd(df)
    df = pd.concat([df, macd], axis=1)

    # ATR
    if dynamic_atr:
        atr_fast = compute_atr(df, period=atr_period_fast)
        atr_slow = compute_atr(df, period=atr_period_slow)
        atr_pct_ref = (atr_slow / df["close"]) * 100.0
        use_fast = (atr_pct_ref > atr_switch_pct)
        df["atr"] = np.where(use_fast, atr_fast, atr_slow)
    else:
        df["atr"] = compute_atr(df, period=atr_period)

    # HTF trend
    if use_trend:
        ex = make_exchange(Settings.EXCHANGE)
        if limit > 1000:
            htf_df = fetch_ohlcv_df_paginated_local(ex, symbol, trend_timeframe,
                                                    limit_total=max(500, trend_sma + 200))
        else:
            htf_df = fetch_ohlcv_df(ex, symbol, trend_timeframe, limit=max(500, trend_sma + 200))
        merged = merge_trend(df, htf_df, trend_sma, ma_type=trend_ma_type)
        df["htf_close"] = merged["htf_close"]
        df["htf_ma"] = merged["htf_ma"]

    return df


def run_backtest(
    symbol: str,
    timeframe: str,
    limit: int,
    short: int,
    long: int,
    rsi_period: int,
    rsi_lower: float,
    rsi_upper: float,
    use_trend: bool,
    trend_timeframe: str,
    trend_sma: int,
    initial_cash: float,
    fee_rate: float,
    exposure: float,
    csv_in: Optional[str],
    out_dir: str,

    entry_mode: str = "cross",
    exit_mode: str = "trailing",

    atr_period: int = 14,
    sl_atr: float = 1.5,
    tp_atr: float = 0.0,
    trail_atr: float = 1.5,

    spread_bps: float = 2.0,
    slippage_bps: float = 3.0,

    atr_min_pct: float = 0.0,
    atr_max_pct: float = 1e9,
    min_quote_vol_usd: float = 0.0,

    trend_ma_type: str = "sma",

    use_stoch: bool = False,
    stoch_period: int = 14,
    stoch_k_smooth: int = 3,
    stoch_d_smooth: int = 3,
    stoch_max_k: float = 80.0,
    use_macd: bool = False,

    confirm_trend_with_momentum: bool = False,
    momentum_source: str = "either",

    risk_per_trade: float = 0.0,
    tp1_atr: float = 0.0,
    tp1_pct: float = 0.5,
    break_even_after_tp1: bool = True,

    dynamic_atr: bool = False,
    atr_period_fast: int = 10,
    atr_period_slow: int = 20,
    atr_switch_pct: float = 1.5,

    risk_scale_by_atr: bool = False,
    atr_target_pct: float = 1.0,
    risk_scale_min: float = 0.5,
    risk_scale_max: float = 1.5,

    exit_on_rsi_overbought: bool = False,
    exit_on_stoch_cross: bool = False,
    exit_on_macd_hist_neg: bool = False,

    debug_entry_blocks: bool = False,

    intra_bar_priority: str = "stop",
    jsonl_log: bool = False,

    # Yeni: önceden hazırlanmış df kullanımı ve tarih aralığı
    data_df: Optional[pd.DataFrame] = None,
    start_ts: Optional[pd.Timestamp] = None,
    end_ts: Optional[pd.Timestamp] = None,
):
    """
    data_df verilirse veri hazırlama aşaması atlanır. start_ts / end_ts ile slice yapılır.
    """
    if data_df is not None:
        df = data_df.copy()
    else:
        # Tekil koşum için veri hazırlığı (eski yol)
        df = prepare_base_dataframe(
            symbol=symbol, timeframe=timeframe, limit=limit,
            short=short, long=long, rsi_period=rsi_period,
            use_trend=use_trend, trend_timeframe=trend_timeframe, trend_sma=trend_sma,
            trend_ma_type=trend_ma_type, dynamic_atr=dynamic_atr,
            atr_period=atr_period, atr_period_fast=atr_period_fast, atr_period_slow=atr_period_slow,
            atr_switch_pct=atr_switch_pct,
            stoch_period=stoch_period, stoch_k_smooth=stoch_k_smooth, stoch_d_smooth=stoch_d_smooth,
            use_stoch=use_stoch, use_macd=use_macd,
            csv_in=csv_in
        )

    if start_ts:
        df = df[df.index >= pd.to_datetime(start_ts, utc=True)]
    if end_ts:
        df = df[df.index <= pd.to_datetime(end_ts, utc=True)]

    if df.empty or len(df) < 10:
        return {
            "params": {},
            "metrics": {},
            "counts": {"candles": len(df)},
            "outputs": {}
        }

    total_buy_bps = (spread_bps / 2.0) + slippage_bps
    total_sell_bps = (spread_bps / 2.0) + slippage_bps

    def buy_exec(price: float) -> float:
        return price * (1.0 + total_buy_bps / 10000.0)

    def sell_exec(price: float) -> float:
        return price * (1.0 - total_sell_bps / 10000.0)

    cash = initial_cash
    qty = 0.0
    equity_curve: List[float] = []
    trades: List[Trade] = []

    position = 0
    current_trade: Optional[Trade] = None

    initial_stop: Optional[float] = None
    trail_stop: Optional[float] = None
    take_profit: Optional[float] = None
    highest_since_entry: Optional[float] = None
    tp1_done: bool = False

    buy_count = 0
    sell_count = 0

    dbg_trend_fail = dbg_atr_fail = dbg_vol_fail = dbg_momo_fail = 0
    dbg_base_ready = 0

    out_base = f"{symbol_to_filename(symbol)}_{timeframe}"
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    events_path = out_dir_path / f"backtest_{out_base}_events.jsonl"
    event_buffer: List[str] = []

    def log_event(ev_type: str, ts, **payload):
        if not jsonl_log:
            return
        rec = {
            "type": ev_type,
            "ts": ts.isoformat() if isinstance(ts, pd.Timestamp) else str(ts),
            **payload
        }
        event_buffer.append(json.dumps(rec))

    for ts, row in df.iterrows():
        price_c = float(row["close"])
        price_h = float(row["high"])
        price_l = float(row["low"])

        sma_s = float(row.get("sma_short")) if pd.notna(row.get("sma_short")) else np.nan
        sma_l = float(row.get("sma_long")) if pd.notna(row.get("sma_long")) else np.nan
        rsi = float(row.get("rsi")) if pd.notna(row.get("rsi")) else np.nan
        rsi_prev = float(row.get("rsi_prev")) if pd.notna(row.get("rsi_prev")) else np.nan
        atr_val = float(row.get("atr")) if pd.notna(row.get("atr")) else np.nan
        k = float(row.get("stoch_k")) if pd.notna(row.get("stoch_k")) else np.nan
        d = float(row.get("stoch_d")) if pd.notna(row.get("stoch_d")) else np.nan
        k_prev = float(row.get("stoch_k_prev")) if pd.notna(row.get("stoch_k_prev")) else np.nan
        d_prev = float(row.get("stoch_d_prev")) if pd.notna(row.get("stoch_d_prev")) else np.nan
        macd_hist = float(row.get("macd_hist")) if pd.notna(row.get("macd_hist")) else np.nan

        cond_trend = True
        if use_trend:
            htf_close = row.get("htf_close")
            htf_ma = row.get("htf_ma")
            if pd.notna(htf_close) and pd.notna(htf_ma):
                cond_trend = (htf_close > htf_ma)

        cond_stoch = (pd.notna(k) and pd.notna(d) and (k > d) and (k <= stoch_max_k)) if use_stoch else True
        cond_macd = (pd.notna(macd_hist) and macd_hist > 0.0) if use_macd else True

        if confirm_trend_with_momentum:
            if momentum_source == "macd":
                cond_trend = cond_trend and cond_macd
            elif momentum_source == "stoch":
                cond_trend = cond_trend and cond_stoch
            elif momentum_source == "both":
                cond_trend = cond_trend and cond_macd and cond_stoch
            else:
                cond_trend = cond_trend and (cond_macd or cond_stoch)

        cond_atr = True
        if pd.notna(atr_val):
            atr_pct = 100.0 * (atr_val / price_c) if price_c > 0 else np.nan
            if pd.notna(atr_pct):
                cond_atr = (atr_pct >= atr_min_pct) and (atr_pct <= atr_max_pct)

        cond_vol = True
        if min_quote_vol_usd > 0:
            quote_vol = (row.get("volume") or 0.0) * price_c
            cond_vol = quote_vol >= min_quote_vol_usd

        if momentum_source == "macd":
            cond_momo = cond_macd
        elif momentum_source == "stoch":
            cond_momo = cond_stoch
        elif momentum_source == "both":
            cond_momo = cond_macd and cond_stoch
        else:
            if use_stoch or use_macd:
                cond_momo = cond_macd or cond_stoch
            else:
                cond_momo = True

        base_ready = False
        if pd.notna(rsi) and pd.notna(sma_s) and pd.notna(sma_l):
            if entry_mode == "level":
                base_ready = (rsi < rsi_lower) and (sma_s > sma_l)
            else:
                if pd.notna(rsi_prev):
                    base_ready = (rsi_prev < rsi_lower) and (rsi >= rsi_lower) and (sma_s > sma_l)

        if debug_entry_blocks and position == 0 and base_ready:
            dbg_base_ready += 1
            if not cond_trend:
                dbg_trend_fail += 1
            if not cond_atr:
                dbg_atr_fail += 1
            if not cond_vol:
                dbg_vol_fail += 1
            if not cond_momo:
                dbg_momo_fail += 1

        entry_conditions = (base_ready and cond_trend and cond_atr and cond_vol and cond_momo)

        stoch_downcross = (pd.notna(k_prev) and pd.notna(d_prev) and pd.notna(k) and pd.notna(d)
                           and (k_prev >= d_prev) and (k < d))
        extra_exit = (
            (exit_on_rsi_overbought and pd.notna(rsi) and (rsi >= rsi_upper)) or
            (exit_on_stoch_cross and stoch_downcross) or
            (exit_on_macd_hist_neg and pd.notna(macd_hist) and (macd_hist <= 0.0))
        )

        if position == 0 and entry_conditions:
            exec_price = buy_exec(price_c)

            if risk_per_trade > 0 and pd.notna(atr_val) and sl_atr > 0:
                stop_level = exec_price - sl_atr * atr_val
                stop_distance = max(1e-8, exec_price - stop_level)
                equity_now = cash
                risk_scale = 1.0
                if risk_scale_by_atr:
                    atr_pct_now = (atr_val / price_c) * 100.0 if price_c > 0 else np.nan
                    if pd.notna(atr_pct_now) and atr_pct_now > 0:
                        risk_scale = atr_target_pct / atr_pct_now
                        risk_scale = max(risk_scale_min, min(risk_scale_max, risk_scale))
                risk_cash = equity_now * max(0.0, min(1.0, risk_per_trade * risk_scale))
                target_qty = risk_cash / stop_distance
                max_affordable_qty = max(0.0, (cash - (cash * fee_rate)) / exec_price)
                size_qty = min(target_qty, max_affordable_qty)
            else:
                spend = cash * max(0.0, min(1.0, exposure))
                size_qty = max(0.0, (spend - spend * fee_rate) / exec_price)

            spend_est = size_qty * exec_price
            fee = spend_est * fee_rate
            cash -= spend_est
            cash -= fee
            qty += size_qty
            position = 1
            buy_count += 1

            if pd.notna(atr_val) and sl_atr > 0:
                initial_stop = exec_price - sl_atr * atr_val
            else:
                initial_stop = None
            if pd.notna(atr_val) and trail_atr > 0:
                trail_stop = exec_price - trail_atr * atr_val
            else:
                trail_stop = None
            if pd.notna(atr_val) and tp_atr > 0:
                take_profit = exec_price + tp_atr * atr_val
            else:
                take_profit = None
            highest_since_entry = price_h
            tp1_done = False

            risk_distance = (exec_price - initial_stop) if (initial_stop is not None) else None
            current_trade = Trade(
                entry_ts=ts,
                entry_price=exec_price,
                exit_ts=None,
                exit_price=None,
                pnl_pct=None,
                initial_stop=initial_stop,
                risk_distance=risk_distance,
                r_multiple=None
            )
            log_event("entry", ts,
                      price=exec_price, qty=size_qty,
                      initial_stop=initial_stop, trail_stop=trail_stop,
                      take_profit=take_profit, risk_distance=risk_distance)

        elif position == 1:
            highest_since_entry = max(highest_since_entry or -float("inf"), price_h)
            exit_price_exec: Optional[float] = None
            will_exit = False

            if exit_mode == "trailing":
                stop_candidates = [x for x in (initial_stop, trail_stop) if x is not None]
                stop_lvl = max(stop_candidates) if stop_candidates else None
                tp_lvl = None
                if tp_atr > 0 and pd.notna(atr_val) and current_trade:
                    tp_lvl = current_trade.entry_price + tp_atr * atr_val

                if pd.notna(atr_val) and trail_atr > 0 and highest_since_entry is not None:
                    trail_candidate = highest_since_entry - trail_atr * atr_val
                    new_trail = max(trail_stop or -float("inf"), trail_candidate)
                    if new_trail != trail_stop:
                        trail_stop = new_trail
                        log_event("trail_update", ts, trail_stop=trail_stop)

                stop_hit = (stop_lvl is not None) and (price_l <= stop_lvl)
                tp_hit = (tp_lvl is not None) and (price_h >= tp_lvl)

                if stop_hit and tp_hit:
                    if intra_bar_priority in ("stop", "stop_then_tp"):
                        exit_price_exec = sell_exec(stop_lvl)
                    else:
                        exit_price_exec = sell_exec(tp_lvl)
                    will_exit = True
                elif stop_hit:
                    exit_price_exec = sell_exec(stop_lvl)
                    will_exit = True
                elif tp_hit:
                    exit_price_exec = sell_exec(tp_lvl)
                    will_exit = True

                if not tp1_done and tp1_atr > 0 and pd.notna(atr_val) and current_trade and qty > 0:
                    tp1_level = current_trade.entry_price + tp1_atr * atr_val
                    if price_h >= tp1_level:
                        exec_price_tp1 = sell_exec(tp1_level)
                        part_qty = qty * max(0.0, min(1.0, tp1_pct))
                        if part_qty > 0:
                            gross = part_qty * exec_price_tp1
                            fee = gross * fee_rate
                            cash += max(0.0, gross - fee)
                            qty -= part_qty
                            tp1_done = True
                            if break_even_after_tp1 and current_trade:
                                initial_stop = max(initial_stop or -float("inf"), current_trade.entry_price)
                            log_event("partial_tp", ts,
                                      tp1_price=exec_price_tp1, qty_sold=part_qty, remain_qty=qty)

            elif exit_mode == "sma":
                if pd.notna(sma_s) and pd.notna(sma_l) and (sma_s < sma_l):
                    will_exit = True
            elif exit_mode == "rsi":
                if pd.notna(rsi) and (rsi > rsi_upper) and pd.notna(sma_s) and pd.notna(sma_l) and (sma_s < sma_l):
                    will_exit = True

            if not will_exit and extra_exit:
                will_exit = True

            if will_exit:
                exec_price = exit_price_exec if exit_price_exec is not None else sell_exec(price_c)
                gross = qty * exec_price
                fee = gross * fee_rate
                net = max(0.0, gross - fee)
                cash += net
                sell_count += 1
                position = 0
                if current_trade:
                    pnl_pct = (exec_price / current_trade.entry_price) - 1.0
                    if current_trade.risk_distance and current_trade.risk_distance > 0:
                        r_mult = (exec_price - current_trade.entry_price) / current_trade.risk_distance
                    else:
                        r_mult = None
                    current_trade.exit_ts = ts
                    current_trade.exit_price = exec_price
                    current_trade.pnl_pct = pnl_pct
                    current_trade.r_multiple = r_mult
                    trades.append(current_trade)
                    log_event("exit", ts,
                              price=exec_price,
                              pnl_pct=pnl_pct * 100.0,
                              r_multiple=r_mult,
                              qty_closed=qty)
                qty = 0.0
                current_trade = None
                initial_stop = trail_stop = take_profit = None
                highest_since_entry = None
                tp1_done = False

        equity_curve.append(cash + qty * price_c)

    metrics = compute_metrics(equity_curve, df.index, timeframe)
    extra = extra_trade_metrics(trades)
    metrics.update(extra)

    closed_trades = [t for t in trades if t.exit_ts is not None]
    wins = [t for t in closed_trades if (t.pnl_pct or 0) > 0]
    losses = [t for t in closed_trades if (t.pnl_pct or 0) <= 0]
    win_rate = (len(wins) / len(closed_trades) * 100.0) if closed_trades else 0.0
    avg_win = (np.mean([t.pnl_pct for t in wins]) * 100.0) if wins else 0.0
    avg_loss = (np.mean([t.pnl_pct for t in losses]) * 100.0) if losses else 0.0
    avg_r_multiple = metrics.get("avg_r_multiple", np.nan)

    trades_path = out_dir_path / f"backtest_{out_base}_trades.csv"
    equity_path = out_dir_path / f"backtest_{out_base}_equity.csv"

    if closed_trades:
        pd.DataFrame([{
            "entry_ts": t.entry_ts.isoformat(),
            "entry_price": t.entry_price,
            "exit_ts": t.exit_ts.isoformat() if t.exit_ts else "",
            "exit_price": t.exit_price if t.exit_price else "",
            "pnl_pct": t.pnl_pct * 100.0 if t.pnl_pct is not None else "",
            "initial_stop": t.initial_stop if t.initial_stop is not None else "",
            "risk_distance": t.risk_distance if t.risk_distance is not None else "",
            "r_multiple": t.r_multiple if t.r_multiple is not None else "",
        } for t in closed_trades]).to_csv(trades_path, index=False)

    pd.DataFrame({
        "ts": [ix.isoformat() for ix in df.index],
        "equity": equity_curve
    }).to_csv(equity_path, index=False)

    if jsonl_log and event_buffer:
        with open(events_path, "w", encoding="utf-8") as f:
            for line in event_buffer:
                f.write(line + "\n")

    # Console summary (büyük grid aramalarında kapatmak istersen ileride flag eklenebilir)
    print(f"[SUMMARY] {symbol} {timeframe} bars={len(df)} ret={metrics.get('total_return_pct', np.nan):.2f}% "
          f"sharpe={metrics.get('sharpe', 0.0):.2f} pf={metrics.get('profit_factor', np.nan):.2f}")

    return {
        "params": {
            "symbol": symbol, "timeframe": timeframe, "limit": limit,
            "short": short, "long": long, "rsi_period": rsi_period,
            "rsi_lower": rsi_lower, "rsi_upper": rsi_upper,
            "use_trend": use_trend, "trend_timeframe": trend_timeframe, "trend_sma": trend_sma, "trend_ma_type": trend_ma_type,
            "entry_mode": entry_mode, "exit_mode": exit_mode,
            "atr_period": atr_period, "sl_atr": sl_atr, "tp_atr": tp_atr, "trail_atr": trail_atr,
            "fee_rate": fee_rate, "spread_bps": spread_bps, "slippage_bps": slippage_bps,
            "atr_min_pct": atr_min_pct, "atr_max_pct": atr_max_pct, "min_quote_vol_usd": min_quote_vol_usd,
            "use_stoch": use_stoch, "use_macd": use_macd,
            "confirm_trend_with_momentum": confirm_trend_with_momentum, "momentum_source": momentum_source,
            "risk_per_trade": risk_per_trade, "tp1_atr": tp1_atr, "tp1_pct": tp1_pct, "break_even_after_tp1": break_even_after_tp1,
            "dynamic_atr": dynamic_atr, "atr_period_fast": atr_period_fast, "atr_period_slow": atr_period_slow, "atr_switch_pct": atr_switch_pct,
            "risk_scale_by_atr": risk_scale_by_atr, "atr_target_pct": atr_target_pct, "risk_scale_min": risk_scale_min, "risk_scale_max": risk_scale_max,
            "exit_on_rsi_overbought": exit_on_rsi_overbought, "exit_on_stoch_cross": exit_on_stoch_cross, "exit_on_macd_hist_neg": exit_on_macd_hist_neg,
            "exposure": exposure, "initial_cash": initial_cash,
            "intra_bar_priority": intra_bar_priority,
            "jsonl_log": jsonl_log,
            "start_ts": start_ts.isoformat() if isinstance(start_ts, pd.Timestamp) else (str(start_ts) if start_ts else ""),
            "end_ts": end_ts.isoformat() if isinstance(end_ts, pd.Timestamp) else (str(end_ts) if end_ts else ""),
        },
        "metrics": metrics,
        "counts": {
            "candles": len(df),
            "buy_signals": buy_count, "sell_signals": sell_count,
            "closed_trades": len(closed_trades),
            "win_rate_pct": win_rate, "avg_win_pct": avg_win, "avg_loss_pct": avg_loss,
        },
        "outputs": {
            "trades_path": str(trades_path) if closed_trades else "",
            "equity_path": str(equity_path),
            "events_path": str(events_path) if jsonl_log else "",
        },
    }


# ------------------ Grid & Walk-Forward ------------------ #

def load_param_grid(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        # Already explicit list of dicts
        if not all(isinstance(x, dict) for x in data):
            raise ValueError("Param grid list format must be list of objects (dict).")
        return data
    elif isinstance(data, dict):
        # Cartesian expansion
        keys = list(data.keys())
        values_lists = []
        for k in keys:
            v = data[k]
            if not isinstance(v, list):
                v = [v]
            values_lists.append(v)
        combos = []
        for vals in itertools.product(*values_lists):
            combos.append({k: v for k, v in zip(keys, vals)})
        return combos
    else:
        raise ValueError("Unsupported param grid JSON format.")


def select_metric_value(metrics: Dict[str, Any], metric: str) -> float:
    v = metrics.get(metric, None)
    if v is None:
        return float("nan")
    return float(v)


def grid_search(
    base_df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    base_kwargs: Dict[str, Any],
    param_grid: List[Dict[str, Any]],
    opt_metric: str,
    out_dir: str,
    train_range: Optional[tuple] = None
) -> pd.DataFrame:
    """
    param_grid içindeki her kombinasyon için run_backtest çalıştırır (data_df üzerinden).
    train_range: (start_ts, end_ts) verilirse sadece bu slice üzerinde backtest yapılır.
    """
    records = []
    start_ts, end_ts = None, None
    if train_range:
        start_ts, end_ts = train_range

    for i, override in enumerate(param_grid, 1):
        params = base_kwargs.copy()
        params.update(override)  # override grid values

        # run_backtest çağrısı
        res = run_backtest(
            data_df=base_df,
            symbol=symbol,
            timeframe=timeframe,
            start_ts=start_ts,
            end_ts=end_ts,
            **params
        )
        m = res["metrics"]
        metric_val = select_metric_value(m, opt_metric)
        rec = {
            "combo_index": i,
            "metric": metric_val,
        }
        # Param birleşimi
        for k, v in override.items():
            rec[f"p_{k}"] = v
        # Önemli metrikler
        for k_out in ["total_return_pct", "sharpe", "cagr_pct", "profit_factor", "expectancy_pct",
                      "max_drawdown_pct", "avg_r_multiple"]:
            rec[k_out] = m.get(k_out, np.nan)
        records.append(rec)

    df_grid = pd.DataFrame(records)
    if not df_grid.empty:
        if opt_metric in ("max_drawdown_pct",):
            # drawdown min is better
            df_grid = df_grid.sort_values("metric", ascending=True)
        else:
            df_grid = df_grid.sort_values("metric", ascending=False)
    df_grid.reset_index(drop=True, inplace=True)
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    out_base = f"{symbol_to_filename(symbol)}_{timeframe}"
    opt_path = out_dir_path / f"opt_{out_base}.csv"
    df_grid.to_csv(opt_path, index=False)
    print(f"[GRID] Saved optimization results to {opt_path}")
    return df_grid


# ------------------ CLI / Main ------------------ #

def main():
    p = argparse.ArgumentParser(description="Backtest + Param Grid + Walk-Forward (Step2).")
    p.add_argument("--symbol", default=Settings.SYMBOL)
    p.add_argument("--timeframe", default=Settings.TIMEFRAME)
    p.add_argument("--limit", type=int, default=20000)
    p.add_argument("--short", type=int, default=Settings.SMA_SHORT)
    p.add_argument("--long", type=int, default=Settings.SMA_LONG)
    p.add_argument("--rsi", type=int, default=Settings.RSI_PERIOD)
    p.add_argument("--rsi-lower", type=float, default=Settings.RSI_LOWER)
    p.add_argument("--rsi-upper", type=float, default=Settings.RSI_UPPER)
    p.add_argument("--trend", action="store_true")
    p.add_argument("--trend-timeframe", default=Settings.TREND_TIMEFRAME)
    p.add_argument("--trend-sma", type=int, default=Settings.TREND_SMA)
    p.add_argument("--trend-ma-type", choices=["sma", "ema"], default="sma")
    p.add_argument("--initial-cash", type=float, default=1000.0)
    p.add_argument("--fee", type=float, default=0.0006)
    p.add_argument("--exposure", type=float, default=1.0)
    p.add_argument("--csv", default="")
    p.add_argument("--out-dir", default=str(Path(Settings.DATA_DIR) / "backtests"))

    p.add_argument("--entry-mode", choices=["level", "cross"], default="cross")
    p.add_argument("--exit-mode", choices=["sma", "rsi", "trailing"], default="trailing")
    p.add_argument("--atr-period", type=int, default=14)
    p.add_argument("--sl-atr", type=float, default=1.5)
    p.add_argument("--tp-atr", type=float, default=0.0)
    p.add_argument("--trail-atr", type=float, default=1.5)

    p.add_argument("--spread-bps", type=float, default=2.0)
    p.add_argument("--slippage-bps", type=float, default=3.0)

    p.add_argument("--atr-min-pct", type=float, default=0.0)
    p.add_argument("--atr-max-pct", type=float, default=1e9)
    p.add_argument("--min-quote-vol-usd", type=float, default=0.0)

    p.add_argument("--use-stoch", action="store_true")
    p.add_argument("--stoch-period", type=int, default=14)
    p.add_argument("--stoch-k-smooth", type=int, default=3)
    p.add_argument("--stoch-d-smooth", type=int, default=3)
    p.add_argument("--stoch-max-k", type=float, default=80.0)
    p.add_argument("--use-macd", action="store_true")

    p.add_argument("--confirm-trend-with-momentum", action="store_true")
    p.add_argument("--momentum-source", choices=["macd", "stoch", "both", "either"], default="either")

    p.add_argument("--risk-per-trade", type=float, default=0.0)
    p.add_argument("--tp1-atr", type=float, default=0.0)
    p.add_argument("--tp1-pct", type=float, default=0.5)
    p.add_argument("--break-even-after-tp1", action="store_true")

    p.add_argument("--dynamic-atr", action="store_true")
    p.add_argument("--atr-period-fast", type=int, default=10)
    p.add_argument("--atr-period-slow", type=int, default=20)
    p.add_argument("--atr-switch-pct", type=float, default=1.5)

    p.add_argument("--risk-scale-by-atr", action="store_true")
    p.add_argument("--atr-target-pct", type=float, default=1.0)
    p.add_argument("--risk-scale-min", type=float, default=0.5)
    p.add_argument("--risk-scale-max", type=float, default=1.5)

    p.add_argument("--exit-on-rsi-overbought", action="store_true")
    p.add_argument("--exit-on-stoch-cross", action="store_true")
    p.add_argument("--exit-on-macd-hist-neg", action="store_true")

    p.add_argument("--debug-entry-blocks", action="store_true")

    p.add_argument("--intra-bar-priority", choices=["stop", "tp", "stop_then_tp", "tp_then_stop"], default="stop")
    p.add_argument("--jsonl-log", action="store_true")

    # Step 2: grid & walk-forward
    p.add_argument("--param-grid", default="", help="JSON param grid path.")
    p.add_argument("--opt-metric", choices=["total_return", "sharpe", "cagr_pct", "profit_factor", "expectancy_pct"],
                   default="sharpe")
    p.add_argument("--walk-forward", action="store_true", help="Enable one-split walk-forward.")
    p.add_argument("--wf-train-ratio", type=float, default=0.7)
    p.add_argument("--max-grid", type=int, default=500, help="Maximum allowed param combos to prevent explosion.")

    args = p.parse_args()

    # Baz paramlar (grid override edebilir)
    base_kwargs = dict(
        timeframe=args.timeframe,
        limit=args.limit,
        short=args.short,
        long=args.long,
        rsi_period=args.rsi,
        rsi_lower=args.rsi_lower,
        rsi_upper=args.rsi_upper,
        use_trend=args.trend,
        trend_timeframe=args.trend_timeframe,
        trend_sma=args.trend_sma,
        initial_cash=args.initial_cash,
        fee_rate=args.fee,
        exposure=args.exposure,
        csv_in=(args.csv or None),
        out_dir=args.out_dir,
        entry_mode=args.entry_mode,
        exit_mode=args.exit_mode,
        atr_period=args.atr_period,
        sl_atr=args.sl_atr,
        tp_atr=args.tp_atr,
        trail_atr=args.trail_atr,
        spread_bps=args.spread_bps,
        slippage_bps=args.slippage_bps,
        atr_min_pct=args.atr_min_pct,
        atr_max_pct=args.atr_max_pct,
        min_quote_vol_usd=args.min_quote_vol_usd,
        trend_ma_type=args.trend_ma_type,
        use_stoch=args.use_stoch,
        stoch_period=args.stoch_period,
        stoch_k_smooth=args.stoch_k_smooth,
        stoch_d_smooth=args.stoch_d_smooth,
        stoch_max_k=args.stoch_max_k,
        use_macd=args.use_macd,
        confirm_trend_with_momentum=args.confirm_trend_with_momentum,
        momentum_source=args.momentum_source,
        risk_per_trade=args.risk_per_trade,
        tp1_atr=args.tp1_atr,
        tp1_pct=args.tp1_pct,
        break_even_after_tp1=args.break_even_after_tp1,
        dynamic_atr=args.dynamic_atr,
        atr_period_fast=args.atr_period_fast,
        atr_period_slow=args.atr_period_slow,
        atr_switch_pct=args.atr_switch_pct,
        risk_scale_by_atr=args.risk_scale_by_atr,
        atr_target_pct=args.atr_target_pct,
        risk_scale_min=args.risk_scale_min,
        risk_scale_max=args.risk_scale_max,
        exit_on_rsi_overbought=args.exit_on_rsi_overbought,
        exit_on_stoch_cross=args.exit_on_stoch_cross,
        exit_on_macd_hist_neg=args.exit_on_macd_hist_neg,
        debug_entry_blocks=args.debug_entry_blocks,
        intra_bar_priority=args.intra_bar_priority,
        jsonl_log=args.jsonl_log,
        symbol=args.symbol  # run_backtest requires symbol
    )

    param_grid: List[Dict[str, Any]] = []
    if args.param_grid:
        param_grid = load_param_grid(args.param_grid)
        if len(param_grid) > args.max_grid:
            print(f"[WARN] Grid size {len(param_grid)} exceeds max-grid {args.max_grid}. Truncating.")
            param_grid = param_grid[:args.max_grid]
        print(f"[GRID] Loaded {len(param_grid)} param combinations.")

    # Eğer grid yoksa klasik tek koşum
    if not param_grid and not args.walk_forward:
        res = run_backtest(symbol=args.symbol, **base_kwargs)
        # Tek koşum çıktısı yazdırıldı zaten run_backtest içinde.
        return

    # Veri sadece bir kez hazırlanır (grid/wf için)
    print("[INFO] Preparing base dataframe once...")
    base_df = prepare_base_dataframe(
        symbol=args.symbol, timeframe=args.timeframe, limit=args.limit,
        short=args.short, long=args.long, rsi_period=args.rsi,
        use_trend=args.trend, trend_timeframe=args.trend_timeframe, trend_sma=args.trend_sma,
        trend_ma_type=args.trend_ma_type, dynamic_atr=args.dynamic_atr,
        atr_period=args.atr_period, atr_period_fast=args.atr_period_fast, atr_period_slow=args.atr_period_slow,
        atr_switch_pct=args.atr_switch_pct,
        stoch_period=args.stoch_period, stoch_k_smooth=args.stoch_k_smooth, stoch_d_smooth=args.stoch_d_smooth,
        use_stoch=args.use_stoch, use_macd=args.use_macd,
        csv_in=(args.csv or None)
    )

    if args.walk_forward:
        # Tek split
        n = len(base_df)
        split_idx = int(n * args.wf_train_ratio)
        if split_idx < 10 or n - split_idx < 10:
            raise ValueError("Walk-forward split too small. Adjust wf-train-ratio or provide more data.")
        train_df = base_df.iloc[:split_idx]
        test_df = base_df.iloc[split_idx:]
        train_start, train_end = train_df.index[0], train_df.index[-1]
        test_start, test_end = test_df.index[0], test_df.index[-1]
        print(f"[WF] Train bars: {len(train_df)}, Test bars: {len(test_df)}")

        if not param_grid:
            raise ValueError("Walk-forward requires a param grid (--param-grid).")

        df_grid = grid_search(
            base_df=base_df,
            symbol=args.symbol,
            timeframe=args.timeframe,
            base_kwargs=base_kwargs,
            param_grid=param_grid,
            opt_metric=args.opt_metric,
            out_dir=args.out_dir,
            train_range=(train_start, train_end)
        )
        if df_grid.empty:
            print("[WF] Grid search returned empty results.")
            return

        best_row = df_grid.iloc[0]
        # Seçilen paramları override edip test set koş
        best_params = {}
        for c in df_grid.columns:
            if c.startswith("p_"):
                best_params[c[2:]] = best_row[c]

        print(f"[WF] Best combo (train {args.opt_metric}={best_row['metric']:.4f}): {best_params}")

        # Train sonuçlarını tekrar almak istersen (zaten grid var) → seçilen train metriğini saklıyoruz
        # Test koşumu
        final_kwargs = base_kwargs.copy()
        final_kwargs.update(best_params)
        res_test = run_backtest(
            data_df=base_df,
            symbol=args.symbol,
            timeframe=args.timeframe,
            start_ts=test_start,
            end_ts=test_end,
            **final_kwargs
        )
        # WF özet
        summary = {
            "train": {
                "best_metric": args.opt_metric,
                "best_metric_value": float(best_row["metric"]),
                "best_params": best_params,
                "train_range": [train_start.isoformat(), train_end.isoformat()],
            },
            "test": {
                "range": [test_start.isoformat(), test_end.isoformat()],
                "metrics": res_test["metrics"],
            }
        }
        out_dir_path = Path(args.out_dir)
        out_dir_path.mkdir(parents=True, exist_ok=True)
        wf_path = out_dir_path / f"wf_summary_{symbol_to_filename(args.symbol)}_{args.timeframe}.json"
        with open(wf_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[WF] Summary saved to {wf_path}")
    else:
        # Sadece grid (full dataset)
        _ = grid_search(
            base_df=base_df,
            symbol=args.symbol,
            timeframe=args.timeframe,
            base_kwargs=base_kwargs,
            param_grid=param_grid,
            opt_metric=args.opt_metric,
            out_dir=args.out_dir,
            train_range=None
        )


if __name__ == "__main__":
    main()