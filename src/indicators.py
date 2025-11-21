import pandas as pd
from ta.trend import SMAIndicator
from ta.momentum import RSIIndicator

def compute_indicators(df: pd.DataFrame, sma_short: int, sma_long: int, rsi_period: int) -> pd.DataFrame:
    out = df.copy()
    if len(out) == 0:
        return out
    out["sma_short"] = SMAIndicator(close=out["close"], window=sma_short, fillna=False).sma_indicator()
    out["sma_long"]  = SMAIndicator(close=out["close"], window=sma_long,  fillna=False).sma_indicator()
    out["rsi"]       = RSIIndicator(close=out["close"], window=rsi_period, fillna=False).rsi()
    return out