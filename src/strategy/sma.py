import pandas as pd

def add_sma_signals(df: pd.DataFrame, short: int = 10, long: int = 30) -> pd.DataFrame:
    out = df.copy()
    out["sma_short"] = out["close"].rolling(window=short, min_periods=short).mean()
    out["sma_long"]  = out["close"].rolling(window=long,  min_periods=long).mean()
    out["signal"] = 0
    mask = (out["sma_short"].notna()) & (out["sma_long"].notna())
    out.loc[mask & (out["sma_short"] > out["sma_long"]), "signal"] = 1
    out.loc[mask & (out["sma_short"] < out["sma_long"]), "signal"] = -1
    out["position"] = out["signal"].shift(1).fillna(0)  
    return out