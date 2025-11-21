from typing import Literal
import pandas as pd

Signal = Literal["AL", "SAT", "NONE"]

def evaluate_signal(row: pd.Series) -> Signal:
    try:
        sma_s = float(row.get("sma_short"))
        sma_l = float(row.get("sma_long"))
        rsi   = float(row.get("rsi"))
    except Exception:
        return "NONE"

    if pd.isna(sma_s) or pd.isna(sma_l) or pd.isna(rsi):
        return "NONE"

    # Koşullar
    if (rsi < 30) and (sma_s > sma_l):
        return "AL"
    if (rsi > 70) and (sma_s < sma_l):
        return "SAT"
    return "NONE"