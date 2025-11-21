import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import os
from pathlib import Path
import pandas as pd
import streamlit as st
from datetime import datetime, timezone
from src.config import Settings
from src.storage import csv_path

st.set_page_config(page_title="Sinyal Botu Dashboard", layout="wide")

symbol = Settings.SYMBOL
timeframe = Settings.TIMEFRAME
data_file = csv_path(Settings.DATA_DIR, symbol, timeframe)

st.title("Sinyal Botu Dashboard")
st.caption(f"Sembol: {symbol} | Zaman Dilimi: {timeframe}")
st.write(f"Veri dosyası: `{data_file}`")

# Healthcheck
hc_file = Settings.HEALTHCHECK_FILE
if Path(hc_file).exists():
    txt = Path(hc_file).read_text(encoding="utf-8").strip()
    st.info(f"Health: {txt}")
else:
    st.warning("Healthcheck dosyası bulunamadı.")

if not Path(data_file).exists():
    st.warning("CSV henüz oluşmadı. Botun bir süre çalışmasına izin verin.")
    st.stop()

df = pd.read_csv(data_file)
if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"])
df.sort_values("timestamp", inplace=True)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Kayıtlı Mum", f"{len(df)}")
col2.metric("AL Sayısı", f"{(df['signal']=='AL').sum()}")
col3.metric("SAT Sayısı", f"{(df['signal']=='SAT').sum()}")
col4.metric("Son Sinyal", df["signal"].iloc[-2] if len(df) > 1 else "N/A")

st.subheader("Son 50 Kayıt")
st.dataframe(df.tail(50), use_container_width=True, height=400)

st.subheader("Fiyat ve SMA'lar")
plot_df = df.tail(1000).copy()
st.line_chart(plot_df.set_index("timestamp")[["close","sma_short","sma_long"]])

st.subheader("RSI")
st.line_chart(plot_df.set_index("timestamp")[["rsi"]])