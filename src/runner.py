import argparse
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

from src.config import Settings
from src.exchange import make_exchange, fetch_ohlcv_df, append_latest_candle
from src.indicators import compute_indicators
from src.signals import evaluate_signal
from src.state import StateStore
from src.storage import csv_path, append_row, maybe_rotate_csv
from src.telegram_client import TelegramClient
from src.logging_utils import setup_logger

def format_price(p: float) -> str:
    if p >= 1000:
        return f"{p:,.0f}$".replace(",", ".")
    elif p >= 1:
        return f"{p:,.2f}$".replace(",", ".")
    else:
        return f"{p:,.6f}$".replace(",", ".")

def get_tz():
    # TIMEZONE ayarını uygula, yoksa UTC
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(Settings.TIMEZONE)
    except Exception:
        return timezone.utc

def write_healthcheck(path: str):
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"OK {datetime.now(timezone.utc).isoformat()}")
    except Exception:
        pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default=Settings.SYMBOL)
    parser.add_argument("--timeframe", default=Settings.TIMEFRAME)
    parser.add_argument("--short", type=int, default=Settings.SMA_SHORT)
    parser.add_argument("--long", type=int, default=Settings.SMA_LONG)
    parser.add_argument("--rsi", type=int, default=Settings.RSI_PERIOD)
    parser.add_argument("--limit", type=int, default=Settings.LIMIT)
    parser.add_argument("--poll", type=int, default=Settings.POLL_SECONDS)
    args = parser.parse_args()

    logger = setup_logger(Settings.DATA_DIR, Settings.LOG_LEVEL)
    tg = TelegramClient(
        Settings.TELEGRAM_BOT_TOKEN,
        Settings.TELEGRAM_CHAT_ID,
        retries=Settings.TELEGRAM_RETRY,
        retry_delay=Settings.TELEGRAM_RETRY_DELAY,
    )
    state = StateStore(path=f"{Settings.DATA_DIR}/state.json")
    tz = get_tz()

    try:
        ex = make_exchange(Settings.EXCHANGE)
    except Exception as e:
        logger.error(f"Borsa bağlanma hatası: {e}")
        tg.send(f"Hata: Borsa bağlanamadı: {e}")
        return

    symbol = args.symbol
    timeframe = args.timeframe
    short = args.short
    long = args.long
    rsi_p = args.rsi
    limit = args.limit
    poll = args.poll

    rsi_lower = Settings.RSI_LOWER
    rsi_upper = Settings.RSI_UPPER

    # Trend filtresi için üst zaman dilimi verisi
    htf_df = None
    htf_last_index = None
    if Settings.TREND_FILTER:
        try:
            htf_df = fetch_ohlcv_df(ex, symbol, Settings.TREND_TIMEFRAME, limit=max(200, Settings.TREND_SMA + 50))
            htf_df["sma_trend"] = htf_df["close"].rolling(window=Settings.TREND_SMA, min_periods=Settings.TREND_SMA).mean()
            htf_last_index = htf_df.index[-1] if len(htf_df) else None
            logger.info(f"Trend filtresi aktif: {Settings.TREND_TIMEFRAME} SMA({Settings.TREND_SMA})")
        except Exception as e:
            logger.warning(f"Trend filtresi başlatılamadı: {e}")
            htf_df = None

    logger.info(f"Bot başlıyor | {symbol} {timeframe} | SMA({short},{long}) RSI({rsi_p}) [{rsi_lower}/{rsi_upper}]")
    tg.send(f"Bot başladı: {symbol} {timeframe} | SMA({short},{long}) RSI({rsi_p})")

    # İlk veri
    df = fetch_ohlcv_df(ex, symbol, timeframe, limit=limit)
    df = compute_indicators(df, short, long, rsi_p)

    if len(df) < max(short, long, rsi_p):
        logger.warning("Yeterli veri yok, bekleniyor...")
    last_index = df.index[-1] if len(df) else None

    csv_file = csv_path(Settings.DATA_DIR, symbol, timeframe)
    last_status_at = datetime.now(timezone.utc) - timedelta(minutes=Settings.STATUS_INTERVAL_MIN - 1)
    last_health_at = datetime.now(timezone.utc)

    write_healthcheck(Settings.HEALTHCHECK_FILE)

    while True:
        try:
            # Veriyi güncelle
            df = append_latest_candle(df, ex, symbol, timeframe)
            df = compute_indicators(df, short, long, rsi_p)

            # Trend filtresi için üst zaman dilimi mumu güncelle
            if Settings.TREND_FILTER and htf_df is not None:
                try:
                    htf_df = append_latest_candle(htf_df, ex, symbol, Settings.TREND_TIMEFRAME)
                    if "sma_trend" not in htf_df.columns:
                        htf_df["sma_trend"] = htf_df["close"].rolling(window=Settings.TREND_SMA, min_periods=Settings.TREND_SMA).mean()
                    else:
                        # son satır için yeniden hesaplamak yeterli
                        htf_df["sma_trend"] = htf_df["close"].rolling(window=Settings.TREND_SMA, min_periods=Settings.TREND_SMA).mean()
                except Exception as e:
                    logger.warning(f"Trend verisi güncellenemedi: {e}")

            if len(df) == 0:
                time.sleep(poll)
                continue

            current_last = df.index[-1]

            # Yeni mum başladıysa -> önceki mum KAPANDI
            if (last_index is not None) and (current_last != last_index):
                closed_ts = last_index
                row = df.loc[closed_ts]

                price = float(row["close"])

                # RSI eşiklerini satır üzerinde uygula (signals.evaluate_signal temel kuralı veriyor)
                sig = evaluate_signal(row)

                # Trend filtresi uygula (opsiyonel)
                if Settings.TREND_FILTER and htf_df is not None and len(htf_df) > 0:
                    htf_row = htf_df.iloc[-1]
                    htf_close = float(htf_row["close"])
                    htf_sma = float(htf_row.get("sma_trend")) if not pd.isna(htf_row.get("sma_trend")) else None
                    if htf_sma is not None:
                        trend_up = htf_close > htf_sma
                        trend_down = htf_close < htf_sma
                        if sig == "AL" and not trend_up:
                            sig = "NONE"
                        if sig == "SAT" and not trend_down:
                            sig = "NONE"

                # CSV'ye yaz
                append_row(csv_file, {
                    "timestamp": closed_ts.isoformat(),
                    "open": row.get("open",""),
                    "high": row.get("high",""),
                    "low": row.get("low",""),
                    "close": row.get("close",""),
                    "volume": row.get("volume",""),
                    "sma_short": row.get("sma_short",""),
                    "sma_long": row.get("sma_long",""),
                    "rsi": row.get("rsi",""),
                    "signal": sig
                })
                maybe_rotate_csv(csv_file, Settings.MAX_CSV_ROWS)

                # Sinyal gönder (cooldown + tekrarı engelle)
                now_utc = datetime.now(timezone.utc)
                can_send = True
                if sig in ("AL", "SAT"):
                    # Aynı sinyalin cooldown süresi içinde tekrarını gönderme
                    if state.last_signal == sig and state.last_signal_at:
                        try:
                            last_at = datetime.fromisoformat(state.last_signal_at)
                        except Exception:
                            last_at = None
                        if last_at:
                            delta_min = (now_utc - last_at).total_seconds() / 60.0
                            if delta_min < Settings.SIGNAL_COOLDOWN_MIN:
                                can_send = False
                    if can_send:
                        dt_local = closed_ts.astimezone(tz)
                        msg = f"{sig} sinyali: {symbol} {format_price(price)} | {dt_local.strftime('%Y-%m-%d %H:%M')} {Settings.TIMEZONE}"
                        tg.send(msg)
                        logger.info(msg)
                        state.last_signal = sig
                        state.last_signal_at = now_utc.isoformat()

                # State'i güncelle
                state.last_closed_ts = closed_ts.isoformat()
                state.save()

                # Durum mesajı (heartbeat)
                if Settings.SEND_STATUS and (now_utc - last_status_at) >= timedelta(minutes=Settings.STATUS_INTERVAL_MIN):
                    tg.send(f"Durum: {symbol} {timeframe} | Son fiyat {format_price(price)} | Son sinyal: {state.last_signal}")
                    last_status_at = now_utc

                # Sağlık dosyası per closed candle
                write_healthcheck(Settings.HEALTHCHECK_FILE)
                last_health_at = now_utc

                # Bir sonraki kıyas için
                last_index = current_last

            elif last_index is None:
                last_index = current_last

            # Periyodik healthcheck (mumu beklerken de güncelle)
            now_utc = datetime.now(timezone.utc)
            if (now_utc - last_health_at) >= timedelta(minutes=1):
                write_healthcheck(Settings.HEALTHCHECK_FILE)
                last_health_at = now_utc

            time.sleep(poll)

        except KeyboardInterrupt:
            logger.info("Durduruldu (CTRL+C).")
            tg.send("Bot durduruldu.")
            break
        except Exception as e:
            logger.error(f"Hata: {e}")
            tg.send(f"Hata: {e}")
            time.sleep(max(5, poll))

if __name__ == "__main__":
    main()