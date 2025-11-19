import os
from dotenv import load_dotenv

load_dotenv()

def _get_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")

def _get_int(name: str, default: int) -> int:
    v = os.getenv(name)
    try:
        return int(v) if v is not None else default
    except:
        return default

def _get_float(name: str, default: float) -> float:
    v = os.getenv(name)
    try:
        return float(v) if v is not None else default
    except:
        return default

class Settings:
    EXCHANGE = os.getenv("EXCHANGE", "binance")
    SYMBOL = os.getenv("SYMBOL", "BTC/USDT")
    TIMEFRAME = os.getenv("TIMEFRAME", "1m")
    LIMIT = _get_int("LIMIT", 500)
    POLL_SECONDS = _get_int("POLL_SECONDS", 10)

    SMA_SHORT = _get_int("SMA_SHORT", 20)
    SMA_LONG = _get_int("SMA_LONG", 50)
    RSI_PERIOD = _get_int("RSI_PERIOD", 14)
    RSI_LOWER = _get_float("RSI_LOWER", 30.0)
    RSI_UPPER = _get_float("RSI_UPPER", 70.0)

    SIGNAL_COOLDOWN_MIN = _get_int("SIGNAL_COOLDOWN_MIN", 15)

    TREND_FILTER = _get_bool("TREND_FILTER", False)
    TREND_TIMEFRAME = os.getenv("TREND_TIMEFRAME", "15m")
    TREND_SMA = _get_int("TREND_SMA", 100)

    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
    TELEGRAM_RETRY = _get_int("TELEGRAM_RETRY", 3)
    TELEGRAM_RETRY_DELAY = _get_float("TELEGRAM_RETRY_DELAY", 3.0)

    DATA_DIR = os.getenv("DATA_DIR", "./data")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    SEND_STATUS = _get_bool("SEND_STATUS", True)
    STATUS_INTERVAL_MIN = _get_int("STATUS_INTERVAL_MIN", 360)
    MAX_CSV_ROWS = _get_int("MAX_CSV_ROWS", 200000)

    HEALTHCHECK_FILE = os.getenv("HEALTHCHECK_FILE", "./data/health.txt")
    TIMEZONE = os.getenv("TIMEZONE", "UTC")