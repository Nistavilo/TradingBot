import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler

def setup_logger(data_dir: str, level: str = "INFO") -> logging.Logger:
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("bot")
    if logger.handlers:
        return logger
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = RotatingFileHandler(str(Path(data_dir) / "bot.log"), maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger