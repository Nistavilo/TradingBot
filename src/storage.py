from pathlib import Path
import os
import csv
from datetime import datetime, timezone
from typing import Dict, List

FIELDS: List[str] = [
    "timestamp","open","high","low","close","volume",
    "sma_short","sma_long","rsi","signal"
]

def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def symbol_to_filename(symbol: str) -> str:
    return symbol.replace("/", "_")

def csv_path(data_dir: str, symbol: str, timeframe: str) -> str:
    ensure_dir(data_dir)
    name = f"{symbol_to_filename(symbol)}_{timeframe}.csv"
    return str(Path(data_dir) / name)

def append_row(csv_file: str, row: Dict):
    file_exists = os.path.exists(csv_file)
    with open(csv_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in FIELDS})
        f.flush()
        os.fsync(f.fileno())

def _count_rows(csv_file: str) -> int:
    if not os.path.exists(csv_file):
        return 0
    with open(csv_file, "r", encoding="utf-8") as f:
        return sum(1 for _ in f) - 1  # header hariç

def maybe_rotate_csv(csv_file: str, max_rows: int):
    if max_rows <= 0 or not os.path.exists(csv_file):
        return
    try:
        rows = _count_rows(csv_file)
        if rows > max_rows:
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            p = Path(csv_file)
            rotated = p.with_name(f"{p.stem}.{ts}.csv")
            p.rename(rotated)
            # yeni dosya header ile başlasın
            with open(csv_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDS)
                writer.writeheader()
    except Exception:
        # sessizce geç
        pass