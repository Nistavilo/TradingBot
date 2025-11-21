import json
from pathlib import Path
from typing import Any, Dict

DEFAULT_STATE = {
    "last_signal": "NONE",
    "last_closed_ts": None,
    "last_signal_at": None  # ISO datetime of last sent signal
}

class StateStore:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data: Dict[str, Any] = DEFAULT_STATE.copy()
        self.load()

    def load(self):
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                self.data = DEFAULT_STATE.copy()

    def save(self):
        try:
            self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    @property
    def last_signal(self) -> str:
        return self.data.get("last_signal", "NONE")

    @last_signal.setter
    def last_signal(self, val: str):
        self.data["last_signal"] = val

    @property
    def last_closed_ts(self):
        return self.data.get("last_closed_ts", None)

    @last_closed_ts.setter
    def last_closed_ts(self, val):
        self.data["last_closed_ts"] = val

    @property
    def last_signal_at(self):
        return self.data.get("last_signal_at", None)

    @last_signal_at.setter
    def last_signal_at(self, val):
        self.data["last_signal_at"] = val