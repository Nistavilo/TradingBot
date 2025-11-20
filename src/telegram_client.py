import logging
import time
import requests

class TelegramClient:
    def __init__(self, token: str, chat_id: str, retries: int = 3, retry_delay: float = 3.0):
        self.token = token or ""
        self.chat_id = chat_id or ""
        self.base_url = f"https://api.telegram.org/bot{self.token}"
        self.enabled = bool(self.token and self.chat_id)
        self.retries = max(0, int(retries))
        self.retry_delay = max(0.5, float(retry_delay))

    def send(self, text: str):
        logger = logging.getLogger("bot")
        if not self.enabled:
            logger.warning("Telegram kapalı: token/chat_id eksik")
            return
        url = f"{self.base_url}/sendMessage"
        payload = {"chat_id": self.chat_id, "text": text}
        for attempt in range(1, self.retries + 1):
            try:
                r = requests.post(url, data=payload, timeout=10)
                if r.status_code == 200:
                    return
                else:
                    logger.warning(f"Telegram status {r.status_code}: {r.text}")
            except Exception as e:
                logger.error(f"Telegram hata (deneme {attempt}/{self.retries}): {e}")
            time.sleep(self.retry_delay)