import os
import logging
import requests
import json
from datetime import datetime

class AlertManager:
    def __init__(self, prefix="BOT"):
        self.prefix = prefix
        self.discord_url = os.getenv("DISCORD_WEBHOOK_URL", "")
        self.tg_token = os.getenv("TELEGRAM_TOKEN", "")
        self.tg_chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        self.logger = logging.getLogger("alerts")

    def send(self, message: str, level: str = "INFO"):
        """
        Sends an alert to all configured channels.
        Levels: INFO, WARNING, ERROR, CRITICAL
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_msg = f"[{level}] {self.prefix} ({timestamp}): {message}"
        
        # 1. Discord Webhook
        if self.discord_url:
            self._send_discord(formatted_msg, level)

        # 2. Telegram Bot API
        if self.tg_token and self.tg_chat_id:
            self._send_telegram(formatted_msg)

    def _send_discord(self, text, level):
        colors = {
            "INFO": 3447003,      # Blue
            "WARNING": 16776960,  # Yellow
            "ERROR": 15158332,    # Red
            "CRITICAL": 10038562  # Dark Red
        }
        payload = {
            "username": f"{self.prefix} Alert",
            "embeds": [{
                "description": text,
                "color": colors.get(level, 3447003)
            }]
        }
        for attempt in range(3):
            try:
                resp = requests.post(self.discord_url, json=payload, timeout=5)
                if resp.status_code in [200, 204]:
                    return
                elif resp.status_code == 429: # Rate limit
                    time.sleep(2)
            except Exception as e:
                self.logger.warning(f"Discord Alert Failed (Attempt {attempt+1}): {e}")
                time.sleep(1)
        
        self.logger.error("Failed to send Discord alert after retries.")

    def _send_telegram(self, text):
        url = f"https://api.telegram.org/bot{self.tg_token}/sendMessage"
        payload = {"chat_id": self.tg_chat_id, "text": text}
        for attempt in range(3):
            try:
                requests.post(url, json=payload, timeout=5)
                return
            except Exception:
                time.sleep(1)