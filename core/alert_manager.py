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
        try:
            requests.post(self.discord_url, json=payload, timeout=3)
        except Exception as e:
            self.logger.error(f"Failed to send Discord alert: {e}")

    def _send_telegram(self, text):
        url = f"https://api.telegram.org/bot{self.tg_token}/sendMessage"
        payload = {"chat_id": self.tg_chat_id, "text": text}
        try:
            requests.post(url, json=payload, timeout=3)
        except Exception as e:
            self.logger.error(f"Failed to send Telegram alert: {e}")