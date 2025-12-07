"""
tools/alerts.py - Simple Alert Utility

A lightweight utility to send notifications via webhook (e.g. Slack/Discord/Mattermost).
"""
import json, os, urllib.request

def notify(text: str, webhook: str = None):
    """
    Sends a simple text notification to a webhook URL.

    Args:
        text: The message to send.
        webhook: Optional webhook URL override. If None, reads from 'ALERT_WEBHOOK' env var.
    """
    url = webhook or os.getenv("ALERT_WEBHOOK","")
    if not url: return
    data = json.dumps({"text": text}).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type":"application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            r.read()
    except Exception:
        pass