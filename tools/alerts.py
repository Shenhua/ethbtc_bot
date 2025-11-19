import json, os, urllib.request
def notify(text: str, webhook: str = None):
    url = webhook or os.getenv("ALERT_WEBHOOK","")
    if not url: return
    data = json.dumps({"text": text}).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type":"application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            r.read()
    except Exception:
        pass
