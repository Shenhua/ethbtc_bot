#!/usr/bin/env python3
import os, json, time
STATE = os.getenv("STATE_FILE","/data/state.json")
INTERVAL = os.getenv("INTERVAL","15m")
units = {"m":60,"h":3600,"d":86400}
iv_sec = int(INTERVAL[:-1]) * units.get(INTERVAL[-1],60)
try:
    with open(STATE,"r",encoding="utf-8") as f:
        st = json.load(f)
    last_ts = int(st.get("last_bar_close",0))
    now = int(time.time())
    if now - last_ts <= 2*iv_sec + 30:
        print("OK"); raise SystemExit(0)
    print("STALE"); raise SystemExit(1)
except Exception as e:
    print("NO_STATE", e); raise SystemExit(1)
