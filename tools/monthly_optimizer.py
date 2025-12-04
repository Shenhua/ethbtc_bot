#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Magic path fix
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from tools.alerts import notify  # uses ALERT_WEBHOOK


STATE_FILE = Path(os.getenv("MONTHLY_OPT_STATE", "/data/monthly_opt_state.json"))


def previous_month_range(now: datetime):
    first_this_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    last_prev_month = first_this_month - timedelta(seconds=1)
    first_prev_month = last_prev_month.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    return first_prev_month, last_prev_month


def already_ran_for(year: int, month: int) -> bool:
    if not STATE_FILE.exists():
        return False
    try:
        import json
        with STATE_FILE.open() as f:
            data = json.load(f)
        return data.get("last_year") == year and data.get("last_month") == month
    except Exception:
        return False


def mark_ran_for(year: int, month: int):
    try:
        import json
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with STATE_FILE.open("w") as f:
            json.dump({"last_year": year, "last_month": month}, f)
    except Exception as e:
        print(f"[monthly_optimizer] Failed to write state: {e}", file=sys.stderr)


def run_once_for_previous_month():
    now = datetime.now(timezone.utc)
    start, end = previous_month_range(now)
    year, month = start.year, start.month

    if already_ran_for(year, month):
        print(f"[monthly_optimizer] Already ran optimization for {year}-{month:02d}.")
        return

    symbol = os.getenv("SYMBOL", "ETHBTC")
    intervals = os.getenv("OPT_INTERVALS", "15m")
    out_dir = os.getenv("RAW_OUT_DIR", "data/raw")

    # 1) Download previous month's data
    vision_cmd = [
        sys.executable, "tools/download_vision.py",
        "--symbol", symbol,
        "--intervals", intervals,
        "--start", start.strftime("%Y-%m"),
        "--end", end.strftime("%Y-%m"),
        "--out-dir", out_dir,
    ]
    print("[monthly_optimizer] Running:", " ".join(vision_cmd))
    vision_res = subprocess.run(vision_cmd, capture_output=True, text=True)
    print(vision_res.stdout)
    if vision_res.returncode != 0:
        msg = f"❌ Monthly optimizer: download_vision failed for {year}-{month:02d}.\n{vision_res.stderr[:4000]}"
        notify(msg)
        return

    # 2) Run your existing optimisation pipeline
    # We assume run_complete_optimization.sh writes a suggested config, e.g. configs/prod_meta_live_suggested.json
    opt_script = os.getenv("RUN_COMPLETE_SCRIPT", "run_complete_optimization.sh")
    opt_cmd = ["bash", opt_script]
    print("[monthly_optimizer] Running:", " ".join(opt_cmd))
    opt_res = subprocess.run(opt_cmd, capture_output=True, text=True)
    print(opt_res.stdout)

    if opt_res.returncode != 0:
        msg = f"❌ Monthly optimizer: {opt_script} failed.\n{opt_res.stderr[:4000]}"
        notify(msg)
        return

    # 3) Notify with suggestion location
    suggested_cfg = os.getenv("SUGGESTED_CONFIG", "configs/prod_meta_live_suggested.json")
    cfg_path = ROOT / suggested_cfg

    if cfg_path.exists():
        msg = (
            f"📊 **Monthly Optimization Complete** for {year}-{month:02d}\n"
            f"• Symbol: {symbol}\n"
            f"• Intervals: {intervals}\n"
            f"• Suggested config: `{suggested_cfg}`\n\n"
            f"Please review & promote manually to prod_meta_live.json when satisfied."
        )
    else:
        msg = (
            f"⚠️ Monthly optimization finished but `{suggested_cfg}` not found.\n"
            f"Check logs of {opt_script} for details."
        )

    notify(msg)
    mark_ran_for(year, month)


def main():
    # Simple scheduler: wake up every 6 hours, run for previous month if today is 1st and not done yet
    sleep_sec = float(os.getenv("MONTHLY_OPT_POLL_SEC", 6 * 60 * 60))

    while True:
        now = datetime.now(timezone.utc)
        if now.day == 1:
            try:
                run_once_for_previous_month()
            except Exception as e:
                print(f"[monthly_optimizer] ERROR: {e}", file=sys.stderr)
        else:
            print(f"[monthly_optimizer] Not 1st of month (today={now.date()}), sleeping...")

        time.sleep(sleep_sec)


if __name__ == "__main__":
    main()