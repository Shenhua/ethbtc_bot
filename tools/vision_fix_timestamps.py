
import argparse, re
import pandas as pd
from datetime import datetime, timezone
import sys
import os

# --- MAGIC PATH FIX ---
# Allow importing 'core' even if running from tools/ folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ----------------------
def parse_start_from_name(name: str):
    # try patterns like: ..._2021-01_2025-12_...
    m = re.search(r'_(\d{4}-\d{2})_(\d{4}-\d{2})_', name)
    if not m: return None, None
    s_ym, e_ym = m.group(1), m.group(2)
    try:
        start = pd.Timestamp(s_ym + "-01T00:00:00Z")
        end   = (pd.Timestamp(e_ym + "-01T00:00:00Z") + pd.offsets.MonthEnd(1) + pd.Timedelta(days=1)).normalize()
        return start, end
    except Exception:
        return None, None

def main():
    ap = argparse.ArgumentParser(description="Fix Binance Vision CSV timestamps that were written with wrong unit scaling.")
    ap.add_argument("--in", dest="inp", required=True, help="Input CSV path")
    ap.add_argument("--out", dest="out", required=True, help="Output CSV path")
    ap.add_argument("--freq", required=True, choices=["15T","30T","1H"], help="Candle frequency: 15T, 30T, or 1H")
    ap.add_argument("--start", help="Start datetime in ISO (UTC). If omitted, try to infer from filename (YYYY-MM).")
    args = ap.parse_args()

    df = pd.read_csv(args.inp)
    cols = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df.columns = cols

    # try to infer the start if not provided
    start = pd.Timestamp(args.start) if args.start else None
    if start is None or pd.isna(start):
        s, e = parse_start_from_name(args.inp)
        if s is not None:
            start = s

    if start is None or pd.isna(start):
        raise SystemExit("Could not infer --start. Please pass --start YYYY-MM-DDTHH:MM:SSZ")

    # Build a clean index with the expected frequency and same length
    n = len(df)
    idx = pd.date_range(start=start, periods=n, freq=args.freq, tz="UTC")
    # Binance close_time is end of interval; set close_time = idx + freq - 1ms; open_time = idx
    if args.freq == "15T":
        delta = pd.Timedelta(minutes=15) - pd.Timedelta(milliseconds=1)
    elif args.freq == "30T":
        delta = pd.Timedelta(minutes=30) - pd.Timedelta(milliseconds=1)
    else:
        delta = pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1)

    df["open_time"] = idx
    df["close_time"] = idx + delta

    # Reorder to standard Binance Vision column order if present
    order = ["open_time","open","high","low","close","volume","close_time","qav","trades","taker_base","taker_quote","ignore"]
    present = [c for c in order if c in df.columns]
    rest = [c for c in df.columns if c not in present]
    df = df[present + rest]

    # Save with ISO8601 (UTC) strings
    df.to_csv(args.out, index=False, date_format="%Y-%m-%d %H:%M:%S.%f%z")
    print(f"Wrote fixed CSV → {args.out}  (rows={n}, freq={args.freq}, start={start})")

if __name__ == "__main__":
    main()
