import os, sys, argparse, io, time, zipfile, hashlib
from datetime import datetime, timedelta
from urllib.parse import urljoin
import requests
import pandas as pd

# --- MAGIC PATH FIX ---
# Allow importing 'core' even if running from tools/ folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ----------------------
BASE = "https://data.binance.vision/"

def month_range(start: datetime, end: datetime):
    cur = datetime(start.year, start.month, 1)
    while cur <= end:
        yield cur
        # next month
        if cur.month == 12:
            cur = datetime(cur.year+1, 1, 1)
        else:
            cur = datetime(cur.year, cur.month+1, 1)

def day_range(start: datetime, end: datetime):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)

def url_monthly(symbol, interval, y, m):
    mm = f"{m:02d}"
    fname = f"{symbol}-{interval}-{y}-{mm}.zip"
    path = f"data/spot/monthly/klines/{symbol}/{interval}/{fname}"
    return urljoin(BASE, path), fname

def url_daily(symbol, interval, y, m, d):
    dd = f"{d:02d}"
    mm = f"{m:02d}"
    fname = f"{symbol}-{interval}-{y}-{mm}-{dd}.zip"
    path = f"data/spot/daily/klines/{symbol}/{interval}/{fname}"
    return urljoin(BASE, path), fname

def head_ok(url):
    try:
        r = requests.head(url, timeout=10)
        return r.status_code == 200
    except Exception:
        return False

def download(url):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.content

def verify_checksum(zip_bytes, chk_bytes):
    # Binance provides .CHECKSUM with "sha256  filename.zip"
    try:
        expected = chk_bytes.decode("utf-8").strip().split()[0]
        h = hashlib.sha256()
        h.update(zip_bytes)
        return h.hexdigest() == expected
    except Exception:
        return None  # unknown
 
def load_zip_csvs(content):
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        dfs = []
        for name in z.namelist():
            if not name.endswith(".csv"): continue
            with z.open(name) as f:
                df = pd.read_csv(f, header=None)
                df.columns = ["open_time","open","high","low","close","volume","close_time","qav","trades","taker_base","taker_quote","ignore"]
                dfs.append(df)
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        return pd.DataFrame(columns=["open_time","open","high","low","close","volume","close_time","qav","trades","taker_base","taker_quote","ignore"])

def maybe_to_datetime(df):
    # robust per-row handling: if >1e13 assume microseconds else milliseconds
    import numpy as np
    ot = pd.to_numeric(df["open_time"], errors="coerce")
    ct = pd.to_numeric(df["close_time"], errors="coerce")

    ms_mask_o = ot <= 10_000_000_000_000
    ms_mask_c = ct <= 10_000_000_000_000

    # normalize to milliseconds
    ot_ms = np.where(ms_mask_o, ot, ot / 1000.0)
    ct_ms = np.where(ms_mask_c, ct, ct / 1000.0)

    df["open_time"]  = pd.to_datetime(ot_ms, unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(ct_ms, unit="ms", utc=True)
    return df


def main():
    ap = argparse.ArgumentParser(description="Download Binance Vision klines (monthly with daily fallback) and merge to CSV.")
    ap.add_argument("--symbol", default=os.getenv("SYMBOL","ETHBTC"))
    ap.add_argument("--intervals", default="1m,15m,30m", help="Comma-separated intervals")
    ap.add_argument("--start", required=True, help="YYYY-MM or YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM or YYYY-MM-DD")
    ap.add_argument("--out-dir", default="vision_data")
    ap.add_argument("--prefer-daily", action="store_true", help="Use daily files first (default monthly first)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    intervals = [x.strip() for x in args.intervals.split(",") if x.strip()]

    # Parse dates
    def parse_date(s):
        parts = s.split("-")
        if len(parts)==2:  # YYYY-MM
            return datetime(int(parts[0]), int(parts[1]), 1)
        return datetime.fromisoformat(s)
    start = parse_date(args.start)
    end = parse_date(args.end)
    if end.day != 1 and args.start.count("-")==1 and args.end.count("-")==1:
        # If both are YYYY-MM, set end to last day of month
        if end.month == 12:
            end = end.replace(day=31)
        else:
            nxt = datetime(end.year + (1 if end.month==12 else 0), 1 if end.month==12 else end.month+1, 1)
            end = nxt - timedelta(days=1)

    for interval in intervals:
        print(f"== {args.symbol} {interval} ==")
        all_frames = []
        downloaded_months = set()

        if not args.prefer_daily:
            # --- OPTIMIZED DEFAULT PATH (Monthly First) ---
            
            # 1. Try monthly first
            for mdt in month_range(start, end):
                url, fname = url_monthly(args.symbol, interval, mdt.year, mdt.month)
                if not head_ok(url): 
                    continue
                print(f"Downloading monthly {fname}")
                zbytes = download(url)
                
                # Checksum (Optional)
                chk_url = url + ".CHECKSUM"
                chk = None
                try:
                    chk = download(chk_url)
                except Exception:
                    pass
                ok = verify_checksum(zbytes, chk) if chk else None
                if ok is False:
                    print(f"WARNING: checksum failed for {fname}")
                
                df = load_zip_csvs(zbytes)
                all_frames.append(df)
                
                # Mark this month as done so we don't re-download days
                downloaded_months.add((mdt.year, mdt.month))

            # 2. Fill gaps with daily
            for ddt in day_range(start, end):
                # NEW: Skip if we already have the full month
                if (ddt.year, ddt.month) in downloaded_months:
                    continue
                
                url, fname = url_daily(args.symbol, interval, ddt.year, ddt.month, ddt.day)
                if not head_ok(url): 
                    continue
                print(f"Downloading daily {fname}")
                zbytes = download(url)
                df = load_zip_csvs(zbytes)
                all_frames.append(df)

        else:
            # --- ALTERNATIVE PATH (--prefer-daily) ---
            # Keeps original logic: try daily first, then monthly fallback
            
            # 1. Daily first
            for ddt in day_range(start, end):
                url, fname = url_daily(args.symbol, interval, ddt.year, ddt.month, ddt.day)
                if not head_ok(url): 
                    continue
                print(f"Downloading daily {fname}")
                zbytes = download(url)
                df = load_zip_csvs(zbytes)
                all_frames.append(df)
            
            # 2. Then monthly for any remaining (simplified fallback)
            for mdt in month_range(start, end):
                url, fname = url_monthly(args.symbol, interval, mdt.year, mdt.month)
                if not head_ok(url): 
                    continue
                print(f"Downloading monthly {fname}")
                zbytes = download(url)
                df = load_zip_csvs(zbytes)
                all_frames.append(df)

        if not all_frames:
            print(f"No data found for {args.symbol} {interval} in range {args.start}..{args.end}")
            continue

        if not all_frames:
            print(f"No data found for {args.symbol} {interval} in range {args.start}..{args.end}")
            continue

        merged = pd.concat(all_frames, ignore_index=True).drop_duplicates(subset=["open_time"]).sort_values("open_time")
        merged["open_time"]  = pd.to_numeric(merged["open_time"], errors="coerce").astype("Int64")
        merged["close_time"] = pd.to_numeric(merged["close_time"], errors="coerce").astype("Int64")
        merged = maybe_to_datetime(merged)
        out_path = os.path.join(args.out_dir, f"{args.symbol}_{interval}_{args.start}_{args.end}_vision.csv")
        merged.to_csv(out_path, index=False, date_format="%Y-%m-%d %H:%M:%S.%f%z")
        print(f"Saved merged CSV → {out_path} ({len(merged)} rows)")

if __name__ == "__main__":
    main()
