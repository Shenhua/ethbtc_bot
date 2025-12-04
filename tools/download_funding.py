#!/usr/bin/env python3
"""
download_funding.py — Downloads historical Funding Rates from Binance Futures.
Usage:
  python download_funding.py --symbol ETHUSDT --start 2023-01-01 --end 2024-12-31
"""

import argparse
import time
import pandas as pd
import requests
from datetime import datetime, timezone
import sys
import os

# --- MAGIC PATH FIX ---
# Allow importing 'core' even if running from tools/ folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ----------------------
API_URL = "https://fapi.binance.com/fapi/v1/fundingRate"

def get_timestamp(dt_str):
    """Convert YYYY-MM-DD to milliseconds timestamp"""
    dt = pd.to_datetime(dt_str, utc=True)
    return int(dt.timestamp() * 1000)

def fetch_chunk(symbol, start_ts, end_ts):
    """Fetch one chunk of 1000 records"""
    params = {
        "symbol": symbol,
        "startTime": start_ts,
        "endTime": end_ts,
        "limit": 1000
    }
    try:
        resp = requests.get(API_URL, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"Error fetching chunk {start_ts}: {e}")
        time.sleep(5)
        return []

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True, help="Futures Symbol (e.g. ETHUSDT, BNBUSDT)")
    ap.add_argument("--start", required=True, help="Start Date (YYYY-MM-DD)")
    ap.add_argument("--end", required=True, help="End Date (YYYY-MM-DD)")
    ap.add_argument("--out",default="",help="Output CSV path (default: data/raw/{symbol}_funding_{start}_{end}.csv)")
    args = ap.parse_args()

    out_dir = "data/raw"
    os.makedirs(out_dir, exist_ok=True)

    if not args.out:
        fname = f"{args.symbol}_funding_{args.start}_{args.end}_funding.csv"
        args.out = os.path.join(out_dir, fname)

    df.to_csv(args.out, index=False)
    
    start_ms = get_timestamp(args.start)
    end_ms = get_timestamp(args.end)
    
    print(f"Downloading Funding Rates for {args.symbol}...")
    print(f"Range: {args.start} to {args.end}")

    all_data = []
    current_start = start_ms
    
    while True:
        chunk = fetch_chunk(args.symbol, current_start, end_ms)
        
        if not chunk:
            break
            
        all_data.extend(chunk)
        last_ts = chunk[-1]["fundingTime"]
        
        print(f"Fetched {len(chunk)} records. Last: {pd.to_datetime(last_ts, unit='ms')}")
        
        # Advance cursor
        current_start = last_ts + 1
        
        if current_start >= end_ms:
            break
            
        # Be nice to the API
        time.sleep(0.2)

    if not all_data:
        print("No data found!")
        return

    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    # Keep relevant columns
    df = df[["fundingTime", "fundingRate"]]
    # Rename for compatibility
    df.columns = ["time", "rate"]
    
    # Format types
    df["rate"] = df["rate"].astype(float) * 100.0 # Convert to % immediately (e.g. 0.01)
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    
    # Deduplicate and Sort
    df = df.drop_duplicates(subset="time").sort_values("time")
    
    # Save
    df.to_csv(args.out, index=False)
    print(f"Success! Saved {len(df)} rows to {args.out}")

if __name__ == "__main__":
    main()