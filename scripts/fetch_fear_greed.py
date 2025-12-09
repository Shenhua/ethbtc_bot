#!/usr/bin/env python3
"""
Fetch Historical Fear & Greed Index from alternative.me API.

This script downloads the Crypto Fear & Greed Index data for use in
ML-based regime detection. The index is updated daily.

Usage:
    python scripts/fetch_fear_greed.py --output data/raw/fear_greed_index_2021-2025.csv
    python scripts/fetch_fear_greed.py --days 1500 --output data/raw/fear_greed_index.csv

API Documentation: https://alternative.me/crypto/fear-and-greed-index/
"""

import argparse
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime


def fetch_fear_greed(days: int = 1500) -> pd.DataFrame:
    """
    Fetch Fear & Greed Index from alternative.me API.
    
    Args:
        days: Number of days of history to fetch (max ~1500 available)
    
    Returns:
        DataFrame with columns: value, value_classification
        Index: timestamp (UTC datetime)
    """
    url = f"https://api.alternative.me/fng/?limit={days}&format=json"
    
    print(f"Fetching Fear & Greed Index ({days} days)...")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    
    data = response.json()
    
    if "data" not in data:
        raise ValueError(f"Unexpected API response: {data}")
    
    records = data["data"]
    print(f"Received {len(records)} records")
    
    # Parse records
    rows = []
    for record in records:
        rows.append({
            "timestamp": pd.to_datetime(int(record["timestamp"]), unit="s", utc=True),
            "value": int(record["value"]),
            "value_classification": record["value_classification"],
        })
    
    df = pd.DataFrame(rows)
    df = df.set_index("timestamp").sort_index()
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Fetch Fear & Greed Index data from alternative.me"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=1500,
        help="Number of days of history to fetch (default: 1500)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/fear_greed_index_2021-2025.csv",
        help="Output CSV file path"
    )
    
    args = parser.parse_args()
    
    # Fetch data
    df = fetch_fear_greed(args.days)
    
    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path)
    
    # Print summary
    print(f"\n✅ Data saved to: {output_path}")
    print(f"   Date range: {df.index.min()} to {df.index.max()}")
    print(f"   Total days: {len(df)}")
    print(f"   Value range: {df['value'].min()} to {df['value'].max()}")
    print(f"\nSample data:")
    print(df.head())


if __name__ == "__main__":
    main()
