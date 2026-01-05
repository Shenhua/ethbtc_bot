# Chapter 11: Data Pipeline

> **Purpose:** This chapter provides exhaustive documentation of the data pipeline, covering historical data download, format specifications, loading functions, and data management best practices.

---

## 11.1 Data Pipeline Architecture

### 1. Concept & "The Why"

* **What it is:** A complete data acquisition and processing pipeline that fetches historical OHLCV data and funding rates from Binance, normalizes timestamps, and prepares data for backtesting and optimization.

* **Purpose:** 
  1. **Reproducibility:** Consistent data format across all analyses
  2. **Quality:** Checksum verification, deduplication, and gap detection
  3. **Flexibility:** Support for multiple timeframes and symbols

* **Location:** 
  - OHLCV Download: [`tools/download_vision.py`](../../tools/download_vision.py)
  - Funding Download: [`tools/download_funding.py`](../../tools/download_funding.py)
  - Data Loader: [`core/ethbtc_accum_bot.py`](../../core/ethbtc_accum_bot.py) → `load_vision_csv()`
  - Output Directory: `data/raw/`

### 2. Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    BINANCE DATA SOURCES                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   data.binance.vision                  fapi.binance.com             │
│   (Historical Klines)                  (Funding Rates)              │
│         │                                     │                     │
│         ▼                                     ▼                     │
│   download_vision.py                  download_funding.py           │
│         │                                     │                     │
│         ▼                                     ▼                     │
│   ┌─────────────────┐                 ┌─────────────────┐           │
│   │ Monthly Zips    │                 │ Paginated JSON  │           │
│   │ (Checksum)      │                 │ (1000/request)  │           │
│   │       +         │                 └────────┬────────┘           │
│   │ Daily Zips      │                          │                    │
│   │ (Gap Fill)      │                          │                    │
│   └────────┬────────┘                          │                    │
│            │                                   │                    │
│            ▼                                   ▼                    │
│   ┌───────────────────────────────────────────────────────────────┐ │
│   │                     data/raw/                                  │ │
│   │                                                                 │ │
│   │  ETHBTC_15m_2021-01_2025-01_vision.csv                        │ │
│   │  BTCUSDT_15m_2021-01_2025-01_vision.csv                       │ │
│   │  ETHUSDT_funding_2021-01-01_2025-01-01_funding.csv            │ │
│   └───────────────────────────────────────────────────────────────┘ │
│                            │                                        │
│                            ▼                                        │
│                    load_vision_csv()                                │
│                            │                                        │
│                            ▼                                        │
│                    pd.DataFrame                                     │
│                    (Indexed by close_time)                          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 11.2 Downloading OHLCV Data

### 1. Concept & "The Why"

* **What it is:** Downloads historical kline (candlestick) data from Binance Vision—the official historical data repository.

* **Purpose:** 
  - Free, official data without API rate limits
  - Pre-packaged monthly/daily ZIP files with checksums
  - Reliable for multi-year backtests

* **Location:** [`tools/download_vision.py`](../../tools/download_vision.py)

### 2. Configuration & Parameters

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--symbol` | ❌ | `ETHBTC` (or `$SYMBOL` env) | Trading pair |
| `--intervals` | ❌ | `1m,15m,30m` | Comma-separated intervals |
| `--start` | ✅ | — | Start date (YYYY-MM or YYYY-MM-DD) |
| `--end` | ✅ | — | End date (YYYY-MM or YYYY-MM-DD) |
| `--out-dir` | ❌ | `data/raw/` | Output directory |
| `--prefer-daily` | ❌ | false | Use daily files first |

### 3. Step-by-Step Guide

1. **Download ETH/BTC 15m data for 4 years:**
   ```bash
   python tools/download_vision.py \
     --symbol ETHBTC \
     --intervals 15m \
     --start 2021-01 \
     --end 2025-01 \
     --out-dir data/raw/
   ```

2. **Expected output:**
   ```
   == ETHBTC 15m ==
   Downloading monthly ETHBTC-15m-2021-01.zip
   Downloading monthly ETHBTC-15m-2021-02.zip
   ...
   Downloading monthly ETHBTC-15m-2024-11.zip
   Downloading daily ETHBTC-15m-2024-12-01.zip  (gap fill)
   ...
   Saved merged CSV → data/raw/ETHBTC_15m_2021-01_2025-01_vision.csv (140160 rows)
   ```

3. **Download multiple intervals:**
   ```bash
   python tools/download_vision.py \
     --symbol BTCUSDT \
     --intervals 15m,30m,1h \
     --start 2021-01 \
     --end 2025-01
   ```

4. **Download for Futures symbols:**
   ```bash
   python tools/download_vision.py \
     --symbol BTCUSDT \
     --intervals 15m \
     --start 2021-01 \
     --end 2025-01
   ```

### 4. Download Logic

```
1. MONTHLY FIRST (default):
   - Try downloading monthly ZIP files (faster, fewer requests)
   - For each successful month, mark as complete
   
2. DAILY GAP FILL:
   - For dates NOT covered by monthly files
   - Download daily ZIPs to fill gaps
   - Common for recent months (not yet archived monthly)

3. MERGE & DEDUPLICATE:
   - Concatenate all dataframes
   - Remove duplicate timestamps (keep first)
   - Sort by open_time
   
4. SAVE:
   - Output: {SYMBOL}_{interval}_{start}_{end}_vision.csv
```

### 5. Checksums

Each ZIP file has a corresponding `.CHECKSUM` file:
```
sha256  ETHBTC-15m-2021-01.zip
```

The script automatically:
- Downloads checksum file
- Verifies SHA256 hash
- Warns if mismatch (but continues)

### 6. Troubleshooting

**No data found:**
```
No data found for ETHBTC 15m in range 2025-02..2025-03
```
**Cause:** Data not yet published on Binance Vision.
**Fix:** Use an earlier date range or wait for Binance to publish.

**Checksum warning:**
```
WARNING: checksum failed for ETHBTC-15m-2021-01.zip
```
**Cause:** Download corruption or file mismatch.
**Fix:** Delete and re-download.

---

## 11.3 Downloading Funding Rates

### 1. Concept & "The Why"

* **What it is:** Downloads historical funding rate data from Binance Futures API for funding-aware backtesting.

* **Purpose:** 
  - Simulate funding payments in backtests
  - Enable funding rate filters in strategies
  - Track extreme funding conditions

* **Location:** [`tools/download_funding.py`](../../tools/download_funding.py)

### 2. Configuration & Parameters

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--symbol` | ✅ | — | Futures symbol (e.g., ETHUSDT) |
| `--start` | ✅ | — | Start date (YYYY-MM-DD) |
| `--end` | ✅ | — | End date (YYYY-MM-DD) |
| `--out` | ❌ | auto-generated | Output CSV path |

### 3. Step-by-Step Guide

1. **Download funding rates:**
   ```bash
   python tools/download_funding.py \
     --symbol ETHUSDT \
     --start 2021-01-01 \
     --end 2025-01-01
   ```

2. **Expected output:**
   ```
   Downloading Funding Rates for ETHUSDT...
   Range: 2021-01-01 to 2025-01-01
   Fetched 1000 records. Last: 2021-04-12 08:00:00
   Fetched 1000 records. Last: 2021-07-23 16:00:00
   ...
   Success! Saved 11680 rows to data/raw/ETHUSDT_funding_2021-01-01_2025-01-01_funding.csv
   ```

3. **Output format:**
   ```csv
   time,rate
   2021-01-01 00:00:00+00:00,0.01
   2021-01-01 08:00:00+00:00,0.02
   2021-01-01 16:00:00+00:00,0.015
   ```

### 4. API Details

- **Endpoint:** `https://fapi.binance.com/fapi/v1/fundingRate`
- **Limit:** 1000 records per request
- **Rate:** Funding occurs every 8 hours (3 times daily)
- **Rate Limiting:** Hardcoded 0.2s delay between requests

### 5. Troubleshooting

**No data found:**
```
No data found!
```
**Cause:** Symbol doesn't exist on Futures or date range invalid.
**Fix:** Verify symbol is traded on Binance Futures (e.g., ETHUSDT not ETHBTC).

---

## 11.4 Data Format Specification

### 1. OHLCV Format (Vision CSV)

| Column | Type | Description |
|--------|------|-------------|
| `open_time` | datetime | Bar open timestamp (UTC) |
| `open` | float | Opening price |
| `high` | float | High price |
| `low` | float | Low price |
| `close` | float | Closing price |
| `volume` | float | Base asset volume |
| `close_time` | datetime | Bar close timestamp (UTC) |
| `qav` | float | Quote asset volume |
| `trades` | int | Number of trades |
| `taker_base` | float | Taker buy base volume |
| `taker_quote` | float | Taker buy quote volume |
| `ignore` | — | Unused column |

**Example:**
```csv
open_time,open,high,low,close,volume,close_time,qav,trades,taker_base,taker_quote,ignore
2021-01-01 00:00:00+00:00,0.034,0.0342,0.0339,0.0341,1234.5,2021-01-01 00:14:59+00:00,42.1,567,617.2,21.0,0
```

### 2. Funding Rate Format

| Column | Type | Description |
|--------|------|-------------|
| `time` | datetime | Funding timestamp (UTC) |
| `rate` | float | Funding rate as percentage |

**Example:**
```csv
time,rate
2021-01-01 00:00:00+00:00,0.01
2021-01-01 08:00:00+00:00,0.02
```

**Note:** Rate is stored as percentage (0.01 = 0.01%, not 1%).

---

## 11.5 Loading Data

### 1. Concept & "The Why"

* **What it is:** The `load_vision_csv()` function parses CSV files into pandas DataFrames with proper datetime indexing and data cleaning.

* **Purpose:** 
  - Handles both numeric and string timestamps
  - Auto-detects milliseconds vs seconds
  - Cleans and deduplicates data

* **Location:** [`core/ethbtc_accum_bot.py`](../../core/ethbtc_accum_bot.py) → `load_vision_csv()`

### 2. Usage

```python
from core.ethbtc_accum_bot import load_vision_csv

# Load OHLCV data
df = load_vision_csv("data/raw/ETHBTC_15m_2021-01_2025-01_vision.csv")

print(df.head())
#                              open     high      low    close   volume
# close_time                                                           
# 2021-01-01 00:14:59+00:00  0.0340  0.0342  0.0339  0.0341   1234.5
# 2021-01-01 00:29:59+00:00  0.0341  0.0343  0.0340  0.0342   1567.8

print(df.index)
# DatetimeIndex(['2021-01-01 00:14:59+00:00', ...], dtype='datetime64[ns, UTC]')
```

### 3. Processing Steps

```python
def load_vision_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    
    # 1. Normalize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    
    # 2. Handle column aliases
    alias = {"opentime": "open_time", "closetime": "close_time"}
    
    # 3. Auto-detect timestamp format
    #    - If > 1e11: assume milliseconds
    #    - Otherwise: assume seconds
    
    # 4. Convert to pandas datetime (UTC)
    df["close_time"] = pd.to_datetime(...)
    
    # 5. Convert OHLCV to numeric
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    
    # 6. Drop invalid rows, set index
    df = df.dropna(subset=["close"]).set_index("close_time")
    
    # 7. Remove duplicates, sort
    df = df[~df.index.duplicated(keep='first')]
    df = df.sort_index()
    
    return df
```

### 4. Loading Funding Rates

```python
import pandas as pd

def load_funding_series(path: str, ref_index: pd.DatetimeIndex) -> pd.Series:
    """Load and align funding rates to OHLCV index."""
    f_df = pd.read_csv(path)
    f_df["time"] = pd.to_datetime(f_df["time"], utc=True, format="mixed")
    f_df = f_df.set_index("time").sort_index()
    
    # Align to OHLCV index using forward-fill
    funding = f_df["rate"].reindex(ref_index).ffill().fillna(0.0)
    return funding
```

---

## 11.6 Data Quality Checks

### 1. Gap Detection

```python
def detect_gaps(df: pd.DataFrame, expected_interval: str = "15min") -> pd.DataFrame:
    """Find gaps in time series data."""
    import pandas as pd
    
    expected = pd.Timedelta(expected_interval)
    time_diff = df.index.to_series().diff()
    
    # Gaps are where diff > expected + tolerance
    gaps = time_diff[time_diff > expected * 1.5]
    
    if len(gaps) > 0:
        print(f"Found {len(gaps)} gaps:")
        for idx, gap in gaps.items():
            print(f"  {idx}: {gap}")
    
    return gaps
```

### 2. Duplicate Detection

```python
# Check for duplicates
duplicates = df.index.duplicated()
print(f"Duplicate rows: {duplicates.sum()}")

# Remove duplicates (keep first)
df_clean = df[~df.index.duplicated(keep='first')]
```

### 3. Missing Value Detection

```python
# Check for NaN values
print(df.isna().sum())

# OHLCV should have no NaN
assert df[["open", "high", "low", "close", "volume"]].isna().sum().sum() == 0
```

---

## 11.7 Real-World Use Case (The "Cookbook")

### Scenario: Prepare Complete Dataset for BTC/USDT Optimization

**Step 1: Download 4 years of OHLCV data**
```bash
python tools/download_vision.py \
  --symbol BTCUSDT \
  --intervals 15m,30m,1h \
  --start 2021-01 \
  --end 2025-01 \
  --out-dir data/raw/
```

**Step 2: Download funding rates**
```bash
python tools/download_funding.py \
  --symbol BTCUSDT \
  --start 2021-01-01 \
  --end 2025-01-01
```

**Step 3: Download BNB price (for fee calculation)**
```bash
python tools/download_vision.py \
  --symbol BNBUSDT \
  --intervals 15m \
  --start 2021-01 \
  --end 2025-01
```

**Step 4: Verify data quality**
```python
from core.ethbtc_accum_bot import load_vision_csv

# Load and check
df = load_vision_csv("data/raw/BTCUSDT_15m_2021-01_2025-01_vision.csv")

print(f"Rows: {len(df)}")
print(f"Date range: {df.index[0]} to {df.index[-1]}")
print(f"Missing values: {df.isna().sum().sum()}")

# Check for gaps
expected = pd.Timedelta("15min")
gaps = df.index.to_series().diff()
big_gaps = gaps[gaps > expected * 1.5]
print(f"Gaps found: {len(big_gaps)}")
```

**Step 5: Run backtest**
```bash
python core/ethbtc_accum_bot.py backtest \
  --data data/raw/BTCUSDT_15m_2021-01_2025-01_vision.csv \
  --funding-data data/raw/BTCUSDT_funding_2021-01-01_2025-01-01_funding.csv \
  --bnb-data data/raw/BNBUSDT_15m_2021-01_2025-01_vision.csv \
  --config configs/prod_btc_meta_live.json \
  --report
```

**Expected Outcome:**
- ~140,160 bars of 15m data (4 years)
- ~11,680 funding rate records
- 0 gaps, 0 duplicates

---

## 11.8 Directory Structure

```
data/
├── raw/                              # Downloaded raw data
│   ├── ETHBTC_15m_2021-01_2025-01_vision.csv
│   ├── BTCUSDT_15m_2021-01_2025-01_vision.csv
│   ├── ETHUSDT_funding_2021-01-01_2025-01-01_funding.csv
│   └── BNBUSDT_15m_2021-01_2025-01_vision.csv
├── db/                               # Optuna studies
│   └── optuna.db
└── processed/                        # (Optional) Pre-processed data
```

---

## 11.9 Troubleshooting

### Common Errors

```
ValueError: close_time column not found
```
**Cause:** CSV uses different column name.
**Fix:** Rename column to `close_time` or `date`.

```
requests.exceptions.Timeout
```
**Cause:** Network timeout during download.
**Fix:** Re-run script; downloads resume where left off.

```
KeyError: 'close'
```
**Cause:** OHLCV columns missing from CSV.
**Fix:** Verify CSV has open, high, low, close, volume columns.

### Performance Tips

1. **Use monthly files (default):** Faster than daily (12 files vs 365)

2. **Download once, use many times:** Data is deterministic

3. **Pre-download during off-hours:** Avoid API congestion

4. **Store in SSD:** Faster CSV parsing for large files

---

*Previous Chapter: [Chapter 10: Monitoring & Observability](./CHAPTER_10_MONITORING.md)*  
*Next Chapter: [Chapter 12: Troubleshooting & FAQ](./CHAPTER_12_TROUBLESHOOTING.md)*
