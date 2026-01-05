# Chapter 12: Utility Tools

> **Purpose:** This chapter provides exhaustive documentation of the utility tools included with the bot, covering dust sweeping, config validation, PnL reconciliation, WFO window selection, and testnet seeding.

---

## 12.1 Dust Sweeper

### 1. Concept & "The Why"

* **What it is:** Automatically converts small "dust" balances (< $5 value) to BNB using Binance's dust-to-BNB feature.

* **Purpose:** 
  - Prevents accumulation of untradeable micro-balances
  - Consolidates small amounts into usable BNB
  - Can run as a periodic background service

* **Location:** [`tools/dust_sweeper.py`](../../tools/dust_sweeper.py)

### 2. Configuration & Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `auto` | Mode: `auto`, `dry`, `wet` |
| `--threshold-usd` | `5.0` | Assets below this USD value are swept |
| `--schedule` | `once` | Run schedule: `once`, `daily`, `weekly` |
| `--ignore` | `""` | Comma-separated assets to skip |

**Environment Variables:**
| Variable | Description |
|----------|-------------|
| `MODE` | Deployment mode (`dry`, `testnet`, `live`) |
| `SWEEPER_IGNORE` | Protected assets (e.g., `SHIB,DOGE`) |
| `BINANCE_KEY` | API key |
| `BINANCE_SECRET` | API secret |

**Auto-Mode Logic:**
- `MODE=dry` → Dry run (preview only)
- `MODE=testnet` → Wet run (fails gracefully—not supported on testnet)
- `MODE=live` → Wet run (executes sweep)

**Protected Assets (always ignored):**
- `BNB` (destination asset)
- `BTC` (core trading asset)

### 3. Step-by-Step Guide

1. **Preview dust (dry run):**
   ```bash
   python tools/dust_sweeper.py --mode dry
   ```
   Output:
   ```
   [SWEEPER] Found Dust: 0.00123456 XRP (~$0.58)
   [SWEEPER] Found Dust: 0.00045678 ADA (~$0.22)
   [SWEEPER] [DRY] Would convert: ['XRP', 'ADA']
   ```

2. **Execute sweep on live account:**
   ```bash
   export MODE=live
   python tools/dust_sweeper.py --mode auto
   ```

3. **Run as daily service:**
   ```bash
   python tools/dust_sweeper.py --mode auto --schedule daily
   ```

4. **Ignore specific assets:**
   ```bash
   python tools/dust_sweeper.py --mode dry --ignore SHIB,DOGE
   ```

### 4. Real-World Use Case

**Scenario:** Weekly cleanup of trading account.

**Configuration:**
```bash
# .env
MODE=live
SWEEPER_IGNORE=SHIB,PEPE
```

**Command:**
```bash
python tools/dust_sweeper.py --threshold-usd 10.0 --schedule weekly
```

**Expected Outcome:**
```
[SWEEPER] Found Dust: 0.00234567 XRP (~$1.12)
[SWEEPER] SUCCESS! Sweep response: {'totalServiceCharge': '0.00001', 'transferResult': [...]}
[SWEEPER] Sleeping for weekly (604800s)...
```

### 5. Troubleshooting

**Not supported on Testnet:**
```
Sweep failed (API Error): You are not authorized to execute this request.
```
**Cause:** Dust transfer is not available on Binance Testnet.
**Fix:** Only use on live accounts.

**Method not found:**
```
Sweep failed: Method 'dust_transfer' not found.
```
**Fix:** Update binance-connector: `pip install --upgrade binance-connector`

---

## 12.2 Config Sanity Checker

### 1. Concept & "The Why"

* **What it is:** Validates a configuration file by running a full backtest and reporting summary statistics.

* **Purpose:** 
  - Quickly verify config produces reasonable results
  - Catch configuration errors before live deployment
  - Compare different configs objectively

* **Location:** [`tools/sanity_check_config.py`](../../tools/sanity_check_config.py)

### 2. Configuration & Parameters

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--data` | ✅ | — | Path to OHLCV CSV |
| `--config` | ✅ | — | JSON config file |
| `--funding-data` | ❌ | — | Optional funding CSV |
| `--start-bnb` | ❌ | `0.05` | Initial BNB balance |

### 3. Step-by-Step Guide

1. **Validate a config:**
   ```bash
   python tools/sanity_check_config.py \
     --data data/raw/ETHBTC_15m_2021-01_2025-01_vision.csv \
     --config configs/prod_eth_long_wfo_robust.json
   ```

2. **Expected output:**
   ```json
   {
     "initial_btc": 1.0,
     "final_btc": 1.4567,
     "total_return": 0.4567,
     "max_drawdown_pct": 0.12,
     "fees_btc": 0.0034,
     "turnover_btc": 2.5,
     "n_trades": 178,
     "n_bars": 140160
   }
   ```

3. **With funding rates:**
   ```bash
   python tools/sanity_check_config.py \
     --data data/raw/BTCUSDT_15m_2021-01_2025-01_vision.csv \
     --config configs/prod_btc_meta_live.json \
     --funding-data data/raw/BTCUSDT_funding_2021-01-01_2025-01-01_funding.csv
   ```

### 4. Troubleshooting

**Config validation error:**
```
pydantic.error_wrappers.ValidationError: 1 validation error for Strategy
```
**Cause:** Missing or invalid field in config.
**Fix:** Check config against schema in `core/config_schema.py`.

---

## 12.3 PnL Reconciler

### 1. Concept & "The Why"

* **What it is:** Compares the bot's internal state with actual exchange balances to detect discrepancies.

* **Purpose:** 
  - Detect missed fills or syncing issues
  - Audit trail for accounting
  - Automated alerts on divergence

* **Location:** [`tools/reconcile_pnl.py`](../../tools/reconcile_pnl.py)

### 2. Configuration & Parameters

**Environment Variables:**
| Variable | Default | Description |
|----------|---------|-------------|
| `SYMBOL` | `ETHBTC` | Trading pair |
| `MODE` | `testnet` | Deployment mode |
| `STATE_FILE` | `/data/state_{mode}.json` | Bot state file |
| `BINANCE_KEY` | — | API key |
| `BINANCE_SECRET` | — | API secret |

### 3. Step-by-Step Guide

1. **Run reconciliation:**
   ```bash
   export SYMBOL=ETHBTC
   export MODE=live
   python tools/reconcile_pnl.py
   ```

2. **Expected output:**
   ```
   --- PnL Reconciler (LIVE) ---
   Target: ETHBTC
   State File: /data/state_live.json
   API URL: https://api.binance.com

   🔍 **AUDIT REPORT (ETHBTC)**
   • Bot Start W: 1.000000
   • Real Wallet: 1.045678 BTC (Base=12.5678, Quote=0.6234)
   • Actual PnL:  +4.57% (|Δ|=4.57%)
   ```

3. **Alerts on divergence:**
   - `> 1%` divergence → WARNING alert
   - `> 5%` divergence → CRITICAL alert

### 4. Troubleshooting

**State file not found:**
```
❌ Error: No state file found at /data/state_live.json
```
**Cause:** Bot hasn't run yet or wrong state file path.
**Fix:** Verify `STATE_FILE` env var points to correct path.

---

## 12.4 WFO Window Selector

### 1. Concept & "The Why"

* **What it is:** Intelligently selects the best WFO window parameters using various scoring strategies.

* **Purpose:** 
  - Choose optimal parameters from WFO results
  - Balance performance, consistency, and recency
  - Create ensemble averages from multiple windows

* **Location:** [`tools/wfo_select_best.py`](../../tools/wfo_select_best.py)

### 2. Configuration & Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--wfo-csv` | (required) | WFO results CSV file |
| `--out` | (required) | Output JSON config |
| `--strategy` | `weighted` | Selection strategy |
| `--ensemble-n` | `5` | Top N for ensemble |
| `--stability-lambda` | `0.1` | Stability penalty |

**Selection Strategies:**

| Strategy | Description |
|----------|-------------|
| `best_oos` | Best out-of-sample performance only |
| `weighted` | Balanced: 60% OOS + 30% avg + 10% recency |
| `consistent` | Harmonic mean of train/test |
| `recent` | OOS weighted by recency |
| `ensemble` | Average params from top N windows |
| `stable` | Weighted + stability penalty |
| `stable_ensemble` | Ensemble from stable windows |

### 3. Step-by-Step Guide

1. **Select best weighted window:**
   ```bash
   python tools/wfo_select_best.py \
     --wfo-csv results/wfo_mr_ethbtc.csv \
     --strategy weighted \
     --out configs/selected_params.json
   ```

2. **Create ensemble from top 5:**
   ```bash
   python tools/wfo_select_best.py \
     --wfo-csv results/wfo_mr_ethbtc.csv \
     --strategy ensemble \
     --ensemble-n 5 \
     --out configs/ensemble_params.json
   ```

3. **Expected output:**
   ```
   ============================================================
   SMART WFO SELECTION (Strategy: weighted)
   ============================================================
   Selected Window: 2024-06-30
   OOS Profit: 1.1234
   Train Profit: 1.1567
   Consistency Ratio: 0.97
   Score: 1.0876
   ============================================================

   Top 5 Windows:
   ------------------------------------------------------------
     2024-06-30: OOS=1.1234 Train=1.1567 Ratio=0.97 Score=1.0876 ✅
     2024-05-31: OOS=1.0987 Train=1.0123 Ratio=1.09 Score=1.0654 ✅
     ...

   ✅ Saved to: configs/selected_params.json
   ```

### 4. Scoring Formula

```python
# Weighted strategy (default)
score = (
    oos_profit * 0.6                    # 60% OOS performance
    + (oos_profit + train_profit) / 2 * 0.3  # 30% average
    + recency_weight / max * 0.1        # 10% recency bonus
    - train_test_gap * 0.2              # Generalization penalty
)

# Suspicious window detection
suspicious = (train_test_ratio > 1.5) or (train_test_ratio < 0.7)
```

---

## 12.5 Futures Testnet Seeder

### 1. Concept & "The Why"

* **What it is:** Places a market order on Binance Futures Testnet to create a test position.

* **Purpose:** 
  - Initialize testnet with realistic position
  - Test Futures adapter before live deployment
  - Verify leverage and filter handling

* **Location:** [`tools/seed_futures_testnet.py`](../../tools/seed_futures_testnet.py)

### 2. Configuration & Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--symbol` | `BTCUSDT` | Trading pair |
| `--side` | `BUY` | Order side (`BUY`/`SELL`) |
| `--qty` | (required) | Quantity in base units |
| `--base-url` | Testnet URL | API base URL |

**Environment Variables:**
| Variable | Description |
|----------|-------------|
| `FUTURES_TESTNET_KEY` | Testnet API key |
| `FUTURES_TESTNET_SECRET` | Testnet API secret |

### 3. Step-by-Step Guide

1. **Get testnet credentials:**
   - Visit `https://testnet.binancefuture.com`
   - Create API key

2. **Set environment:**
   ```bash
   export FUTURES_TESTNET_KEY=your_testnet_key
   export FUTURES_TESTNET_SECRET=your_testnet_secret
   ```

3. **Open long position:**
   ```bash
   python tools/seed_futures_testnet.py \
     --symbol BTCUSDT \
     --side BUY \
     --qty 0.01
   ```

4. **Open short position:**
   ```bash
   python tools/seed_futures_testnet.py \
     --symbol BTCUSDT \
     --side SELL \
     --qty 0.01
   ```

5. **Expected output:**
   ```
   Set leverage to 1x for BTCUSDT
   Placing MARKET BUY BTCUSDT qty=0.01000000 (mid≈45123.50000000) on https://testnet.binancefuture.com
   OK: {'orderId': 123456789, 'status': 'FILLED', ...}

   Account balances:
     USDT: 10000.00000000

   Position BTCUSDT: 0.01 @ 45123.50
     Unrealized PnL: 0.00
   ```

### 4. Troubleshooting

**Quantity too small:**
```
Quantity too small for min notional.
  notional    = 0.451235
  minNotional = 5.000000
Try qty >= ~0.000111 (base units).
```
**Fix:** Increase `--qty` to meet minimum notional.

---

## 12.6 Additional Tools Reference

| Tool | Purpose | Command |
|------|---------|---------|
| `analyze_exposure.py` | Analyze exposure patterns | `python tools/analyze_exposure.py` |
| `analyze_meta.py` | Analyze Meta strategy regimes | `python tools/analyze_meta.py` |
| `vision_fix_timestamps.py` | Fix CSV timestamp formats | `python tools/vision_fix_timestamps.py` |
| `wfo_analyzer.py` | Deep WFO result analysis | `python tools/wfo_analyzer.py` |
| `regime_analysis.py` | Regime score analysis | `python tools/regime_analysis.py` |

---

## 12.7 Real-World Workflow

### Complete Pre-Deployment Checklist

**Step 1: Download fresh data**
```bash
python tools/download_vision.py --symbol BTCUSDT --intervals 15m --start 2021-01 --end 2025-01
python tools/download_funding.py --symbol BTCUSDT --start 2021-01-01 --end 2025-01-01
```

**Step 2: Run WFO optimization**
```bash
python tools/optimizer_cli.py --data data/raw/BTCUSDT_15m_*.csv --config configs/base.json --wfo
```

**Step 3: Select best parameters**
```bash
python tools/wfo_select_best.py --wfo-csv results/wfo_mr.csv --strategy stable_ensemble --out configs/prod.json
```

**Step 4: Validate config**
```bash
python tools/sanity_check_config.py --data data/raw/BTCUSDT_15m_*.csv --config configs/prod.json
```

**Step 5: Test on Futures testnet**
```bash
python tools/seed_futures_testnet.py --symbol BTCUSDT --side BUY --qty 0.01
```

**Step 6: Deploy and monitor**
```bash
docker-compose up -d
python tools/reconcile_pnl.py
```

**Step 7: Periodic maintenance**
```bash
python tools/dust_sweeper.py --mode auto --schedule weekly
```

---

*Previous Chapter: [Chapter 11: Data Pipeline](./CHAPTER_11_DATA_PIPELINE.md)*  
*Next Chapter: [Appendix A: Configuration Reference](./APPENDIX_A_CONFIG_REFERENCE.md)*
