# Chapter 14: Workflows & Recipes

> **Purpose:** This chapter provides step-by-step operational workflows for common tasks, from initial setup to production deployment, optimization cycles, and emergency recovery.

---

## 14.1 First-Time Setup Workflow

### 1. Concept & "The Why"

* **What it is:** Complete workflow from cloning the repo to running your first backtest.

* **Purpose:** Get new users to a working state as quickly as possible.

### 2. Step-by-Step Guide

**Step 1: Clone and setup environment**
```bash
git clone https://github.com/your-repo/ethbtc_bot_3.git
cd ethbtc_bot_3
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Step 2: Configure environment variables**
```bash
cp .env.example .env
# Edit .env with your Binance API keys
```

**Step 3: Download historical data**
```bash
python tools/download_vision.py \
  --symbol ETHBTC \
  --intervals 15m \
  --start 2023-01 \
  --end 2025-01
```

**Step 4: Run first backtest**
```bash
python core/ethbtc_accum_bot.py backtest \
  --data data/raw/ETHBTC_15m_2023-01_2025-01_vision.csv \
  --config configs/prod_eth_long_wfo_robust.json \
  --report
```

**Step 5: Review results**
```bash
cat results/backtest_report_*.md
```

**Expected Outcome:**
- Clean environment with all dependencies
- 2 years of 15m data downloaded
- Backtest report showing strategy performance

---

## 14.2 Full Optimization Cycle

### 1. Concept & "The Why"

* **What it is:** End-to-end workflow for optimizing strategy parameters using WFO.

* **Purpose:** Find robust, generalizable parameters for live trading.

### 2. Step-by-Step Guide

**Step 1: Download fresh data**
```bash
# Download 4 years of OHLCV
python tools/download_vision.py \
  --symbol BTCUSDT \
  --intervals 15m \
  --start 2021-01 \
  --end 2025-01

# Download funding rates
python tools/download_funding.py \
  --symbol BTCUSDT \
  --start 2021-01-01 \
  --end 2025-01-01
```

**Step 2: Run Mean Reversion WFO**
```bash
python tools/optimizer_cli.py \
  --data data/raw/BTCUSDT_15m_2021-01_2025-01_vision.csv \
  --config configs/base_mr.json \
  --funding-data data/raw/BTCUSDT_funding_*.csv \
  --wfo \
  --window-days 180 \
  --step-days 30 \
  --n-trials 100 \
  --out results/wfo_mr_btcusdt.csv
```

**Step 3: Run Trend WFO**
```bash
python tools/optimize_trend.py \
  --data data/raw/BTCUSDT_15m_2021-01_2025-01_vision.csv \
  --funding-data data/raw/BTCUSDT_funding_*.csv \
  --wfo \
  --window-days 180 \
  --step-days 30 \
  --n-trials 50 \
  --allow-shorts \
  --out results/wfo_trend_btcusdt.csv
```

**Step 4: Select best parameters**
```bash
python tools/wfo_select_best.py \
  --wfo-csv results/wfo_mr_btcusdt.csv \
  --strategy stable_ensemble \
  --ensemble-n 5 \
  --out configs/optimized_mr.json

python tools/wfo_select_best.py \
  --wfo-csv results/wfo_trend_btcusdt.csv \
  --strategy weighted \
  --out configs/optimized_trend.json
```

**Step 5: Optimize Meta threshold**
```bash
python tools/optimize_meta.py \
  --data data/raw/BTCUSDT_15m_2021-01_2025-01_vision.csv \
  --mr-config configs/optimized_mr.json \
  --trend-config configs/optimized_trend.json \
  --out results/opt_meta.csv
```

**Step 6: Assemble production config**
```bash
python tools/assemble_v2_config.py \
  --mr-params configs/optimized_mr.json \
  --trend-params configs/optimized_trend.json \
  --adx-threshold 25 \
  --out configs/prod_btcusdt_optimized.json
```

**Step 7: Validate final config**
```bash
python tools/sanity_check_config.py \
  --data data/raw/BTCUSDT_15m_2021-01_2025-01_vision.csv \
  --config configs/prod_btcusdt_optimized.json \
  --funding-data data/raw/BTCUSDT_funding_*.csv
```

**Expected Outcome:**
- WFO results with 40+ windows
- Stable ensemble parameters
- Production-ready config file
- Sanity check showing positive returns

---

## 14.3 Testnet Deployment

### 1. Concept & "The Why"

* **What it is:** Workflow for deploying to Binance Testnet before live trading.

* **Purpose:** Validate end-to-end system without risking real capital.

### 2. Step-by-Step Guide

**Step 1: Get testnet credentials**
- Spot: https://testnet.binance.vision
- Futures: https://testnet.binancefuture.com

**Step 2: Configure .env**
```bash
# .env
MODE=testnet
BINANCE_KEY=your_testnet_spot_key
BINANCE_SECRET=your_testnet_spot_secret
FUTURES_TESTNET_KEY=your_testnet_futures_key
FUTURES_TESTNET_SECRET=your_testnet_futures_secret
BINANCE_BASE_URL=https://testnet.binance.vision
```

**Step 3: Seed testnet with initial position (Futures)**
```bash
python tools/seed_futures_testnet.py \
  --symbol BTCUSDT \
  --side BUY \
  --qty 0.01
```

**Step 4: Start bot in testnet mode**
```bash
python live_executor.py \
  --params configs/prod_btcusdt_optimized.json \
  --mode testnet \
  --symbol BTCUSDT
```

**Step 5: Monitor for 24-48 hours**
```bash
# Check metrics
curl http://localhost:9109/metrics | grep wealth_total

# Check logs
docker logs -f bot_btc
```

**Step 6: Verify with PnL reconciler**
```bash
python tools/reconcile_pnl.py
```

**Expected Outcome:**
- Bot trading on testnet without errors
- Metrics flowing to Prometheus
- PnL reconciliation within 1%

---

## 14.4 Production Deployment

### 1. Concept & "The Why"

* **What it is:** Workflow for going live with real capital.

* **Purpose:** Minimize risk during production launch.

### 2. Pre-Flight Checklist

```
☐ Testnet validated for 48+ hours
☐ All tests passing (pytest tests/ -v)
☐ Config validated (sanity_check_config.py)
☐ API permissions set (read, trade, no withdraw)
☐ Alerts configured (Discord/Telegram)
☐ Monitoring stack running (Prometheus/Grafana)
☐ Initial capital deposited
☐ Emergency procedures documented
```

### 3. Step-by-Step Guide

**Step 1: Update .env for production**
```bash
# .env
MODE=live
BINANCE_KEY=your_live_api_key
BINANCE_SECRET=your_live_api_secret
STATE_FILE=/data/state_live.json
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
LOGLEVEL=INFO
```

**Step 2: Start Docker stack**
```bash
docker-compose up -d
```

**Step 3: Verify bot started correctly**
```bash
# Check container status
docker ps

# Check initial log output
docker logs bot_eth --tail 50

# Verify metrics
curl http://localhost:9109/metrics | grep up{
```

**Step 4: Enable alerts**
```bash
# Test Discord alert
python -c "
from core.alert_manager import AlertManager
alerter = AlertManager(prefix='PROD')
alerter.send('Production deployment initiated', level='INFO')
"
```

**Step 5: Monitor first 24 hours closely**
- Watch Grafana dashboards
- Check for unexpected risk flags
- Verify trade execution quality

**Expected Outcome:**
- Bot running smoothly in production
- Trades executing correctly
- Alerts working
- Metrics visible in Grafana

---

## 14.5 Adding a New Trading Pair

### 1. Concept & "The Why"

* **What it is:** Workflow for adding support for a new symbol.

* **Purpose:** Expand trading to additional pairs.

### 2. Step-by-Step Guide

**Step 1: Download data for new pair**
```bash
python tools/download_vision.py \
  --symbol LTCBTC \
  --intervals 15m \
  --start 2021-01 \
  --end 2025-01

python tools/download_funding.py \
  --symbol LTCUSDT \
  --start 2021-01-01 \
  --end 2025-01-01
```

**Step 2: Create base config**
```bash
cp configs/prod_eth_long_wfo_robust.json configs/base_ltc.json
# Edit symbol references if needed
```

**Step 3: Run optimization**
```bash
python tools/optimizer_cli.py \
  --data data/raw/LTCBTC_15m_*.csv \
  --config configs/base_ltc.json \
  --wfo \
  --n-trials 100 \
  --out results/wfo_ltc.csv
```

**Step 4: Select and validate**
```bash
python tools/wfo_select_best.py \
  --wfo-csv results/wfo_ltc.csv \
  --strategy stable_ensemble \
  --out configs/prod_ltc.json

python tools/sanity_check_config.py \
  --data data/raw/LTCBTC_15m_*.csv \
  --config configs/prod_ltc.json
```

**Step 5: Add to docker-compose.yml**
```yaml
  bot_ltc:
    <<: *bot_template
    container_name: bot_ltc
    environment:
      - SYMBOL=LTCBTC
      - CONFIG_FILE=/app/configs/prod_ltc.json
    volumes:
      - ./run_state/ltc:/data
```

**Step 6: Deploy**
```bash
docker-compose up -d bot_ltc
```

**Expected Outcome:**
- New pair optimized and validated
- Bot instance running in Docker
- Metrics appearing in Prometheus

---

## 14.6 Emergency Recovery

### 1. Concept & "The Why"

* **What it is:** Procedures for handling critical failures.

* **Purpose:** Minimize damage during emergencies.

### 2. Scenarios and Responses

#### Scenario A: Max Drawdown Hit

**Symptoms:**
- Alert: "🚨 MAX DRAWDOWN HIT"
- `risk_flags{kind="maxdd_hit"} == 1`

**Response:**
```bash
# 1. Bot automatically halts trading
# 2. Investigate cause
docker logs bot_eth --since 1h | grep -i "drawdown\|error"

# 3. If legitimate market move, wait for Phoenix reset
# 4. If bug, fix and redeploy

# 5. Manual reset (if needed)
# Edit state file:
jq '.maxdd_hit = false | .equity_high = 0.85' /data/state_live.json > tmp.json && mv tmp.json /data/state_live.json
```

#### Scenario B: Exchange Connection Lost

**Symptoms:**
- `up{instance="ethbtc_live"} == 0`
- Repeated "Connection refused" in logs

**Response:**
```bash
# 1. Check Binance status
curl -s https://api.binance.com/api/v3/ping

# 2. Restart bot
docker-compose restart bot_eth

# 3. If persistent, check API keys
docker logs bot_eth | grep -i "auth\|key\|permission"
```

#### Scenario C: Config Error After Deploy

**Symptoms:**
- Bot crashing on startup
- "ValidationError" in logs

**Response:**
```bash
# 1. Roll back to previous config
cp configs/prod_eth_backup.json configs/prod_eth.json

# 2. Restart
docker-compose restart bot_eth

# 3. Debug config offline
python tools/sanity_check_config.py --config configs/broken_config.json
```

#### Scenario D: Significant PnL Divergence

**Symptoms:**
- `reconcile_pnl.py` shows > 5% divergence
- Alert: "WALLET vs BOT DIVERGENCE"

**Response:**
```bash
# 1. Stop trading immediately
docker stop bot_eth

# 2. Investigate
python tools/reconcile_pnl.py

# 3. Check trade logs
cat /data/trade_log.jsonl | tail -20

# 4. Check Binance order history
# Compare with bot's recorded trades

# 5. If state is corrupt, resync from exchange
# Create fresh state file with current balances
```

---

## 14.7 Monthly Maintenance

### 1. Concept & "The Why"

* **What it is:** Recurring tasks to keep the system healthy.

* **Purpose:** Prevent drift and maintain optimal performance.

### 2. Monthly Checklist

```
☐ Download latest month's data
☐ Re-run WFO with new data
☐ Compare new params vs current
☐ Update config if significantly better
☐ Run dust sweeper
☐ Review trade logs for anomalies
☐ Check Prometheus disk usage
☐ Verify backup procedures
☐ Update dependencies (security patches)
```

### 3. Step-by-Step Guide

**Step 1: Download new data**
```bash
python tools/download_vision.py \
  --symbol ETHBTC \
  --intervals 15m \
  --start 2024-12 \
  --end 2025-01
```

**Step 2: Append to existing dataset**
```bash
# Merge new data with existing
python -c "
import pandas as pd
old = pd.read_csv('data/raw/ETHBTC_15m_full.csv')
new = pd.read_csv('data/raw/ETHBTC_15m_2024-12_2025-01_vision.csv')
merged = pd.concat([old, new]).drop_duplicates().sort_values('close_time')
merged.to_csv('data/raw/ETHBTC_15m_full.csv', index=False)
"
```

**Step 3: Re-run optimization**
```bash
python tools/optimizer_cli.py \
  --data data/raw/ETHBTC_15m_full.csv \
  --config configs/current_prod.json \
  --wfo \
  --n-trials 50 \
  --out results/wfo_monthly_update.csv
```

**Step 4: Compare results**
```bash
python tools/wfo_select_best.py \
  --wfo-csv results/wfo_monthly_update.csv \
  --strategy weighted \
  --out configs/candidate_params.json

# Compare backtests
python core/ethbtc_accum_bot.py backtest \
  --config configs/current_prod.json \
  --data data/raw/ETHBTC_15m_full.csv \
  --start 2024-01-01 \
  --report

python core/ethbtc_accum_bot.py backtest \
  --config configs/candidate_params.json \
  --data data/raw/ETHBTC_15m_full.csv \
  --start 2024-01-01 \
  --report
```

**Step 5: Deploy if better**
```bash
# Only if candidate shows meaningful improvement
cp configs/candidate_params.json configs/current_prod.json
docker-compose restart bot_eth
```

**Step 6: Clean up**
```bash
# Run dust sweeper
python tools/dust_sweeper.py --mode auto

# Prune old logs
find logs/ -name "*.log" -mtime +30 -delete
```

---

## 14.8 Quick Reference Commands

### Data Management

```bash
# Download OHLCV
python tools/download_vision.py --symbol ETHBTC --intervals 15m --start 2024-01 --end 2025-01

# Download Funding
python tools/download_funding.py --symbol ETHUSDT --start 2024-01-01 --end 2025-01-01
```

### Backtesting

```bash
# Basic backtest
python core/ethbtc_accum_bot.py backtest --data data.csv --config config.json

# With report
python core/ethbtc_accum_bot.py backtest --data data.csv --config config.json --report

# With date range
python core/ethbtc_accum_bot.py backtest --data data.csv --config config.json --start 2024-01-01 --end 2024-12-31
```

### Optimization

```bash
# WFO Mode
python tools/optimizer_cli.py --data data.csv --config base.json --wfo --n-trials 100

# Select best params
python tools/wfo_select_best.py --wfo-csv results.csv --strategy stable_ensemble --out params.json
```

### Deployment

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker logs -f bot_eth

# Restart single service
docker-compose restart bot_eth
```

### Monitoring

```bash
# Check metrics
curl localhost:9109/metrics | grep wealth_total

# PnL reconciliation
python tools/reconcile_pnl.py

# Run tests
pytest tests/ -v
```

---

*Previous Chapter: [Chapter 13: Testing & Quality Assurance](./CHAPTER_13_TESTING.md)*  
*Next Chapter: [Appendix A: Configuration Reference](./APPENDIX_A_CONFIG_REFERENCE.md)*
