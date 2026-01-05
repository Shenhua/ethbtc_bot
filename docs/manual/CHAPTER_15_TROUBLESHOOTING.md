# Chapter 15: Troubleshooting & Edge Cases

> **Purpose:** This chapter provides a comprehensive troubleshooting guide organized by symptom, covering error messages, unexpected behaviors, and edge cases across all system components.

---

## 15.1 Troubleshooting by Symptom

### Quick Diagnosis Flow

```
SYMPTOM                           LIKELY CAUSE                    SECTION
─────────────────────────────────────────────────────────────────────────
Bot not starting                  Config error, missing deps      15.2
Bot running but not trading       Risk halt, gate closed          15.3
Trades not filling                Spread too wide, filters        15.4
Unexpected losses                 Slippage, funding, fees         15.5
Metrics not appearing             Port conflict, server not up    15.6
State file corrupted              Crash during write              15.7
Backtest fails                    Data format, missing columns    15.8
Optimization hangs                DB lock, memory                 15.9
Docker issues                     Permissions, networking         15.10
```

---

## 15.2 Bot Startup Issues

### Error: ModuleNotFoundError

```
ModuleNotFoundError: No module named 'binance'
```

**Cause:** Dependencies not installed.

**Fix:**
```bash
pip install -r requirements.txt
```

---

### Error: ValidationError (Pydantic)

```
pydantic.error_wrappers.ValidationError: 1 validation error for Strategy
flip_band_entry
  ensure this value is greater than 0 (type=value_error)
```

**Cause:** Invalid configuration value.

**Fix:** Check the field mentioned in error. Verify value is within valid range in `core/config_schema.py`.

---

### Error: FileNotFoundError (Config)

```
FileNotFoundError: [Errno 2] No such file or directory: 'configs/prod.json'
```

**Cause:** Config file path incorrect.

**Fix:**
```bash
ls configs/
# Verify file exists, check spelling
```

---

### Error: API Key Not Found

```
Error: BINANCE_KEY not found in environment.
```

**Cause:** Environment variables not set.

**Fix:**
```bash
# Check .env file exists and is sourced
cat .env | grep BINANCE_KEY

# For Docker, verify docker-compose.yml env_file directive
```

---

### Error: Connection Refused (Testnet)

```
binance.exceptions.BinanceAPIException: Invalid API-key, IP, or permissions
```

**Cause:** Using live keys on testnet URL or vice versa.

**Fix:**
- Testnet Spot: `https://testnet.binance.vision`
- Testnet Futures: `https://testnet.binancefuture.com`
- Live: `https://api.binance.com`

Ensure keys match the URL.

---

## 15.3 Bot Running But Not Trading

### Symptom: No trades for hours

**Diagnosis Checklist:**

1. **Check risk halt status:**
   ```bash
   curl localhost:9109/metrics | grep risk_flags
   # risk_flags{kind="maxdd_hit"} 1.0  <- Trading halted!
   ```

2. **Check gate state:**
   ```bash
   curl localhost:9109/metrics | grep gate_state
   # gate_state{gate_state="closed"} 1.0  <- Gate closed
   ```

3. **Check signal proximity:**
   ```bash
   curl localhost:9109/metrics | grep next_action_dist_bps
   # next_action_dist_bps 234.5  <- 234 bps away from trigger
   ```

4. **Check cooldown:**
   ```bash
   docker logs bot_eth | grep -i "cooldown"
   ```

---

### Max DD Hit - Trading Halted

**Cause:** Drawdown exceeded `max_dd_frac` threshold.

**Options:**
1. **Wait for Phoenix reset** (if configured)
2. **Manual reset:**
   ```bash
   # Edit state file
   cat /data/state_live.json | jq '.maxdd_hit = false' > tmp.json
   mv tmp.json /data/state_live.json
   docker restart bot_eth
   ```

---

### Gate Closed (Mean Reversion)

**Cause:** Gate ROC threshold not met.

**Verify:**
```bash
curl localhost:9109/metrics | grep gate_state
```

**Wait:** Gate will open when long-term trend ROC exceeds `gate_roc_threshold`.

---

### Daily Loss Limit Hit

**Cause:** Daily loss exceeded `max_daily_loss_btc` or `max_daily_loss_frac`.

**Wait:** Limit resets at UTC midnight automatically.

**Verify:**
```bash
curl localhost:9109/metrics | grep daily_limit
```

---

## 15.4 Trades Not Filling

### Symptom: Orders placed but expire unfilled

**Cause 1: Spread too wide**
```bash
curl localhost:9109/metrics | grep spread_bps
# spread_bps 15.0  <- If > max_spread_bps, taker fallback skipped
```

**Fix:** Increase `max_spread_bps_for_taker` in config or wait for tighter markets.

---

**Cause 2: LIMIT_MAKER rejected**
```
Order would immediately trigger
```

**Explanation:** Post-only order crossed the spread.

**Fix:** This is normal behavior. Order will retry on next bar.

---

**Cause 3: MIN_NOTIONAL filter**
```
binance.exceptions.BinanceAPIException: Filter failure: MIN_NOTIONAL
```

**Cause:** Trade size below exchange minimum.

**Fix:** Increase `min_trade_btc` or increase capital.

---

**Cause 4: LOT_SIZE filter**
```
binance.exceptions.BinanceAPIException: Filter failure: LOT_SIZE
```

**Cause:** Quantity precision incorrect.

**Fix:** This should be auto-handled. If persists, check `get_filters()` returns correct `step_size`.

---

## 15.5 Unexpected Losses

### Symptom: Live PnL worse than backtest

**Common Causes:**

1. **Slippage underestimated**
   - Backtest uses `slippage_bps` fixed estimate
   - Live may experience worse fills
   - **Fix:** Increase `slippage_bps` in backtest (2-5 bps)

2. **Funding costs (Futures)**
   - Backtest may not include funding
   - **Fix:** Include `--funding-data` in backtests

3. **Taker fills instead of maker**
   - Check `max_taker_btc` settings
   - Review trade logs for taker fills

4. **Different market conditions**
   - Strategy optimized on past data
   - **Verify:** Run recent backtest, compare to live

---

### Symptom: Sudden large loss

**Diagnosis:**
```bash
# Check trade log
tail -20 /data/trade_log.jsonl

# Check if max DD was hit
curl localhost:9109/metrics | grep maxdd_hit

# Check regime state
curl localhost:9109/metrics | grep regime_state
```

---

## 15.6 Metrics Issues

### Symptom: Metrics endpoint not responding

```
curl: (7) Failed to connect to localhost port 9109
```

**Cause 1:** Metrics server not started.

**Fix:** Ensure executor calls `start_metrics_server(9109)`.

---

**Cause 2:** Port conflict.

**Fix:**
```bash
lsof -i :9109
# Kill conflicting process or use different port
```

---

### Symptom: Prometheus not scraping

**Check Prometheus targets:**
```
http://localhost:9090/targets
```

**If DOWN:**
1. Verify container networking
2. Check `prometheus.yml` has correct target hostname
3. For Docker: use container name, not `localhost`

---

### Symptom: Grafana empty dashboards

**Causes:**
1. Prometheus data source not configured
2. Query syntax error
3. No data in time range

**Fix:**
1. Configuration → Data Sources → Add Prometheus
2. URL: `http://prometheus:9090` (for Docker networking)
3. Test with: `up{instance=~".*"}`

---

## 15.7 State File Issues

### Error: JSONDecodeError

```
json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
```

**Cause:** Corrupted state file (crash during write).

**Fix:**
```bash
# Backup corrupted file
mv /data/state_live.json /data/state_live.json.corrupted

# Create fresh state
echo '{}' > /data/state_live.json

# Or restore from backup
cp /data/state_live.json.backup /data/state_live.json

# Restart bot
docker restart bot_eth
```

---

### State drift from exchange

**Symptom:** PnL reconciler shows divergence > 5%.

**Diagnosis:**
```bash
python tools/reconcile_pnl.py
```

**Fix:**
1. Stop bot
2. Get actual balances from Binance
3. Update state file manually:
   ```bash
   jq '.session_start_W = 1.045' /data/state_live.json > tmp.json
   mv tmp.json /data/state_live.json
   ```
4. Restart bot

---

## 15.8 Backtest Issues

### Error: close_time column not found

```
ValueError: close_time column not found
```

**Cause:** CSV format doesn't match expected Vision format.

**Fix:**
```python
# Check actual column names
import pandas as pd
df = pd.read_csv("data.csv")
print(df.columns.tolist())
```

Rename columns or use correct data source.

---

### Error: Need OHLC

```
ValueError: Need OHLC
```

**Cause:** Strategy requires full OHLC but only close was passed.

**Fix:** Ensure backtest passes `full_df=df` parameter.

---

### Symptom: Backtest returns 0 trades

**Causes:**
1. Bands too wide (flip_band_entry too high)
2. Gate always closed
3. Funding filter rejecting all signals

**Diagnosis:**
```bash
# Run with diagnostics
python core/ethbtc_accum_bot.py backtest \
  --data data.csv \
  --config config.json \
  --out results/debug.csv

# Check target_w column - all zeros?
```

---

### Symptom: Backtest memory error

```
MemoryError: Unable to allocate...
```

**Cause:** Dataset too large for available memory.

**Fix:**
```bash
# Use date slicing
--start 2024-01-01 --end 2024-12-31

# Or use 30m/1h timeframe instead of 15m
```

---

## 15.9 Optimization Issues

### Error: Database is locked

```
sqlite3.OperationalError: database is locked
```

**Cause:** Multiple processes accessing SQLite simultaneously.

**Fix:**
1. Use `--jobs 1` to disable parallelism
2. Or switch to PostgreSQL for multi-process

---

### Symptom: Optimization finds no trades

```
Trial N DONE: Score=-1000.0 (no trades)
```

**Cause:** All parameter combinations produce zero-trade strategies.

**Fix:**
1. Widen search ranges
2. Reduce penalty weights (`--lambda-turns 0.5`)
3. Check if data contains sufficient price movement

---

### Symptom: Optimization extremely slow

**Causes:**
1. Too many trials
2. Large dataset
3. Slow storage

**Fix:**
```bash
# Reduce trials for initial testing
--n-trials 20

# Use date subset
--start 2023-01-01 --end 2024-01-01

# Use in-memory SQLite
--storage sqlite:///:memory:
```

---

## 15.10 Docker Issues

### Error: Permission denied

```
PermissionError: [Errno 13] Permission denied: '/data/state.json'
```

**Cause:** Volume mount permissions mismatch.

**Fix:**
```bash
# On host
sudo chown -R 1000:1000 ./run_state/

# Or in docker-compose.yml
user: "${UID}:${GID}"
```

---

### Error: Container exits immediately

**Diagnosis:**
```bash
docker logs bot_eth
```

**Common causes:**
1. Missing environment variables
2. Config file not mounted
3. Python exception

---

### Symptom: Container can't reach exchange

```
requests.exceptions.ConnectionError: Failed to establish connection
```

**Fix:**
```bash
# From inside container
docker exec -it bot_eth ping api.binance.com

# If fails, check Docker network
docker network inspect ethbtc_bot_3_default
```

---

### Symptom: Prometheus can't reach bot

**Fix:** Use container name as hostname in `prometheus.yml`:
```yaml
static_configs:
  - targets: ['bot_eth:9109']  # Not localhost!
```

---

## 15.11 Edge Cases

### Edge Case: Market closed (maintenance)

**Symptom:** API returns errors during Binance maintenance.

**Behavior:** Circuit breaker triggers, bot waits for API recovery.

**No action needed:** Bot auto-recovers when API returns.

---

### Edge Case: Extreme volatility

**Symptom:** Multiple trades in quick succession, high slippage.

**Protection:**
1. Cooldown prevents rapid flipping
2. Volatility sizing reduces position in high vol
3. Max DD halts if losses accumulate

---

### Edge Case: Funding rate spike

**Symptom:** Trading halted due to funding filter.

**Behavior:** `funding_limit_long` / `funding_limit_short` prevent trades during extreme funding.

**No action needed:** Wait for funding to normalize.

---

### Edge Case: Flash crash

**Bot behavior:**
1. May trigger BUY signal at flash low
2. If fill occurs, position acquired at discount
3. If Max DD hit, trading halts

**Post-event:**
1. Check trade log
2. Verify PnL
3. Reset state if needed

---

### Edge Case: API rate limit

```
binance.exceptions.BinanceAPIException: -1003 Too many requests
```

**Protection:** `@with_retry` decorator with exponential backoff.

**If persistent:**
1. Reduce polling frequency
2. Check for runaway loops

---

## 15.12 Getting Help

### Information to Collect

When reporting issues, include:

1. **Log output:**
   ```bash
   docker logs bot_eth --tail 200 > debug_logs.txt
   ```

2. **Metrics snapshot:**
   ```bash
   curl localhost:9109/metrics > metrics_snapshot.txt
   ```

3. **State file:**
   ```bash
   cat /data/state_live.json
   ```

4. **Config (sanitized):**
   ```bash
   cat configs/prod.json | jq 'del(.api_key, .api_secret)'
   ```

5. **Environment:**
   ```bash
   python --version
   pip freeze | grep binance
   docker --version
   ```

---

*Previous Chapter: [Chapter 14: Workflows & Recipes](./CHAPTER_14_WORKFLOWS.md)*  
*Next Chapter: [Appendix A: Configuration Reference](./APPENDIX_A_CONFIG_REFERENCE.md)*
