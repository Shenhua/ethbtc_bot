# Chapter 10: Monitoring & Observability

> **Purpose:** This chapter provides exhaustive documentation of the monitoring and observability stack, covering Prometheus metrics, Grafana dashboards, alerting via Discord/Telegram, and structured logging.

---

## 10.1 Observability Architecture

### 1. Concept & "The Why"

* **What it is:** A comprehensive monitoring stack that provides real-time visibility into bot health, trading activity, and risk state.

* **Purpose:** 
  1. **Detect issues early:** Know when something is wrong before it costs money
  2. **Performance tracking:** Measure PnL, exposure, and execution quality
  3. **Debugging:** Correlate events across components with structured logs
  4. **Alerting:** Get notified of critical events (max DD hit, errors)

* **Location:** 
  - Metrics: [`core/metrics.py`](../../core/metrics.py)
  - Alerts: [`core/alert_manager.py`](../../core/alert_manager.py)
  - Logging: [`core/log_setup.py`](../../core/log_setup.py)
  - Prometheus Config: [`prometheus.yml`](../../prometheus.yml)
  - Docker Compose: [`docker-compose.yml`](../../docker-compose.yml)

### 2. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         TRADING BOT                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐ │
│  │  live_executor  │  │    Metrics      │  │   AlertManager      │ │
│  │                 │──▶│  (Port 9109)   │  │                     │ │
│  │  Trading Logic  │  │  30+ Gauges     │  │  Discord Webhook    │ │
│  │                 │  │  Counters       │  │  Telegram Bot       │ │
│  └─────────────────┘  └────────┬────────┘  └──────────┬──────────┘ │
└────────────────────────────────┼──────────────────────┼─────────────┘
                                 │                      │
                                 ▼                      ▼
┌─────────────────────────────────────┐    ┌─────────────────────────┐
│           PROMETHEUS                │    │    DISCORD / TELEGRAM   │
│           (Port 9090)               │    │                         │
│  ┌────────────────────────────────┐ │    │  ⚠️ Max DD Hit!         │
│  │  Scrapes /metrics every 5s    │ │    │  📈 Trade Executed       │
│  │  Time series database         │ │    │  🔄 Phoenix Reset        │
│  └────────────────────────────────┘ │    └─────────────────────────┘
└─────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│            GRAFANA                  │
│           (Port 3000)               │
│                                     │
│  📊 Dashboards                      │
│  📈 Graphs                          │
│  🚨 Alerting Rules                  │
└─────────────────────────────────────┘
```

---

## 10.2 Prometheus Metrics

### 1. Concept & "The Why"

* **What it is:** A collection of 30+ Prometheus metrics exposed on port 9109, providing real-time telemetry about bot health, trading activity, and risk state.

* **Purpose:** Enables time-series analysis of bot performance and integration with Grafana for visualization.

* **Location:** [`core/metrics.py`](../../core/metrics.py)

### 2. Available Metrics

#### Health & Status

| Metric | Type | Description |
|--------|------|-------------|
| `up` | Gauge | Bot health status (1=up, 0=down) |
| `bar_latency_seconds` | Summary | Latency processing each bar |

#### Trading Activity

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `orders_submitted_total` | Counter | instance, kind, side | Orders submitted |
| `fills_total` | Counter | instance | Executed fills |
| `rejections_total` | Counter | instance, reason | Order rejections |
| `skips_total` | Counter | instance, reason | Trade skips by reason |

#### Performance

| Metric | Type | Description |
|--------|------|-------------|
| `pnl_quote` | Gauge | PnL since session start |
| `wealth_total` | Gauge | Total portfolio value |
| `wealth_usd` | Gauge | Portfolio value in USD |
| `price_mid` | Gauge | Mid price of traded pair |
| `balance_free` | Gauge (asset) | Free balance by asset |

#### Exposure & Position

| Metric | Type | Description |
|--------|------|-------------|
| `exposure_base_weight` | Gauge (kind) | Target/current weights |
| `exposure_signal_weight` | Gauge | Unleveraged signal weight |
| `exposure_notional` | Gauge | Leveraged notional % |
| `leverage` | Gauge | Current leverage multiplier |
| `position_step_size` | Gauge | Dynamic position step |

#### Signal & Strategy

| Metric | Type | Description |
|--------|------|-------------|
| `signal_ratio` | Gauge | Current ratio signal |
| `signal_band` | Gauge (kind) | Entry/exit thresholds |
| `dist_to_buy_bps` | Gauge | Distance to BUY entry |
| `dist_to_sell_bps` | Gauge | Distance to SELL entry |
| `regime_score` | Gauge | ADX-based regime score |
| `regime_state` | Gauge | -1=MR, +1=Trend |
| `strategy_mode` | Gauge | 0=MR, 1=Trend |

#### Risk & State

| Metric | Type | Description |
|--------|------|-------------|
| `risk_mode` | Gauge (mode) | Active risk mode |
| `risk_flags` | Gauge (kind) | daily_limit_hit, maxdd_hit |
| `phoenix_active` | Gauge | 1=waiting for reset |
| `funding_rate_pct` | Gauge | Current funding rate |
| `gate_state` | Gauge (state) | open/closed |
| `trade_ready` | Gauge | Overall readiness (1=OK) |
| `trade_ready_condition` | Gauge (cond) | Sub-condition status |

#### Execution Quality

| Metric | Type | Description |
|--------|------|-------------|
| `spread_bps` | Gauge | Current spread |
| `execution_slippage_bps` | Summary | Trade slippage |
| `fees_paid_total` | Counter (asset) | Cumulative fees |
| `last_trade_timestamp_seconds` | Gauge | Last trade time |
| `margin_utilization_pct` | Gauge | Futures margin used |
| `liquidation_distance_pct` | Gauge (symbol) | Distance to liquidation |

### 3. Step-by-Step Guide: Accessing Metrics

1. **View metrics in browser:**
   ```
   http://localhost:9109/metrics
   ```

2. **Sample output:**
   ```
   # HELP up Bot instance health status
   # TYPE up gauge
   up{instance="ethbtc_live"} 1.0
   
   # HELP wealth_total Total wealth in Quote Asset units
   # TYPE wealth_total gauge
   wealth_total{instance="ethbtc_live"} 1.2345
   
   # HELP regime_score Current Trend Consensus Score (0-100)
   # TYPE regime_score gauge
   regime_score{instance="ethbtc_live"} 23.45
   ```

3. **Query in Prometheus:**
   ```promql
   # Current wealth
   wealth_total{instance="ethbtc_live"}
   
   # PnL over time
   pnl_quote{instance="ethbtc_live"}
   
   # Trade count by side
   rate(orders_submitted_total{instance="ethbtc_live"}[1h])
   ```

---

## 10.3 Prometheus Configuration

### 1. Concept & "The Why"

* **What it is:** Configuration for Prometheus scrape jobs that collect metrics from bot instances.

* **Purpose:** Enables Prometheus to automatically discover and scrape bot metrics.

* **Location:** [`prometheus.yml`](../../prometheus.yml)

### 2. Configuration File

```yaml
global:
  scrape_interval: 5s  # Collect metrics every 5 seconds

scrape_configs:
  - job_name: 'ethbtc_bot'
    honor_labels: true  # Preserve bot's instance label
    static_configs:
      - targets: ['bot_eth:9109']

  - job_name: 'btcusdt_bot'
    honor_labels: true
    static_configs:
      - targets: ['bot_btc:9109']
```

### 3. Step-by-Step Guide: Adding a New Bot

1. **Add new job to `prometheus.yml`:**
   ```yaml
   - job_name: 'ltcbtc_bot'
     honor_labels: true
     static_configs:
       - targets: ['bot_ltc:9109']
   ```

2. **Restart Prometheus:**
   ```bash
   docker-compose restart prometheus
   ```

3. **Verify in Prometheus UI:**
   - Open `http://localhost:9090/targets`
   - Check new target is "UP"

---

## 10.4 Alert Manager

### 1. Concept & "The Why"

* **What it is:** A multi-channel alerting system that sends notifications to Discord and Telegram.

* **Purpose:** Immediate notification of critical events—Max DD hit, errors, or important trades.

* **Location:** [`core/alert_manager.py`](../../core/alert_manager.py)

### 2. Configuration

#### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `DISCORD_WEBHOOK_URL` | Discord webhook URL | `https://discord.com/api/webhooks/...` |
| `TELEGRAM_TOKEN` | Telegram bot token | `123456:ABC-...` |
| `TELEGRAM_CHAT_ID` | Telegram chat ID | `-1001234567890` |

#### Alert Levels

| Level | Discord Color | Use Case |
|-------|---------------|----------|
| `INFO` | Blue | Trades, status updates |
| `WARNING` | Yellow | Near-limit events, retries |
| `ERROR` | Red | Failures, exceptions |
| `CRITICAL` | Dark Red | Max DD hit, system down |

### 3. Step-by-Step Guide: Setting Up Discord Alerts

1. **Create Discord webhook:**
   - Open Discord server settings → Integrations → Webhooks
   - Create new webhook → Copy URL

2. **Add to `.env`:**
   ```bash
   DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/1234567890/abcdef...
   ```

3. **Test alert:**
   ```python
   from core.alert_manager import AlertManager
   
   alerter = AlertManager(prefix="TEST")
   alerter.send("Hello from bot!", level="INFO")
   ```

### 4. Step-by-Step Guide: Setting Up Telegram Alerts

1. **Create Telegram bot:**
   - Message @BotFather → `/newbot` → Follow prompts
   - Copy token

2. **Get chat ID:**
   - Add bot to group or start DM
   - Visit `https://api.telegram.org/bot<TOKEN>/getUpdates`
   - Find `chat.id` in response

3. **Add to `.env`:**
   ```bash
   TELEGRAM_TOKEN=123456:ABC-DEF123456...
   TELEGRAM_CHAT_ID=-1001234567890
   ```

### 5. Alert Code Integration

```python
from core.alert_manager import AlertManager

alerter = AlertManager(prefix="ETHBTC")

# In trading loop
if state.risk.maxdd_hit and not state.alert_sent_maxdd:
    alerter.send("🚨 MAX DRAWDOWN HIT - Trading halted!", level="CRITICAL")
    state.alert_sent_maxdd = True

# On trade execution
alerter.send(f"📈 BUY 0.5 ETH @ 0.034 BTC", level="INFO")
```

### 6. Troubleshooting

**Discord rate limiting:**
```
Discord Alert Failed (Attempt 1): 429 Rate Limited
```
**Fix:** AlertManager has built-in 3-retry with 2s backoff for 429s.

**Telegram failures:**
Check token and chat ID are correct. Bot must be added to group/DM first.

---

## 10.5 Structured Logging

### 1. Concept & "The Why"

* **What it is:** Production-grade structured logging using `structlog`, supporting both human-readable console output and JSON for log aggregation.

* **Purpose:** 
  - Consistent log format across all modules
  - Easy parsing for log aggregation (ELK, Loki)
  - Context binding for request-scoped data

* **Location:** [`core/log_setup.py`](../../core/log_setup.py)

### 2. Configuration

#### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LOGLEVEL` | `INFO` | Log level (DEBUG, INFO, WARNING, ERROR) |
| `LOG_JSON` | `false` | Output JSON format (for production) |
| `BINANCE_LOG_LEVEL` | `WARNING` | Binance SDK log level |

### 3. Usage

```python
from core.log_setup import get_logger, bind_context

log = get_logger(__name__)

# Simple logging
log.info("Bot started")

# Structured logging with context
log.info("Trade executed", symbol="ETHBTC", side="BUY", qty=0.5, price=0.034)

# Bind persistent context
bind_context(instance="ethbtc_live", mode="production")
log.info("Starting loop")  # Includes instance and mode automatically
```

### 4. Output Formats

**Console (development):**
```
2026-01-05T14:30:00 [INFO    ] live_executor: Trade executed  symbol=ETHBTC side=BUY qty=0.5 price=0.034
```

**JSON (production with `LOG_JSON=true`):**
```json
{
  "timestamp": "2026-01-05T14:30:00.123456",
  "level": "info",
  "logger": "live_executor",
  "event": "Trade executed",
  "symbol": "ETHBTC",
  "side": "BUY",
  "qty": 0.5,
  "price": 0.034
}
```

### 5. Step-by-Step Guide: Enabling Debug Logging

```bash
# Set environment variable
export LOGLEVEL=DEBUG

# Run bot
python live_executor.py --params configs/prod.json --mode dry
```

---

## 10.6 Grafana Dashboards

### 1. Concept & "The Why"

* **What it is:** Grafana provides visualization of Prometheus metrics with customizable dashboards.

* **Purpose:** Visual monitoring, performance tracking, and historical analysis.

* **Location:** Docker Compose config in [`docker-compose.yml`](../../docker-compose.yml)

### 2. Docker Configuration

```yaml
grafana:
  image: grafana/grafana:latest
  container_name: grafana
  ports:
    - "3000:3000"
  volumes:
    - ./grafana_data:/var/lib/grafana
  environment:
    - GF_SECURITY_ADMIN_PASSWORD=admin123
```

### 3. Step-by-Step Guide: Creating Dashboard

1. **Access Grafana:**
   ```
   http://localhost:3000
   Username: admin
   Password: admin123
   ```

2. **Add Prometheus data source:**
   - Configuration → Data Sources → Add
   - Select "Prometheus"
   - URL: `http://prometheus:9090`
   - Save & Test

3. **Create dashboard panels:**

   **Panel 1: Portfolio Value**
   ```promql
   wealth_total{instance=~"$instance"}
   ```

   **Panel 2: PnL**
   ```promql
   pnl_quote{instance=~"$instance"}
   ```

   **Panel 3: Regime Score**
   ```promql
   regime_score{instance=~"$instance"}
   ```

   **Panel 4: Trade Rate**
   ```promql
   rate(orders_submitted_total{instance=~"$instance"}[5m])
   ```

### 4. Recommended Dashboard Layout

```
┌─────────────────────────────────────────────────────────────┐
│                    BOT OVERVIEW                             │
├─────────────────┬─────────────────┬─────────────────────────┤
│   Wealth (BTC)  │    PnL (BTC)    │     Regime Score        │
│   [Line Chart]  │   [Line Chart]  │     [Gauge 0-100]       │
├─────────────────┴─────────────────┴─────────────────────────┤
│                    EXPOSURE & POSITION                      │
├─────────────────────────────────────────────────────────────┤
│   Target Weight vs Current Weight    │   Position Step Size │
│   [Dual Line Chart]                  │   [Single Stat]      │
├─────────────────────────────────────────────────────────────┤
│                    RISK INDICATORS                          │
├─────────────────┬─────────────────┬─────────────────────────┤
│  Max DD Flag    │   Daily Limit   │     Phoenix Active      │
│  [Status Panel] │  [Status Panel] │     [Status Panel]      │
├─────────────────┴─────────────────┴─────────────────────────┤
│                    TRADING SIGNALS                          │
├─────────────────────────────────────────────────────────────┤
│   Signal Ratio vs Bands            │   Funding Rate         │
│   [Line Chart with thresholds]     │   [Line Chart]         │
└─────────────────────────────────────────────────────────────┘
```

---

## 10.7 Complete Monitoring Setup

### Real-World Use Case (The "Cookbook")

**Scenario:** Set up full monitoring stack for production deployment.

**Step 1: Configure `.env`**
```bash
# Metrics
METRICS_PORT=9109
STATUS_PORT=9110

# Alerts
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
TELEGRAM_TOKEN=123456:ABC...
TELEGRAM_CHAT_ID=-1001234567890

# Logging
LOGLEVEL=INFO
LOG_JSON=true
```

**Step 2: Start stack**
```bash
docker-compose up -d prometheus grafana bot_eth bot_btc
```

**Step 3: Verify metrics**
```bash
# Check bot metrics
curl http://localhost:9109/metrics | grep wealth_total

# Check Prometheus targets
open http://localhost:9090/targets
```

**Step 4: Create Grafana dashboard**
- Import dashboard JSON or create manually
- Set up alert rules for critical metrics

**Step 5: Test alerting**
```python
python -c "
from core.alert_manager import AlertManager
alerter = AlertManager(prefix='TEST')
alerter.send('Monitoring setup complete!', level='INFO')
"
```

**Expected Outcome:**
- All metrics visible in Prometheus
- Grafana dashboards showing real-time data
- Discord/Telegram alerts working
- JSON logs ready for aggregation

---

## 10.8 Key PromQL Queries

### Performance Monitoring

```promql
# Portfolio value over time
wealth_total{instance="ethbtc_live"}

# Daily PnL
pnl_quote{instance="ethbtc_live"}

# Trade count in last hour
increase(orders_submitted_total{instance="ethbtc_live"}[1h])

# Execution slippage p99
histogram_quantile(0.99, execution_slippage_bps{instance="ethbtc_live"})
```

### Risk Monitoring

```promql
# Max DD hit alert (returns 1 when triggered)
risk_flags{instance="ethbtc_live", kind="maxdd_hit"} == 1

# Phoenix waiting for reset
phoenix_active{instance="ethbtc_live"} == 1

# Margin utilization (Futures)
margin_utilization_pct{instance="btcusdt_live"}
```

### Signal Analysis

```promql
# Current regime (MR=-1, Trend=1)
regime_state{instance="ethbtc_live"}

# Distance to nearest trade trigger
next_action_dist_bps{instance="ethbtc_live"}

# Funding rate
funding_rate_pct{instance="btcusdt_live"}
```

---

## 10.9 Troubleshooting

### Metrics Not Appearing

```
Error: connection refused localhost:9109
```
**Cause:** Metrics server not started.
**Fix:** Ensure `start_metrics_server(9109)` is called in executor.

### Prometheus Not Scraping

Check `http://localhost:9090/targets`:
- State should be "UP"
- If "DOWN", check network connectivity between containers

### Alerts Not Sending

**Discord timeouts:**
```
Discord Alert Failed (Attempt 3): Connection timeout
```
**Fix:** Check network, webhook URL validity.

**Telegram silent:**
- Verify bot is added to group/DM
- Check `TELEGRAM_CHAT_ID` is correct (include `-` for groups)

### Log Level Issues

Debug logs not appearing:
```bash
# Verify environment variable
echo $LOGLEVEL

# Should be DEBUG, not INFO
export LOGLEVEL=DEBUG
```

---

*Previous Chapter: [Chapter 9: Walk-Forward Optimization](./CHAPTER_09_OPTIMIZATION.md)*  
*Next Chapter: [Chapter 11: Live Trading Guide](./CHAPTER_11_LIVE_TRADING.md)*
