# Observability & Story Generation Guide

## 📊 New Grafana Metrics

### Phoenix Protocol Status
**Metric**: `phoenix_active`
- **Value**: `1.0` = Bot is halted, waiting for Phoenix reset conditions
- **Value**: `0.0` = Bot is trading normally

**Grafana Query**:
```promql
phoenix_active{job="ethbtc_bot"}
```

**Alert Example** (notify when Phoenix activates):
```promql
phoenix_active == 1
```

### Safety Breaker Status
**Metric**: `risk_flags{kind="maxdd_hit"}`
- **Value**: `1.0` = Max drawdown hit, trading halted
- **Value**: `0.0` = Normal operation

---

## 📖 Story Generation with `explain_story.py`

### What It Does
Generates a human-readable narrative from backtest or live CSV data, showing:
- 🚀 New All-Time Highs
- 🔄 Regime Switches (Trend ↔ Mean Reversion)
- 🟢 Buy / 🔴 Sell Trades
- 🚨 Safety Breaker Trips
- 🔥 Phoenix Protocol Activations
- 📈 Monthly PnL Reports

### Usage

#### For Backtest Results
```bash
python tools/explain_story.py results/backtest_output.csv --threshold 30.0 --out results/story.txt
```

#### For Live Trading (Export from Prometheus/Grafana)
1. Export metrics to CSV from Grafana dashboard
2. Run:
```bash
python tools/explain_story.py results/live_export.csv --threshold 30.0 --out results/live_story.txt
```

### Example Output
```
📖 Reading Story from: results/btc_story.txt
🔍 Analyzing 1440 bars with Threshold 30.0...

======================================================================
TIMESTAMP            | EVENT                                         | DETAILS
======================================================================
2024-01-15 08:00:00  | 🔄 REGIME SWITCH: TREND                       | Score: 35.2
2024-01-15 09:15:00  | 🟢 BUY                                        | Incr Exp 0.00->0.50 @ 42500.00
2024-01-20 14:30:00  | 🚀 NEW ATH REACHED                            | 1.0523 BTC
2024-02-05 11:00:00  | 🚨 SAFETY BREAKER TRIPPED                     | DD: -21.5%
2024-02-12 16:45:00  | 🔥 PHOENIX PROTOCOL ACTIVATED                 | Resuming Trades
======================================================================
FINAL RESULT: 1.1234 BTC
======================================================================
```

---

## 🔧 Automating Live Story Generation

To generate a live story file automatically, you can add a cron job or systemd timer:

### Option 1: Cron (Every Hour)
```bash
0 * * * * cd /path/to/ethbtc_bot_3 && python tools/explain_story.py results/live_export.csv --out results/live_story.txt
```

### Option 2: Add to Docker (Future Enhancement)
We could modify `live_executor.py` to write a live story file on each iteration, but this would add I/O overhead. Better to export from Prometheus periodically.

---

## 📈 Grafana Dashboard Recommendations

### Panel 1: Phoenix & Safety Status
**Type**: Stat
**Query**:
```promql
phoenix_active + (risk_flags{kind="maxdd_hit"} * 10)
```
**Thresholds**:
- 0 = Green (Normal)
- 1 = Yellow (Phoenix Active)
- 10 = Red (Safety Breaker)

### Panel 2: Regime Score
**Type**: Gauge
**Query**:
```promql
regime_score
```
**Thresholds**:
- 0-25 = Blue (Mean Reversion)
- 25-100 = Orange (Trend)

### Panel 3: Strategy Mode
**Type**: Stat
**Query**:
```promql
strategy_mode
```
**Value Mappings**:
- 0 = "Mean Reversion"
- 1 = "Trend Following"
