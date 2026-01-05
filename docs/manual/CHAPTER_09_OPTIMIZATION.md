# Chapter 9: Walk-Forward Optimization

> **Purpose:** This chapter provides exhaustive documentation of the Walk-Forward Optimization (WFO) system, covering the Optuna-based optimizer, search spaces, scoring functions, rolling window methodology, and result analysis.

---

## 9.1 WFO Architecture

### 1. Concept & "The Why"

* **What it is:** Walk-Forward Optimization (WFO) is a rolling window optimization technique that mimics real trading: optimize on past data, test on future data, then roll forward and repeat.

* **Purpose:** 
  1. **Prevents overfitting:** Parameters are always tested on unseen data
  2. **Adapts to regime changes:** Parameters are re-optimized as markets evolve
  3. **Realistic validation:** Results reflect what a live trader would experience

* **Location:** 
  - Mean Reversion: [`tools/optimizer_cli.py`](../../tools/optimizer_cli.py)
  - Trend Strategy: [`tools/optimize_trend.py`](../../tools/optimize_trend.py)
  - Meta Strategy: [`tools/optimize_meta.py`](../../tools/optimize_meta.py)

### 2. Configuration & Parameters

#### Common CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | (required) | Path to OHLCV CSV |
| `--funding-data` | — | Path to funding rate CSV |
| `--config` | — | Base config JSON |
| `--wfo` | — | Enable WFO mode |
| `--window-days` | 180 | Training window size in days |
| `--step-days` | 30 | Step size for re-optimization |
| `--n-trials` | 200 | Optuna trials per window |
| `--jobs` | 1 | Parallel jobs |
| `--storage` | `sqlite:///data/db/optuna.db` | Optuna study storage |
| `--study-name` | varies | Study identifier |
| `--out` | varies | Output CSV path |

### 3. WFO Flow Diagram

```
Full Dataset (4 years of data)
├──────────────────────────────────────────────────────────────────────────┤
│       YEAR 1       │       YEAR 2       │       YEAR 3       │   YEAR 4 │
├──────────────────────────────────────────────────────────────────────────┤

Window 1 (t=0):
├─────────[ TRAIN 180d ]─────────┤────[ TEST 30d ]────┤
                                  ↓
                            Best params for Month 1

Window 2 (t=30d):
     ├─────────[ TRAIN 180d ]─────────┤────[ TEST 30d ]────┤
                                       ↓
                                 Best params for Month 2

Window 3 (t=60d):
          ├─────────[ TRAIN 180d ]─────────┤────[ TEST 30d ]────┤
                                            ↓
                                      Best params for Month 3

... continues until data ends ...
```

---

## 9.2 Mean Reversion Optimizer

### 1. Concept & "The Why"

* **What it is:** Bayesian optimization of Mean Reversion strategy parameters using Optuna's TPE sampler.

* **Purpose:** Find optimal flip bands, volatility settings, and position sizing for MR strategy.

* **Location:** [`tools/optimizer_cli.py`](../../tools/optimizer_cli.py)

### 2. Search Space

| Parameter | Search Range | Type | Description |
|-----------|--------------|------|-------------|
| `trend_kind` | `["sma", "roc"]` | categorical | Trend calculation method |
| `trend_lookback` | `[120, 160, 200, 240, 300]` | categorical | MA lookback bars |
| `flip_band_entry` | `0.01 – 0.06` | float | Entry threshold (%) |
| `flip_band_exit` | `0.005 – 0.03` | float | Exit threshold (%) |
| `vol_window` | `[45, 60, 90]` | categorical | Volatility window |
| `vol_adapt_k` | `[0.0, 0.0025, 0.005, 0.0075]` | categorical | Vol adaptation factor |
| `target_vol` | `[0.3, 0.4, 0.5, 0.6]` | categorical | Target volatility |
| `cooldown_minutes` | `[60, 120, 180, 240]` | categorical | Min time between flips |
| `step_allocation` | `[0.33, 0.5, 0.66, 1.0]` | categorical | Position step size |
| `max_position` | `[0.6, 0.8, 1.0]` | categorical | Max position |
| `position_sizing_mode` | `["static", "volatility"]` | categorical | Sizing algorithm |
| `position_sizing_target_vol` | `0.3 – 0.7` | float | Sizing target vol |
| `position_sizing_min_step` | `0.1 – 0.3` | float | Min step |
| `gate_window_days` | `[30, 60, 90]` | categorical | Gate lookback |
| `gate_roc_threshold` | `[0.0, 0.01, 0.02]` | categorical | Gate ROC threshold |
| `funding_limit_long` | `0.01 – 0.10` | float | Long funding limit |
| `funding_limit_short` | `-0.10 – -0.01` | float | Short funding limit |
| `rebalance_threshold_w` | `[0.0, 0.01]` | categorical | Rebalance threshold |
| `long_only` | `[True, False]` | categorical | Long-only mode |

### 3. Scoring Function

```python
# Robust Score Formula
if turns == 0:
    robust_score = -1000.0  # Massive penalty for not trading
else:
    robust_score = (
        test_final                               # Primary: Out-of-sample profit
        - lambda_turns * (turns / turns_scale)   # Penalty: Too many trades
        - gap_penalty * gen_gap                  # Penalty: Train/test gap
        - lambda_fees * fees                     # Penalty: Fee costs
        - lambda_turnover * turnover             # Penalty: Total turnover
    )
```

**Default Weights:**
| Weight | Default | Description |
|--------|---------|-------------|
| `lambda_turns` | 2.0 | Trade count penalty |
| `gap_penalty` | 0.35 | Generalization gap penalty |
| `lambda_fees` | 2.0 | Fee penalty |
| `lambda_turnover` | 1.0 | Turnover penalty |
| `turns_scale` | 800.0 | Normalizes trade count |

### 4. Step-by-Step Guide: Running MR WFO

1. **Run WFO optimization:**
   ```bash
   python tools/optimizer_cli.py \
     --data data/raw/ETHBTC_15m_2021-2025.csv \
     --config configs/base_mr.json \
     --wfo \
     --window-days 180 \
     --step-days 30 \
     --n-trials 100 \
     --out results/wfo_mr_ethbtc.csv
   ```

2. **View progress:**
   ```
   14:23:45 [OPT] 🚀 Starting Walk-Forward Optimization (Window=180d, Step=30d)
   14:24:12 [OPT] Trial 0 DONE: Score=1.2345 (Profit=1.1234) in 2.14s
   14:24:18 [OPT] Trial 1 DONE: Score=1.1876 (Profit=1.0876) in 1.98s
   ...
   14:45:00 [OPT] [WFO] Window 2021-06-30: Train=1.1234 | OOS=1.0567
   ```

3. **Review output CSV:**
   ```bash
   head results/wfo_mr_ethbtc.csv
   ```
   ```csv
   window_end,oos_start,oos_end,oos_profit,train_profit,best_params
   2021-06-30,2021-07-01,2021-07-31,1.0567,1.1234,"{\"trend_lookback\":120...}"
   2021-07-30,2021-08-01,2021-08-31,1.0892,1.1456,"{\"trend_lookback\":160...}"
   ```

### 5. Force Flags

Lock specific parameters to prevent search:

```bash
python tools/optimizer_cli.py \
  --data data/raw/ETHBTC_15m.csv \
  --config configs/base.json \
  --force-trend-kind sma \
  --force-sizing-mode volatility \
  --force-long-only true \
  --wfo
```

| Flag | Effect |
|------|--------|
| `--force-trend-kind sma` | Only test SMA, not ROC |
| `--force-sizing-mode volatility` | Only test volatility sizing |
| `--force-long-only true` | Only test long-only mode |
| `--long-only-mode both` | Test both long-only and shorting |

---

## 9.3 Trend Strategy Optimizer

### 1. Concept & "The Why"

* **What it is:** Bayesian optimization of Trend Following strategy parameters.

* **Purpose:** Find optimal MA periods, cooldown, and associated filters.

* **Location:** [`tools/optimize_trend.py`](../../tools/optimize_trend.py)

### 2. Search Space

| Parameter | Search Range | Type | Description |
|-----------|--------------|------|-------------|
| `fast_period` | `10 – 200` (step 10) | int | Fast MA period |
| `slow_period` | `40 – 400` (step 20) | int | Slow MA period |
| `ma_type` | `["ema", "sma"]` | categorical | MA type |
| `cooldown_minutes` | `[60, 120, 240, 360]` | categorical | Cooldown |
| `funding_limit_long` | `0.01 – 0.10` | float | Long funding limit |
| `funding_limit_short` | `-0.10 – -0.01` | float | Short funding limit |
| `position_sizing_mode` | `["static", "volatility"]` | categorical | Sizing mode |
| `position_sizing_target_vol` | `0.3 – 0.7` | float | Target vol |
| `position_sizing_min_step` | `0.1 – 0.3` | float | Min step |
| `long_only` | `[True, False]` | categorical | Long-only (if `--allow-shorts`) |

**Constraint:** `fast_period < slow_period` (automatically pruned if violated)

### 3. Scoring Function

```python
score = train_profit

# Penalties
if train_dd < -0.25:
    score -= 0.5  # High drawdown penalty

if train_turns < 5:
    score -= 1.0  # Inactive strategy penalty
```

### 4. Step-by-Step Guide: Running Trend WFO

1. **Run trend WFO:**
   ```bash
   python tools/optimize_trend.py \
     --data data/raw/BTCUSDT_15m_2021-2025_vision.csv \
     --funding-data data/raw/BTCUSDT_funding.csv \
     --wfo \
     --window-days 180 \
     --step-days 30 \
     --n-trials 50 \
     --allow-shorts \
     --out results/wfo_trend_btcusdt.csv
   ```

2. **Static mode (single train/test split):**
   ```bash
   python tools/optimize_trend.py \
     --data data/raw/BTCUSDT_15m_2021-2025_vision.csv \
     --train-start 2021-01-01 \
     --train-end 2023-12-31 \
     --test-start 2024-01-01 \
     --test-end 2024-12-31 \
     --n-trials 100 \
     --out results/static_trend_btcusdt.csv
   ```

---

## 9.4 Meta Strategy Optimizer

### 1. Concept & "The Why"

* **What it is:** Grid search optimization for the Meta (Ensemble) strategy's ADX threshold.

* **Purpose:** Find the optimal regime-switching threshold using pre-optimized MR and Trend configs.

* **Location:** [`tools/optimize_meta.py`](../../tools/optimize_meta.py)

### 2. Search Space

| Parameter | Search Values | Description |
|-----------|---------------|-------------|
| `adx_threshold` | `[10, 15, 20, 25, 30, 35, 40]` | Regime switch threshold |

### 3. Step-by-Step Guide

1. **First, optimize MR and Trend separately:**
   ```bash
   # Optimize Mean Reversion
   python tools/optimizer_cli.py \
     --data data/raw/BTCUSDT_15m.csv \
     --config configs/base_mr.json \
     --wfo --out results/wfo_mr.csv
   
   # Pick best MR config and save to configs/best_mr.json
   
   # Optimize Trend
   python tools/optimize_trend.py \
     --data data/raw/BTCUSDT_15m.csv \
     --wfo --out results/wfo_trend.csv
   
   # Pick best Trend config and save to configs/best_trend.json
   ```

2. **Optimize Meta threshold:**
   ```bash
   python tools/optimize_meta.py \
     --data data/raw/BTCUSDT_15m_2021-2025_vision.csv \
     --funding-data data/raw/BTCUSDT_funding.csv \
     --mr-config configs/best_mr.json \
     --trend-config configs/best_trend.json \
     --out results/opt_meta.csv
   ```

3. **Review results:**
   ```bash
   cat results/opt_meta.csv
   ```
   ```csv
   adx_threshold,final_btc,drawdown,trades,fees
   25,1.4567,0.12,256,0.0034
   20,1.3892,0.15,312,0.0041
   30,1.3234,0.11,198,0.0028
   ```

---

## 9.5 Optuna Integration

### 1. Concept & "The Why"

* **What it is:** Optuna is a hyperparameter optimization framework with advanced sampling algorithms.

* **Purpose:** More efficient than grid search—uses Bayesian optimization (TPE) to focus on promising regions.

### 2. Key Features Used

| Feature | Description |
|---------|-------------|
| **TPE Sampler** | Tree-structured Parzen Estimator for efficient search |
| **SQLite Storage** | Persistent study storage for resumption |
| **Pruning** | Early stopping of bad trials |
| **Parallel Jobs** | Multi-process optimization |
| **Startup Trials** | Random exploration before exploitation |

### 3. Configuration

```python
# Sampler configuration (in optimizer_cli.py)
sampler = optuna.samplers.TPESampler(
    n_startup_trials=50,   # Random exploration trials
    multivariate=True,     # Consider parameter correlations
    seed=42                # Reproducibility
)

study = optuna.create_study(
    study_name="ethbtc_study",
    direction="maximize",
    storage="sqlite:///data/db/optuna.db",
    load_if_exists=True,
    sampler=sampler
)
```

### 4. Study Management

**List existing studies:**
```bash
python -c "import optuna; print([s.study_name for s in optuna.get_all_study_summaries('sqlite:///data/db/optuna.db')])"
```

**Delete a study:**
```bash
python -c "import optuna; optuna.delete_study('old_study', 'sqlite:///data/db/optuna.db')"
```

**Resume optimization:**
```bash
# Just run with same --study-name and --storage; it loads existing trials
python tools/optimizer_cli.py \
  --data data/raw/ETHBTC_15m.csv \
  --study-name ethbtc_study \
  --storage sqlite:///data/db/optuna.db \
  --n-trials 100  # Adds 100 more trials
```

---

## 9.6 Result Analysis

### 1. Output CSV Structure

| Column | Description |
|--------|-------------|
| `window_end` | End of training window |
| `oos_start` | Start of test window |
| `oos_end` | End of test window |
| `oos_profit` | Out-of-sample profit |
| `train_profit` | In-sample profit |
| `best_params` | JSON of best parameters |

### 2. Analyzing WFO Results

```python
import pandas as pd
import json

# Load WFO results
df = pd.read_csv("results/wfo_mr_ethbtc.csv")

# Parse best params
df["params"] = df["best_params"].apply(json.loads)

# Extract specific parameters
df["trend_lookback"] = df["params"].apply(lambda x: x.get("trend_lookback"))
df["flip_band_entry"] = df["params"].apply(lambda x: x.get("flip_band_entry"))

# Summary statistics
print(f"Average OOS Profit: {df['oos_profit'].mean():.4f}")
print(f"Average Train Profit: {df['train_profit'].mean():.4f}")
print(f"Generalization Gap: {(df['train_profit'] - df['oos_profit']).mean():.4f}")

# Most common lookback
print(f"Most common trend_lookback: {df['trend_lookback'].mode()[0]}")
```

### 3. Creating Production Config

```python
import json

# Use median/mode of WFO-optimized parameters
production_config = {
    "fees": {
        "maker_fee": 0.0002,
        "taker_fee": 0.0004
    },
    "strategy": {
        "strategy_type": "mean_reversion",
        "trend_lookback": 160,  # Mode from WFO
        "flip_band_entry": 0.042,  # Median from WFO
        "flip_band_exit": 0.022,
        "position_sizing_mode": "volatility",
        "position_sizing_target_vol": 0.5
    },
    "execution": {
        "interval": "15m"
    },
    "risk": {
        "max_dd_frac": 0.15
    }
}

with open("configs/prod_wfo_optimized.json", "w") as f:
    json.dump(production_config, f, indent=2)
```

---

## 9.7 Real-World Use Case (The "Cookbook")

### Scenario: Full WFO Pipeline for BTC/USDT Meta Strategy

**Goal:** Optimize a production-ready Meta strategy config.

**Step 1: Optimize Mean Reversion**
```bash
python tools/optimizer_cli.py \
  --data data/raw/BTCUSDT_15m_2021-2025_vision.csv \
  --funding-data data/raw/BTCUSDT_funding.csv \
  --config configs/base_mr.json \
  --wfo \
  --window-days 180 \
  --step-days 30 \
  --n-trials 100 \
  --force-long-only true \
  --out results/wfo_mr_btc.csv
```

**Step 2: Optimize Trend**
```bash
python tools/optimize_trend.py \
  --data data/raw/BTCUSDT_15m_2021-2025_vision.csv \
  --funding-data data/raw/BTCUSDT_funding.csv \
  --wfo \
  --window-days 180 \
  --step-days 30 \
  --n-trials 50 \
  --allow-shorts \
  --out results/wfo_trend_btc.csv
```

**Step 3: Create best configs from WFO results**
```bash
# Extract best params from last window and save to JSON
python -c "
import pandas as pd
import json

mr = pd.read_csv('results/wfo_mr_btc.csv')
tr = pd.read_csv('results/wfo_trend_btc.csv')

# Use last window's best params
mr_params = json.loads(mr.iloc[-1]['best_params'])
tr_params = json.loads(tr.iloc[-1]['best_params'])

# Save
with open('configs/wfo_best_mr.json', 'w') as f:
    json.dump({'strategy': mr_params}, f, indent=2)
with open('configs/wfo_best_trend.json', 'w') as f:
    json.dump({'strategy': tr_params}, f, indent=2)
"
```

**Step 4: Optimize Meta threshold**
```bash
python tools/optimize_meta.py \
  --data data/raw/BTCUSDT_15m_2021-2025_vision.csv \
  --funding-data data/raw/BTCUSDT_funding.csv \
  --mr-config configs/wfo_best_mr.json \
  --trend-config configs/wfo_best_trend.json \
  --out results/opt_meta_btc.csv
```

**Step 5: Create final production config**
```bash
# Best ADX threshold from Step 4 (e.g., 15)
# Combine all into final config
```

**Expected Outcome:**
- Rolling 4-year validation with 30-day forward tests
- Generalization gap < 10% between train and OOS
- Production config that adapts to recent market conditions

---

## 9.8 Troubleshooting

### Common Errors

```
ValueError: Trial NNNN CRASHED
```
**Cause:** Parameter combination caused simulation error.
**Fix:** Check for `fast >= slow` in trend or NaN in data.

```
sqlite3.OperationalError: database is locked
```
**Cause:** Multiple processes writing to SQLite.
**Fix:** Use `--jobs 1` or switch to PostgreSQL storage.

```
optuna.exceptions.TrialPruned: Fast >= Slow
```
**Cause:** Trend optimizer detected invalid MA combination.
**Fix:** Normal—Optuna automatically handles pruned trials.

### Performance Tips

1. **Use parallel jobs for faster optimization:**
   ```bash
   --jobs 4
   ```

2. **Reduce trials for initial testing:**
   ```bash
   --n-trials 20
   ```

3. **Use force flags to narrow search:**
   ```bash
   --force-trend-kind sma --force-long-only true
   ```

4. **Resume interrupted optimization:**
   ```bash
   # Just re-run with same --study-name; Optuna loads existing trials
   ```

---

*Previous Chapter: [Chapter 8: Backtesting Engine](./CHAPTER_08_BACKTESTING.md)*  
*Next Chapter: [Chapter 10: Monitoring & Observability](./CHAPTER_10_MONITORING.md)*
