# Chapter 4: Trading Strategies

> **Purpose:** This chapter provides exhaustive documentation of all three trading strategies: Mean Reversion, Trend Following, and Meta (Ensemble). Each strategy is documented with its mathematical foundations, signal generation logic, configuration parameters, and practical usage examples.

---

## 4.1 Mean Reversion Strategy

### 1. Concept & "The Why"

* **What it is:** A momentum-contrarian strategy that profits from price deviations from a moving trend. When price moves too far below the trend (oversold), it enters long; when price reverts toward the trend, it exits.

* **Purpose:** Capitalizes on the statistical tendency of prices to revert to their mean in range-bound markets. Works best when markets are choppy/sideways (low ADX < 25).

* **Location:** 
  - Strategy Class: [`core/ethbtc_accum_bot.py`](../../core/ethbtc_accum_bot.py) → `EthBtcStrategy`
  - Parameters: [`core/ethbtc_accum_bot.py`](../../core/ethbtc_accum_bot.py) → `StratParams` dataclass

### 2. Configuration & Parameters

#### Core Signal Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `trend_kind` | `sma` \| `roc` | `sma` | How trend is calculated |
| `trend_lookback` | int | 200 | Bars for trend calculation |
| `flip_band_entry` | float | 0.025 | Entry threshold (2.5% from trend) |
| `flip_band_exit` | float | 0.015 | Exit threshold (1.5% from trend) |
| `vol_window` | int | 60 | Volatility calculation window |
| `vol_adapt_k` | float | 0.0 | Volatility adaptation factor |
| `cooldown_minutes` | int | 180 | Minimum time between flips |

#### Signal Generation Formula

```
# Step 1: Calculate trend deviation ratio
if trend_kind == "sma":
    ma = close.rolling(trend_lookback).mean()
    ratio = (close / ma) - 1.0
else:  # roc
    ratio = (close / close.shift(trend_lookback)) - 1.0

# Step 2: Calculate adaptive bands
volatility = returns.rolling(vol_window).std() * sqrt(bars_per_year)
band_entry = flip_band_entry + (vol_adapt_k * volatility)
band_exit  = flip_band_exit + (vol_adapt_k * volatility)

# Step 3: State Machine
if ratio < -band_entry:   # Price far BELOW trend
    signal = +1.0         # Go LONG (expect reversion UP)
elif ratio > +band_entry: # Price far ABOVE trend
    signal = -1.0         # Go SHORT (expect reversion DOWN)
elif state == +1.0 and ratio > -band_exit:
    signal = 0.0          # Exit LONG (reverted)
elif state == -1.0 and ratio < +band_exit:
    signal = 0.0          # Exit SHORT (reverted)
```

#### Additional Filters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rsi_filter_enabled` | false | Only enter when RSI confirms |
| `rsi_period` | 14 | RSI lookback |
| `rsi_oversold` | 30.0 | Enter long only if RSI < 30 |
| `rsi_overbought` | 70.0 | Enter short only if RSI > 70 |
| `funding_limit_long` | 0.05 | Block long if funding > 5% |
| `funding_limit_short` | -0.05 | Block short if funding < -5% |
| `gate_window_days` | 60 | Gate filter lookback |
| `gate_roc_threshold` | 0.0 | Gate ROC threshold (0 = disabled) |

**Hidden Logic:**
- RSI is calculated using EMA smoothing: `RSI = 100 - (100 / (1 + RS))` where `RS = avg_gain / avg_loss`
- Gate filter blocks entries when daily ROC exceeds threshold (prevents buying into waterfall declines)
- Cooldown is checked using nanosecond timestamp comparison for performance

### 3. Step-by-Step Guide: Running Mean Reversion

1. **Create a Mean Reversion config:**
   ```bash
   cat > configs/my_mr_config.json << 'EOF'
   {
     "fees": { "maker_fee": 0.0002, "taker_fee": 0.0004 },
     "strategy": {
       "strategy_type": "mean_reversion",
       "trend_kind": "sma",
       "trend_lookback": 120,
       "flip_band_entry": 0.04,
       "flip_band_exit": 0.02,
       "vol_window": 45,
       "step_allocation": 0.5,
       "max_position": 0.8,
       "long_only": true
     },
     "execution": { "interval": "15m", "exchange_type": "spot" },
     "risk": { "max_dd_frac": 0.15 }
   }
   EOF
   ```

2. **Backtest the strategy:**
   ```bash
   python core/ethbtc_accum_bot.py backtest \
     --data data/raw/ETHBTC_15m_2021-2025.csv \
     --params configs/my_mr_config.json
   ```

3. **Review signal behavior in logs:**
   ```bash
   export LOGLEVEL=DEBUG
   python core/ethbtc_accum_bot.py backtest \
     --data data/raw/ETHBTC_15m_2021-2025.csv \
     --params configs/my_mr_config.json 2>&1 | grep "STRATEGY"
   ```
   Expected log output:
   ```
   [STRATEGY] Current ratio: -0.0312
   [STRATEGY] Current volatility: 0.4521
   [STRATEGY] RSI: 28.45
   ```

### 4. Real-World Use Case (The "Cookbook")

**Scenario:** Trader wants to run Mean Reversion on ETH/BTC in a sideways market with RSI confirmation.

**Configuration:**
```json
{
  "fees": { "maker_fee": 0.0002, "taker_fee": 0.0004 },
  "strategy": {
    "strategy_type": "mean_reversion",
    "trend_kind": "sma",
    "trend_lookback": 200,
    "flip_band_entry": 0.03,
    "flip_band_exit": 0.015,
    "rsi_filter_enabled": true,
    "rsi_period": 14,
    "rsi_oversold": 30.0,
    "long_only": true,
    "step_allocation": 0.5,
    "cooldown_minutes": 120
  },
  "execution": { "interval": "15m" },
  "risk": { "max_dd_frac": 0.15 }
}
```

**Expected Outcome:**
- Bot enters long only when price is 3%+ below SMA AND RSI < 30
- Exits when price reverts to within 1.5% of SMA
- No shorts (long_only mode)
- 120-minute minimum between trades

### 5. Troubleshooting & Edge Cases

* **What can go wrong:**
  - **No trades triggered:** Bands too wide (lower `flip_band_entry`) or RSI filter too strict
  - **Too many trades:** Bands too narrow, causing whipsaw
  - **Strategy underperforms in trending markets:** This is expected—use Meta strategy instead

* **Error Messages:**
  ```
  ValueError: close_time column not found
  ```
  **Cause:** CSV data format issue.
  **Fix:** Ensure CSV has `close_time` column with valid timestamps.

* **Edge Case:** If `vol_adapt_k > 0`, bands widen in volatile markets. This is intentional—it prevents entries during extreme volatility.

---

## 4.2 Trend Following Strategy

### 1. Concept & "The Why"

* **What it is:** A momentum strategy that follows the direction of the dominant trend using Moving Average crossovers. Goes long when fast MA crosses above slow MA (Golden Cross), and short (or flat) when fast crosses below slow (Death Cross).

* **Purpose:** Captures large directional moves in trending markets. Works best when markets are trending (ADX > 25).

* **Location:** 
  - Strategy Class: [`core/trend_strategy.py`](../../core/trend_strategy.py) → `TrendStrategy`
  - Parameters: [`core/trend_strategy.py`](../../core/trend_strategy.py) → `TrendParams` dataclass

### 2. Configuration & Parameters

#### Core Trend Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fast_period` | int | 50 | Fast MA period |
| `slow_period` | int | 200 | Slow MA period |
| `ma_type` | `sma` \| `ema` | `ema` | Moving average type |
| `cooldown_minutes` | int | 60 | Minimum time between flips |
| `step_allocation` | float | 1.0 | Position size per trade |
| `max_position` | float | 1.0 | Maximum position |
| `long_only` | bool | true | Only take long positions |

#### Signal Generation Formula

```python
# Step 1: Calculate Moving Averages
if ma_type == "ema":
    fast = close.ewm(span=fast_period).mean()
    slow = close.ewm(span=slow_period).mean()
else:
    fast = close.rolling(fast_period).mean()
    slow = close.rolling(slow_period).mean()

# Step 2: Generate Crossover Signal
if fast > slow:
    signal = +1.0  # Golden Cross → LONG
else:
    signal = -1.0  # Death Cross → SHORT (or 0 if long_only)

# Step 3: Apply Cooldown
if (current_time - last_flip_time) < cooldown_minutes:
    signal = previous_signal  # Hold position, no flip
```

#### Enhanced Filters (5 Optional Features)

##### 1. Higher Timeframe Filter
| Parameter | Default | Description |
|-----------|---------|-------------|
| `htf_filter_enabled` | false | Enable HTF trend filter |
| `htf_multiplier` | 16 | HTF = base × multiplier (16 = 4H for 15m) |
| `htf_ma_period` | 50 | MA period on HTF |
| `htf_ma_type` | `ema` | MA type on HTF |

**Logic:** Only allows entries that align with the higher timeframe trend. Long blocked if HTF is bearish.

##### 2. Volume Confirmation Filter
| Parameter | Default | Description |
|-----------|---------|-------------|
| `volume_confirm_enabled` | false | Require volume spike on entry |
| `volume_threshold_mult` | 1.5 | Volume must be 1.5× average |
| `volume_lookback_bars` | 20 | Bars for volume average |

**Logic:** Blocks new entries unless current volume exceeds threshold × rolling average.

##### 3. Bollinger Squeeze Filter
| Parameter | Default | Description |
|-----------|---------|-------------|
| `bollinger_squeeze_enabled` | false | Enable squeeze detection |
| `bollinger_period` | 20 | Bollinger Band period |
| `bollinger_std` | 2.0 | Standard deviations |
| `squeeze_threshold` | 0.5 | Width < 50% average = squeeze |
| `squeeze_lookback_bars` | 50 | Lookback for average width |
| `squeeze_signal_bars` | 10 | Signal valid N bars after squeeze |

**Logic:** Only allows entries after detecting volatility compression (squeeze), catching breakout moves.

##### 4. Funding Counter-Trade
| Parameter | Default | Description |
|-----------|---------|-------------|
| `funding_counter_enabled` | false | Trade against extreme funding |
| `extreme_funding_long_threshold` | 0.0005 | Go SHORT if funding > 0.05% |
| `extreme_funding_short_threshold` | -0.0005 | Go LONG if funding < -0.05% |
| `funding_counter_position_size` | 0.5 | Position size for counter |
| `funding_counter_cooldown_minutes` | 480 | 8-hour cooldown |

**Logic:** Opens opposite position when funding indicates extreme crowding.

##### 5. Funding Rate Filter
| Parameter | Default | Description |
|-----------|---------|-------------|
| `funding_limit_long` | 0.05 | Block long if funding > 5% |
| `funding_limit_short` | -0.05 | Block short if funding < -5% |

**Hidden Logic:**
- Funding filter uses "Entry Block" logic (not exit): if you're already in a position, you stay
- Volume confirmation only blocks NEW entries, not position holds
- Bollinger squeeze uses `rolling().max()` to propagate signal validity for N bars

### 3. Step-by-Step Guide: Running Trend Strategy

1. **Create a Trend config:**
   ```bash
   cat > configs/my_trend_config.json << 'EOF'
   {
     "fees": { "maker_fee": 0.0002, "taker_fee": 0.0004 },
     "strategy": {
       "strategy_type": "trend",
       "fast_period": 30,
       "slow_period": 200,
       "ma_type": "ema",
       "cooldown_minutes": 60,
       "step_allocation": 1.0,
       "max_position": 1.0,
       "long_only": true,
       "htf_filter_enabled": true,
       "htf_multiplier": 16,
       "htf_ma_period": 50
     },
     "execution": { "interval": "15m" },
     "risk": { "max_dd_frac": 0.20 }
   }
   EOF
   ```

2. **Backtest with trend strategy:**
   ```bash
   python core/ethbtc_accum_bot.py backtest \
     --data data/raw/BTCUSDT_15m_2021-2025_vision.csv \
     --params configs/my_trend_config.json
   ```

3. **Enable all enhanced filters for maximum selectivity:**
   ```json
   {
     "strategy": {
       "strategy_type": "trend",
       "htf_filter_enabled": true,
       "volume_confirm_enabled": true,
       "bollinger_squeeze_enabled": true
     }
   }
   ```

### 4. Real-World Use Case (The "Cookbook")

**Scenario:** Trader wants aggressive trend following on BTC/USDT futures with squeeze detection.

**Configuration:**
```json
{
  "fees": { "maker_fee": 0.0002, "taker_fee": 0.0005 },
  "strategy": {
    "strategy_type": "trend",
    "fast_period": 20,
    "slow_period": 100,
    "ma_type": "ema",
    "bollinger_squeeze_enabled": true,
    "bollinger_period": 20,
    "squeeze_threshold": 0.5,
    "squeeze_signal_bars": 15,
    "step_allocation": 1.0,
    "long_only": false
  },
  "execution": {
    "interval": "15m",
    "exchange_type": "futures",
    "leverage": 2
  },
  "risk": { "max_dd_frac": 0.20 }
}
```

**Expected Outcome:**
- Bot waits for Bollinger Band squeeze (consolidation)
- Enters direction of MA crossover after squeeze ends
- Takes both long AND short positions (long_only: false)
- Uses 2× leverage on futures

### 5. Troubleshooting & Edge Cases

* **What can go wrong:**
  - **No trades:** All 5 filters enabled simultaneously is very restrictive
  - **Whipsaw in ranging markets:** Trend strategy underperforms in sideways conditions
  - **HTF filter too restrictive:** Lower `htf_multiplier` or disable

* **Error Messages:**
  ```
  KeyError: 'volume'
  ```
  **Cause:** Volume confirmation enabled but CSV has no volume column.
  **Fix:** Disable `volume_confirm_enabled` or use data with volume.

---

## 4.3 Meta Strategy (Ensemble)

### 1. Concept & "The Why"

* **What it is:** An ensemble strategy that dynamically switches between Mean Reversion and Trend Following based on market regime. Uses ADX (Average Directional Index) to detect whether the market is trending or ranging.

* **Purpose:** Provides "all-weather" performance by automatically selecting the appropriate sub-strategy for current market conditions.

* **Location:** 
  - Strategy Class: [`core/meta_strategy.py`](../../core/meta_strategy.py) → `MetaStrategy`
  - Regime Detection: [`core/regime.py`](../../core/regime.py) → `get_regime_score()`

### 2. Configuration & Parameters

#### Meta Strategy Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `strategy_type` | enum | — | Must be `"meta"` |
| `adx_threshold` | float | 25.0 | Regime switch threshold |
| `mean_reversion_overrides` | dict | `{}` | Override MR params |
| `trend_overrides` | dict | `{}` | Override Trend params |

#### Regime Score Calculation

```python
# Multi-Timeframe ADX Consensus (from core/regime.py)
adx_15m = calculate_adx(high, low, close, period=14)  # Base

df_30m = resample('30min')
adx_30m = calculate_adx(df_30m).shift(1)  # Avoid look-ahead

df_1h = resample('1h')
adx_1h = calculate_adx(df_1h).shift(1)  # Avoid look-ahead

# Weighted Consensus (higher TFs weighted more)
regime_score = (0.2 × adx_15m) + (0.3 × adx_30m) + (0.5 × adx_1h)
```

#### Hysteresis Logic (Anti-Churn)

```python
# Prevents rapid switching with ±2 buffer zone
buffer = 2.0
upper_bound = adx_threshold + buffer  # e.g., 27.0
lower_bound = adx_threshold - buffer  # e.g., 23.0

if regime_score > upper_bound:
    regime = "TREND"      # Switch to Trend strategy
elif regime_score < lower_bound:
    regime = "MR"         # Switch to Mean Reversion
else:
    regime = previous_regime  # HOLD current (hysteresis)
```

#### Strategy Output

The Meta strategy returns a DataFrame with additional diagnostics:

| Column | Description |
|--------|-------------|
| `target_w` | Final blended signal |
| `regime_score` | Current ADX-based score |
| `regime_state` | -1 = MR, +1 = Trend |
| `sig_mr` | Mean Reversion signal |
| `sig_trend` | Trend signal |

**Hidden Logic:**
- ADX is calculated using Wilder's smoothing (EMA with alpha=1/14)
- Multi-TF resampling uses `label='left', closed='left'` to avoid look-ahead bias
- Regime score alignment uses `pd.merge_asof()` for O(N log N) performance
- Default start state is MR (regime_state = -1)

### 3. Step-by-Step Guide: Running Meta Strategy

1. **Create a Meta config with overrides:**
   ```bash
   cat > configs/my_meta_config.json << 'EOF'
   {
     "fees": { "maker_fee": 0.0002, "taker_fee": 0.0004 },
     "strategy": {
       "strategy_type": "meta",
       "adx_threshold": 15.0,
       "mean_reversion_overrides": {
         "trend_lookback": 120,
         "flip_band_entry": 0.04,
         "flip_band_exit": 0.02,
         "step_allocation": 0.5
       },
       "trend_overrides": {
         "fast_period": 30,
         "slow_period": 200,
         "step_allocation": 1.0
       }
     },
     "execution": { "interval": "15m" },
     "risk": { "max_dd_frac": 0.20, "drawdown_reset_days": 7.0 }
   }
   EOF
   ```

2. **Backtest the Meta strategy:**
   ```bash
   python core/ethbtc_accum_bot.py backtest \
     --data data/raw/BTCUSDT_15m_2021-2025_vision.csv \
     --params configs/my_meta_config.json
   ```

3. **View regime switching in logs:**
   ```bash
   export LOGLEVEL=DEBUG
   python core/ethbtc_accum_bot.py backtest \
     --data data/raw/BTCUSDT_15m_2021-2025_vision.csv \
     --params configs/my_meta_config.json 2>&1 | grep "META"
   ```
   Expected output:
   ```
   [META] Generating Mean Reversion signal
   [META] MR signal: 0.5000
   [META] Generating Trend signal
   [META] Trend signal: 1.0000
   [META] Final signal: 0.5000 (regime=MR, score=18.45)
   ```

### 4. Real-World Use Case (The "Cookbook")

**Scenario:** Production BTC/USDT futures with Meta strategy and Phoenix Protocol.

**Configuration:**
```json
{
  "fees": {
    "maker_fee": 0.0002,
    "taker_fee": 0.0004,
    "slippage_bps": 1.0
  },
  "strategy": {
    "strategy_type": "meta",
    "adx_threshold": 15.0,
    "mean_reversion_overrides": {
      "trend_kind": "sma",
      "trend_lookback": 120,
      "flip_band_entry": 0.042,
      "flip_band_exit": 0.022,
      "cooldown_minutes": 180,
      "step_allocation": 0.5,
      "max_position": 0.8,
      "position_sizing_mode": "volatility",
      "position_sizing_target_vol": 0.5
    },
    "trend_overrides": {
      "fast_period": 30,
      "slow_period": 360,
      "ma_type": "ema",
      "step_allocation": 1.0,
      "max_position": 1.0,
      "long_only": true
    }
  },
  "execution": {
    "interval": "15m",
    "exchange_type": "futures",
    "leverage": 1,
    "taker_fallback": true
  },
  "risk": {
    "risk_mode": "dynamic",
    "max_dd_frac": 0.20,
    "drawdown_reset_days": 7.0,
    "drawdown_reset_score": 30.0
  }
}
```

**Expected Outcome:**
- Bot automatically switches between MR (choppy) and Trend (directional) modes
- MR uses 50% step, Trend uses 100% step
- Phoenix Protocol resets after 7 days if regime score > 30
- Uses volatility-based position sizing for MR mode

### 5. Troubleshooting & Edge Cases

* **What can go wrong:**
  - **Always in one mode:** `adx_threshold` may be too high or too low for current market
  - **Frequent switching:** Lower the buffer or increase threshold separation
  - **Conflicting overrides:** Ensure override keys exist in target strategy

* **Error Messages:**
  ```
  ValueError: Need OHLC
  ```
  **Cause:** Meta strategy received a Series instead of DataFrame.
  **Fix:** Ensure data loading includes OHLC columns, not just close.

* **Edge Case:** During regime transitions, there's a hysteresis buffer of ±2 ADX points. If score hovers around threshold (e.g., 23-27 for threshold=25), the system maintains previous regime to prevent churn.

---

## 4.4 Strategy Comparison

| Aspect | Mean Reversion | Trend | Meta |
|--------|----------------|-------|------|
| **Best Market** | Sideways/Range | Trending | All |
| **Win Rate** | Higher (55-65%) | Lower (40-50%) | Mixed |
| **Avg Win Size** | Smaller | Larger | Mixed |
| **Typical Step** | 0.33-0.5 | 1.0 | Adaptive |
| **ADX Regime** | ADX < 25 | ADX > 25 | Auto-switches |
| **Hold Duration** | Shorter | Longer | Mixed |
| **Complexity** | Simple | Moderate | Highest |

---

## 4.5 Visual: Strategy Decision Flow

```
                              ┌─────────────────────────────┐
                              │       Market Data In        │
                              │    (OHLCV + Funding)        │
                              └─────────────────────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────────┐
                    │         strategy_type == "meta"?         │
                    └──────────────────────────────────────────┘
                         │                              │
                        YES                            NO
                         │                              │
                         ▼                              ▼
          ┌─────────────────────────┐    ┌─────────────────────────┐
          │   Calculate Regime      │    │  strategy_type check    │
          │   Score (ADX Blend)     │    │  "mean_reversion" or    │
          │   0.2×15m + 0.3×30m     │    │  "trend"                │
          │   + 0.5×1H              │    └─────────────────────────┘
          └─────────────────────────┘                 │
                         │                            │
                         ▼                            │
          ┌─────────────────────────┐                 │
          │  score > threshold+2?   │─── YES ───▶ TREND PATH
          │  score < threshold-2?   │─── YES ───▶ MR PATH
          │       (hysteresis)      │─── HOLD ──▶ Previous
          └─────────────────────────┘
                         │
                         ▼
          ┌─────────────────────────┐
          │   Select Sub-Strategy   │
          │   Apply Overrides       │
          │   Generate Signal       │
          └─────────────────────────┘
                         │
                         ▼
          ┌─────────────────────────┐
          │     Output: target_w    │
          │   + regime diagnostics  │
          └─────────────────────────┘
```

---

*Previous Chapter: [Chapter 3: Configuration Reference](./CHAPTER_03_CONFIGURATION.md)*  
*Next Chapter: [Chapter 5: Position Sizing](./CHAPTER_05_POSITION_SIZING.md)*
