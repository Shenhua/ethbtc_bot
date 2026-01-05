# Chapter 3: Configuration Reference

> **Purpose:** This chapter provides exhaustive documentation of every configuration parameter in the ETH/BTC Trading Bot. Each parameter includes its type, valid range, default value, and practical usage examples.

---

## 3.1 Configuration File Structure

### 1. Concept & "The Why"

* **What it is:** JSON configuration files that define trading behavior, risk limits, and execution parameters. The system uses Pydantic for validation with strict type checking.

* **Purpose:** Externalizing configuration allows:
  1. Parameter changes without code modification
  2. Multiple bot instances with different configs
  3. Version-controlled parameter history
  4. Validation on startup (fail-fast on invalid configs)

* **Location:** 
  - Schema: [`core/config_schema.py`](../../core/config_schema.py)
  - Config files: [`configs/*.json`](../../configs/)

### 2. Configuration & Parameters

#### Top-Level Structure

```json
{
  "fees": { },       // Trading fees and slippage
  "strategy": { },   // Signal generation parameters
  "execution": { },  // Order execution settings
  "risk": { }        // Risk management limits
}
```

#### Validation Behavior

| Scenario | Behavior |
|----------|----------|
| Missing required field | `ValidationError` on startup |
| Value out of range | `ValidationError` with field name and constraint |
| Unknown field | Silently ignored (allows forward compatibility) |
| Flat legacy format | Auto-converted to nested structure |

**Hidden Logic:**
- Pydantic V2 `model_validator` automatically converts flat legacy configs to nested format
- Boolean values accept `true`, `false`, `1`, `0`
- Numeric strings are NOT auto-converted (use actual numbers)

### 3. Step-by-Step Guide: Creating a Config File

1. **Copy a template:**
   ```bash
   cp configs/prod_btc_meta_live.json configs/my_custom_config.json
   ```

2. **Edit with your parameters:**
   ```bash
   nano configs/my_custom_config.json
   ```

3. **Validate before running:**
   ```bash
   python -c "from core.config_schema import load_config; load_config('configs/my_custom_config.json')"
   ```
   If no error, config is valid.

4. **Run with your config:**
   ```bash
   python live_executor.py --params configs/my_custom_config.json --mode dry
   ```

### 4. Real-World Use Case (The "Cookbook")

**Scenario:** Create a conservative config for small account.

**Configuration:**
```json
{
  "fees": {
    "maker_fee": 0.0002,
    "taker_fee": 0.0004,
    "slippage_bps": 2.0,
    "bnb_discount": 0.0,
    "pay_fees_in_bnb": false
  },
  "strategy": {
    "strategy_type": "mean_reversion",
    "step_allocation": 0.25,
    "max_position": 0.5,
    "long_only": true
  },
  "execution": {
    "interval": "1h",
    "exchange_type": "spot",
    "taker_fallback": false
  },
  "risk": {
    "basis_btc": 0.1,
    "max_dd_frac": 0.10,
    "risk_mode": "dynamic"
  }
}
```

**Expected Outcome:** Bot trades conservatively with max 50% position, 10% max drawdown, hourly bars.

### 5. Troubleshooting & Edge Cases

* **Error Messages:**

  ```
  pydantic_core._pydantic_core.ValidationError: 1 validation error for AppConfig
  fees -> maker_fee
    Field required [type=missing, input_value={}, input_type=dict]
  ```
  **Cause:** `maker_fee` is required but missing.
  **Fix:** Add `"maker_fee": 0.0002` to the `fees` section.

  ```
  fees -> maker_fee
    Input should be less than or equal to 0.01 [type=less_than_equal]
  ```
  **Cause:** `maker_fee` value exceeds maximum (0.01 = 1%).
  **Fix:** Use a realistic fee like `0.0002` (0.02%).

---

## 3.2 Fees Configuration

### 1. Concept & "The Why"

* **What it is:** Parameters that model trading costs for accurate backtest simulation and live P&L tracking.

* **Purpose:** Unrealistic fee assumptions are the #1 cause of backtest over-optimism. These parameters ensure backtests match live performance.

* **Location:** `fees` section in config JSON; validated by `Fees` class in `config_schema.py`.

### 2. Configuration & Parameters

| Parameter | Type | Range | Default | Required | Description |
|-----------|------|-------|---------|----------|-------------|
| `maker_fee` | float | 0.0 – 0.01 | — | **Yes** | Fee for limit orders (e.g., 0.0002 = 0.02%) |
| `taker_fee` | float | 0.0 – 0.02 | — | **Yes** | Fee for market orders (e.g., 0.0004 = 0.04%) |
| `slippage_bps` | float | 0.0 – 100.0 | 0.0 | No | Expected slippage in basis points |
| `bnb_discount` | float | 0.0 – 1.0 | 0.0 | No | BNB fee discount (e.g., 0.25 = 25% off) |
| `pay_fees_in_bnb` | bool | — | true | No | Use BNB for fee payment |

**Hidden Logic:**
- `slippage_bps` is applied to BOTH buy and sell in backtests (conservative)
- BNB discount is applied as: `effective_fee = base_fee × (1 - bnb_discount)`
- If `pay_fees_in_bnb` is false, `bnb_discount` is ignored

### 3. Standard Fee Configurations

#### Binance Spot (VIP 0)
```json
"fees": {
  "maker_fee": 0.001,
  "taker_fee": 0.001,
  "slippage_bps": 1.0,
  "bnb_discount": 0.25,
  "pay_fees_in_bnb": true
}
```
Effective: 0.075% maker, 0.075% taker (with BNB discount)

#### Binance Futures (VIP 0)
```json
"fees": {
  "maker_fee": 0.0002,
  "taker_fee": 0.0005,
  "slippage_bps": 0.5,
  "bnb_discount": 0.1,
  "pay_fees_in_bnb": true
}
```
Effective: 0.018% maker, 0.045% taker

#### Conservative Backtest (Pessimistic)
```json
"fees": {
  "maker_fee": 0.001,
  "taker_fee": 0.001,
  "slippage_bps": 5.0,
  "bnb_discount": 0.0,
  "pay_fees_in_bnb": false
}
```

### 4. Real-World Use Case

**Scenario:** Trader has Binance VIP 1 status with reduced fees.

**Configuration:**
```json
"fees": {
  "maker_fee": 0.0009,
  "taker_fee": 0.0009,
  "slippage_bps": 1.0,
  "bnb_discount": 0.25,
  "pay_fees_in_bnb": true
}
```

**Expected Outcome:** Backtests reflect VIP 1 fee structure (0.09% base, ~0.0675% with BNB).

---

## 3.3 Strategy Configuration

### 1. Concept & "The Why"

* **What it is:** The largest configuration section, controlling signal generation for Mean Reversion, Trend Following, and Meta strategies.

* **Purpose:** Allows fine-tuning of trading signals without code changes. Most optimization happens here.

* **Location:** `strategy` section in config JSON; validated by `Strategy` class in `config_schema.py`.

### 2. Core Strategy Selection

| Parameter | Type | Options | Default | Description |
|-----------|------|---------|---------|-------------|
| `strategy_type` | enum | `mean_reversion`, `trend`, `meta` | `mean_reversion` | Which strategy to run |

**Behavior by Type:**
- `mean_reversion`: Uses `EthBtcStrategy` (flip bands around trend)
- `trend`: Uses `TrendStrategy` (MA crossover)
- `meta`: Uses `MetaStrategy` (switches between MR and Trend based on ADX)

---

### 3.3.1 Mean Reversion Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `trend_kind` | enum | `sma`, `roc` | `roc` | Trend calculation method |
| `trend_lookback` | int | 1 – 10000 | 200 | Bars for trend calculation |
| `flip_band_entry` | float | 0.0 – 1.0 | 0.025 | Distance from trend to enter (2.5%) |
| `flip_band_exit` | float | 0.0 – 1.0 | 0.015 | Distance from trend to exit (1.5%) |
| `vol_window` | int | 1 – 10000 | 45 | Volatility calculation window |
| `vol_adapt_k` | float | 0.0 – 1.0 | 0.0 | Volatility adaptation factor (0 = disabled) |
| `gate_window_days` | int | 0 – 3660 | 0 | Gate filter lookback (0 = disabled) |
| `gate_roc_threshold` | float | 0.0 – 1.0 | 0.0 | ROC threshold for gate |

**Hidden Logic:**
- `trend_kind: "sma"` uses Simple Moving Average
- `trend_kind: "roc"` uses Rate of Change (momentum)
- `vol_adapt_k` dynamically adjusts bands based on volatility
- Gate filter prevents trading during extreme momentum periods

**Example:**
```json
"strategy": {
  "strategy_type": "mean_reversion",
  "trend_kind": "sma",
  "trend_lookback": 120,
  "flip_band_entry": 0.042,
  "flip_band_exit": 0.022,
  "vol_window": 45,
  "vol_adapt_k": 0.0025
}
```

---

### 3.3.2 Trend Strategy Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `fast_period` | int | 1+ | 50 | Fast moving average period |
| `slow_period` | int | 1+ | 200 | Slow moving average period |
| `ma_type` | enum | `sma`, `ema` | `ema` | Moving average type |

**Hidden Logic:**
- Signal is long when `fast_ma > slow_ma`
- Signal is short when `fast_ma < slow_ma` (if `long_only: false`)
- Crossover detection uses previous bar comparison

**Example:**
```json
"strategy": {
  "strategy_type": "trend",
  "fast_period": 30,
  "slow_period": 360,
  "ma_type": "ema",
  "long_only": true
}
```

---

### 3.3.3 Meta Strategy Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `adx_threshold` | float | 0.0 – 100.0 | 25.0 | Regime switch threshold |
| `mean_reversion_overrides` | dict | — | `{}` | MR-specific param overrides |
| `trend_overrides` | dict | — | `{}` | Trend-specific param overrides |

**Hidden Logic:**
- ADX < `adx_threshold - 2` → Mean Reversion mode
- ADX > `adx_threshold + 2` → Trend mode
- Hysteresis buffer (±2) prevents rapid switching
- ADX is calculated as weighted average: `0.2×ADX_15m + 0.3×ADX_30m + 0.5×ADX_1h`

**Example:**
```json
"strategy": {
  "strategy_type": "meta",
  "adx_threshold": 15.0,
  "mean_reversion_overrides": {
    "trend_lookback": 120,
    "flip_band_entry": 0.042,
    "step_allocation": 0.5
  },
  "trend_overrides": {
    "fast_period": 30,
    "slow_period": 360,
    "step_allocation": 1.0
  }
}
```

---

### 3.3.4 Shared Strategy Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `cooldown_minutes` | int | 0 – 100000 | 0 | Minimum time between trades |
| `step_allocation` | float | 0.0 – 1.0 | 0.33 | Position change per trade |
| `max_position` | float | 0.0+ | 1.0 | Maximum position size (1.0 = 100%) |
| `long_only` | bool | — | true | Only take long positions |
| `rebalance_threshold_w` | float | 0.0+ | 0.0 | Min delta to trigger trade |
| `profit_lock_dd` | float | 0.0 – 1.0 | 0.0 | Lock profits at this DD level |

**Hidden Logic:**
- `step_allocation: 0.33` means move max 33% per bar toward target
- `rebalance_threshold_w: 0.01` skips trades where delta < 1%
- `cooldown_minutes` resets after any trade execution

---

### 3.3.5 Position Sizing Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `position_sizing_mode` | enum | `static`, `volatility`, `kelly` | `static` | Sizing algorithm |
| `position_sizing_target_vol` | float | 0.1 – 2.0 | 0.5 | Target volatility (annualized) |
| `position_sizing_min_step` | float | 0.0 – 1.0 | 0.1 | Minimum step size |
| `position_sizing_max_step` | float | 0.0 – 1.0 | 1.0 | Maximum step size |
| `kelly_win_rate` | float | 0.0 – 1.0 | 0.55 | Win rate for Kelly calc |
| `kelly_avg_win` | float | 0.0+ | 0.02 | Average win size |
| `kelly_avg_loss` | float | 0.0+ | 0.01 | Average loss size |

**Mode Behaviors:**
- `static`: Uses `step_allocation` directly
- `volatility`: `step = base × (target_vol / realized_vol)`
- `kelly`: `f* = (p×b - q) / b` with fractional scaling

**Example (Volatility Mode):**
```json
"strategy": {
  "position_sizing_mode": "volatility",
  "position_sizing_target_vol": 0.5,
  "position_sizing_min_step": 0.2,
  "position_sizing_max_step": 1.0,
  "step_allocation": 0.5
}
```

---

### 3.3.6 Enhanced Indicator Parameters

#### RSI Filter
| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `rsi_filter_enabled` | bool | — | false | Enable RSI filtering |
| `rsi_period` | int | 2 – 50 | 14 | RSI lookback period |
| `rsi_oversold` | float | 0.0 – 50.0 | 30.0 | Oversold threshold |
| `rsi_overbought` | float | 50.0 – 100.0 | 70.0 | Overbought threshold |

#### Volume Confirmation
| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `volume_confirm_enabled` | bool | — | false | Require volume spike |
| `volume_threshold_mult` | float | 1.0 – 5.0 | 1.5 | Volume must be 1.5× average |
| `volume_lookback_bars` | int | 5 – 100 | 20 | Bars for volume average |

#### Bollinger Squeeze
| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `bollinger_squeeze_enabled` | bool | — | false | Enable squeeze detection |
| `bollinger_period` | int | 5 – 100 | 20 | BB period |
| `bollinger_std` | float | 1.0 – 4.0 | 2.0 | Standard deviations |
| `squeeze_threshold` | float | 0.1 – 1.0 | 0.5 | Width threshold |
| `squeeze_lookback_bars` | int | 10 – 200 | 50 | Lookback for squeeze |
| `squeeze_signal_bars` | int | 1 – 50 | 10 | Bars after squeeze |

#### Higher Timeframe Filter
| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `htf_filter_enabled` | bool | — | false | Enable HTF filter |
| `htf_multiplier` | int | 2 – 96 | 16 | HTF = interval × multiplier |
| `htf_ma_period` | int | 10 – 200 | 50 | MA period on HTF |
| `htf_ma_type` | enum | `ema`, `sma` | `ema` | MA type |

#### Funding Counter-Trade
| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `funding_counter_enabled` | bool | — | false | Trade against extreme funding |
| `extreme_funding_long_threshold` | float | 0.0 – 0.01 | 0.0005 | Threshold for counter-short |
| `extreme_funding_short_threshold` | float | -0.01 – 0.0 | -0.0005 | Threshold for counter-long |
| `funding_counter_position_size` | float | 0.0 – 1.0 | 0.5 | Size for counter-trade |
| `funding_counter_cooldown_minutes` | int | 0 – 10000 | 480 | Cooldown (8 hours) |
| `funding_limit_long` | float | 0.0 – 1.0 | 0.05 | Skip long if funding > this |
| `funding_limit_short` | float | -1.0 – 0.0 | -0.05 | Skip short if funding < this |

---

## 3.4 Execution Configuration

### 1. Concept & "The Why"

* **What it is:** Parameters controlling how orders are placed, filled, and retried.

* **Purpose:** Balances execution quality (maker fills, low slippage) against execution certainty (taker fills).

* **Location:** `execution` section in config JSON; validated by `Execution` class.

### 2. Configuration & Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `interval` | enum | `1m`-`1d` | `15m` | Bar interval for trading |
| `poll_sec` | int | 1 – 300 | 5 | Order status poll interval |
| `ttl_sec` | int | 5 – 600 | 30 | Order timeout before cancel |
| `taker_fallback` | bool | — | false | Use market order if maker fails |
| `max_taker_btc` | float | 0.0 – 1.0 | 0.002 | Max size for taker fallback |
| `max_spread_bps_for_taker` | float | 0.0 – 100.0 | 2.0 | Max spread for taker |
| `min_trade_frac` | float | 0.0 – 1.0 | 0.0015 | Min trade as fraction of portfolio |
| `min_trade_floor_btc` | float | 0.0 – 10.0 | 0.0 | Absolute min trade size |
| `min_trade_cap_btc` | float | 0.0 – 10.0 | 0.0 | Cap for min trade calc |
| `min_trade_btc` | float | — | null | Override min trade (if set) |
| `exchange_type` | enum | `spot`, `futures` | `spot` | Exchange market type |
| `leverage` | int | 1 – 20 | 1 | Futures leverage multiplier |

**Available Intervals:**
`1m`, `3m`, `5m`, `15m`, `30m`, `1h`, `2h`, `4h`, `6h`, `8h`, `12h`, `1d`

**Hidden Logic:**
- `taker_fallback` only triggers if spread ≤ `max_spread_bps_for_taker`
- `min_trade_btc` overrides `min_trade_frac` if explicitly set
- Leverage applies only to futures; ignored for spot

### 3. Example Configurations

#### Conservative (Makers Only)
```json
"execution": {
  "interval": "1h",
  "poll_sec": 5,
  "ttl_sec": 60,
  "taker_fallback": false,
  "exchange_type": "spot",
  "min_trade_btc": 0.001
}
```

#### Aggressive (Fast Fill)
```json
"execution": {
  "interval": "15m",
  "poll_sec": 2,
  "ttl_sec": 15,
  "taker_fallback": true,
  "max_taker_btc": 0.01,
  "max_spread_bps_for_taker": 5.0,
  "exchange_type": "futures",
  "leverage": 3
}
```

---

## 3.5 Risk Configuration

### 1. Concept & "The Why"

* **What it is:** Hard limits that override strategy signals to protect capital.

* **Purpose:** Prevents catastrophic losses even if strategy signals are wrong. The "circuit breaker" of the system.

* **Location:** `risk` section in config JSON; validated by `Risk` class.

### 2. Configuration & Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `basis_btc` | float | 0.0 – 100000.0 | 0.0 | Reference capital for calculations |
| `risk_mode` | enum | `fixed_basis`, `dynamic` | `fixed_basis` | How thresholds are calculated |
| `max_dd_btc` | float | 0.0 – 100.0 | 0.0 | Max DD in absolute BTC |
| `max_dd_frac` | float | 0.0 – 1.0 | 0.0 | Max DD as fraction of HWM |
| `max_daily_loss_btc` | float | 0.0 – 100.0 | 0.0 | Daily loss limit (absolute) |
| `max_daily_loss_frac` | float | 0.0 – 1.0 | 0.0 | Daily loss limit (fraction) |
| `drawdown_reset_days` | float | 0.0 – 365.0 | 0.0 | Days before Phoenix reset (0 = disabled) |
| `drawdown_reset_score` | float | 0.0 – 100.0 | 25.0 | Regime score required to reset |

### 3. Risk Modes Explained

#### `fixed_basis` Mode
- Uses `max_dd_btc` and `max_daily_loss_btc` (absolute values)
- Drawdown calculated from `basis_btc` if set
- Best for: Fixed allocation accounts

```json
"risk": {
  "risk_mode": "fixed_basis",
  "basis_btc": 1.0,
  "max_dd_btc": 0.15,
  "max_daily_loss_btc": 0.03
}
```
Trading halts if DD > 0.15 BTC or daily loss > 0.03 BTC.

#### `dynamic` Mode
- Uses `max_dd_frac` and `max_daily_loss_frac` (percentages)
- Drawdown calculated from High Water Mark (HWM)
- Best for: Compounding accounts

```json
"risk": {
  "risk_mode": "dynamic",
  "max_dd_frac": 0.20,
  "max_daily_loss_frac": 0.03
}
```
Trading halts if DD > 20% of peak equity or daily loss > 3%.

### 4. Phoenix Protocol Configuration

```json
"risk": {
  "max_dd_frac": 0.15,
  "drawdown_reset_days": 7.0,
  "drawdown_reset_score": 30.0
}
```

**Behavior:**
1. When DD hits 15%, trading halts (`maxdd_hit = true`)
2. After 7 days, system checks regime score
3. If ADX score ≥ 30, trading auto-resumes
4. HWM resets to current equity

### 5. Troubleshooting

* **Bot stops trading unexpectedly:**
  ```bash
  # Check state file
  cat run_state/eth/state.json | jq '.maxdd_hit, .daily_limit_hit'
  ```
  If either is `true`, risk limit was hit.

* **Reset manually:**
  ```bash
  # Edit state.json
  jq '.maxdd_hit = false | .daily_limit_hit = false' run_state/eth/state.json > tmp.json
  mv tmp.json run_state/eth/state.json
  ```

---

## 3.6 Complete Production Config Example

```json
{
  "fees": {
    "maker_fee": 0.0002,
    "taker_fee": 0.0004,
    "slippage_bps": 1.0,
    "bnb_discount": 0.25,
    "pay_fees_in_bnb": true
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
      "max_position": 1.0
    }
  },
  "execution": {
    "interval": "15m",
    "poll_sec": 5,
    "ttl_sec": 30,
    "taker_fallback": true,
    "max_taker_btc": 0.002,
    "exchange_type": "futures",
    "leverage": 1
  },
  "risk": {
    "basis_btc": 1.0,
    "risk_mode": "dynamic",
    "max_dd_frac": 0.20,
    "drawdown_reset_days": 7.0,
    "drawdown_reset_score": 30.0
  }
}
```

---

*Previous Chapter: [Chapter 2: Installation & Deployment](./CHAPTER_02_INSTALLATION.md)*  
*Next Chapter: [Chapter 4: Trading Strategies](./CHAPTER_04_STRATEGIES.md)*
