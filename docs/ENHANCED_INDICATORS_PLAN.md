# Enhanced Indicators Development Plan

## Overview

Five new indicator features to be added, each **toggleable via config** and **WFO-optimizable**.

| Priority | Feature | Config Toggle | Affects |
|----------|---------|---------------|---------|
| 1 | Funding Counter-Trend | `funding_counter_enabled` | Trend |
| 2 | Volume Confirmation | `volume_confirm_enabled` | Trend |
| 3 | RSI Filter | `rsi_filter_enabled` | MR |
| 4 | Bollinger Squeeze | `bollinger_squeeze_enabled` | Trend |
| 5 | Higher TF Filter | `htf_filter_enabled` | Both |

---

## Feature 1: Funding Counter-Trend (MUST DO)

**Purpose**: Open SHORT when funding indicates overleveraged longs.

### New Parameters (TrendParams)
```python
funding_counter_enabled: bool = False
extreme_funding_long_threshold: float = 0.0005  # >0.05% → short
extreme_funding_short_threshold: float = -0.0005  # <-0.05% → long
funding_counter_position_size: float = 0.5
funding_counter_cooldown_minutes: int = 480
```

### Files to Modify
- `core/trend_strategy.py` - Add logic
- `core/config_schema.py` - Add validation
- `core/strategy_factory.py` - Add param building
- `tools/optimizer/strategies/trend.py` - Add to search space

---

## Feature 2: Volume Confirmation (MUST DO)

**Purpose**: Only enter trend trades if volume confirms the move.

### New Parameters (TrendParams)
```python
volume_confirm_enabled: bool = False
volume_threshold_mult: float = 1.5  # Volume > 1.5x average
volume_lookback_bars: int = 20  # 20-bar average
```

### Logic
```python
if volume_confirm_enabled:
    avg_vol = volume.rolling(volume_lookback_bars).mean()
    vol_confirmed = volume > (avg_vol * volume_threshold_mult)
    signal = signal & vol_confirmed  # Only enter if volume confirms
```

### Files to Modify
- `core/trend_strategy.py` - Add volume filter
- `core/ethbtc_accum_bot.py` - Ensure volume passed to strategy

---

## Feature 3: RSI Filter (MAYBE)

**Purpose**: Improve Mean Reversion entries with RSI confirmation.

### New Parameters (StratParams / MR)
```python
rsi_filter_enabled: bool = False
rsi_period: int = 14
rsi_oversold: float = 30.0  # Enter long only if RSI < 30
rsi_overbought: float = 70.0  # Enter short only if RSI > 70
```

### Logic
```python
if rsi_filter_enabled:
    rsi = calc_rsi(close, rsi_period)
    long_ok = rsi < rsi_oversold
    short_ok = rsi > rsi_overbought
    signal = signal * long_ok  # For longs
```

### Files to Modify
- `core/ethbtc_accum_bot.py` - Add RSI calc + filter to MR strategy

---

## Feature 4: Bollinger Squeeze (MAYBE)

**Purpose**: Detect volatility compression before breakouts.

### New Parameters (TrendParams)
```python
bollinger_squeeze_enabled: bool = False
bollinger_period: int = 20
bollinger_std: float = 2.0
squeeze_threshold: float = 0.5  # Band width < 50% of normal
squeeze_breakout_bars: int = 5  # Signal valid for 5 bars after squeeze
```

### Logic
```python
if bollinger_squeeze_enabled:
    band_width = (upper - lower) / middle
    avg_width = band_width.rolling(20).mean()
    squeeze = band_width < (avg_width * squeeze_threshold)
    # Only enter if recent squeeze detected
```

### Files to Modify
- `core/trend_strategy.py` - Add Bollinger squeeze detection

---

## Feature 5: Higher Timeframe Filter (MAYBE)

**Purpose**: Only trade with the higher timeframe trend.

### New Parameters (TrendParams)
```python
htf_filter_enabled: bool = False
htf_multiplier: int = 4  # 15m bars → use 1H (4x)
htf_ma_period: int = 50  # Above 50 MA on HTF = bullish
```

### Logic
```python
if htf_filter_enabled:
    htf_close = resample_to_htf(close, htf_multiplier)
    htf_ma = htf_close.rolling(htf_ma_period).mean()
    htf_bullish = htf_close > htf_ma
    # Only allow longs if HTF bullish
```

### Files to Modify
- `core/trend_strategy.py` - Add HTF filter
- Requires resampling logic

---

## Shared Infrastructure

### Config Schema Updates
All new params added to `TrendOverrides` and `MeanReversionOverrides` in `config_schema.py`.

### Strategy Factory Updates
`build_tr_params()` and `build_mr_params()` updated in `strategy_factory.py`.

### Optimizer Updates
Each feature adds conditional params to search space:
```python
if trial.suggest_categorical("funding_counter_enabled", [True, False]):
    trial.suggest_float("extreme_funding_long_threshold", ...)
```

### Diagnostics Updates
Add columns to backtest diagnostics:
- `funding_counter_signal`
- `volume_confirmed`
- `rsi_value`
- `bollinger_squeeze`
- `htf_trend`

---

## Testing Strategy

### Unit Tests (per feature)
| Feature | Test File | Test Count |
|---------|-----------|------------|
| Funding Counter | `test_funding_counter.py` | 5 |
| Volume Confirm | `test_volume_confirm.py` | 4 |
| RSI Filter | `test_rsi_filter.py` | 4 |
| Bollinger Squeeze | `test_bollinger_squeeze.py` | 4 |
| HTF Filter | `test_htf_filter.py` | 4 |

### Parity Tests
Add to `test_parity_sync.py`:
- Each feature produces identical signals in backtest vs live
- Disabled features don't affect signals

### Integration Tests
- Run full backtest with all features enabled
- Run full backtest with all features disabled
- Compare results

---

## WFO Optimization

### Search Space Expansion
```python
# Feature toggles (categorical)
funding_counter_enabled: [True, False]
volume_confirm_enabled: [True, False]
rsi_filter_enabled: [True, False]
bollinger_squeeze_enabled: [True, False]
htf_filter_enabled: [True, False]

# Conditional params (only if enabled)
# ... thresholds, periods, etc.
```

### Recommended WFO Runs
1. **Baseline**: All new features disabled
2. **Funding Only**: Only funding counter enabled
3. **All Features**: Everything enabled, let WFO decide

---

## Implementation Order

| Phase | Features | Effort |
|-------|----------|--------|
| 1 | Funding Counter-Trend | 3h |
| 2 | Volume Confirmation | 2h |
| 3 | RSI Filter | 2h |
| 4 | Bollinger Squeeze | 2h |
| 5 | HTF Filter | 2h |
| 6 | Full WFO + Validation | 4h |

**Total: ~15 hours**

---

## Rollback Safety

Each feature is:
1. **Disabled by default** (`*_enabled: false`)
2. **Isolated in code** (clear if-blocks)
3. **Independently testable**
4. **WFO-optimizable** (Optuna can disable it)

If any feature hurts performance, simply set `*_enabled: false` in config.
