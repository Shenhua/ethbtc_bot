# Appendix C: Mathematical Formulas

> **Purpose:** Complete reference of all mathematical formulas used in strategy calculations and risk management.

---

## C.1 Signal Generation

### Mean Reversion Signal

**Ratio calculation:**
```
ratio = price / trend_ma
deviation = (ratio - 1.0) * 100  # As percentage
```

**Entry/Exit bands:**
```
buy_band  = 1.0 - flip_band_entry  # e.g., 0.97 for 3% entry
sell_band = 1.0 + flip_band_entry  # e.g., 1.03 for 3% entry

exit_long  = 1.0 - flip_band_exit  # e.g., 0.985 for 1.5% exit
exit_short = 1.0 + flip_band_exit  # e.g., 1.015 for 1.5% exit
```

**Signal logic:**
```
if ratio <= buy_band:
    signal = +1  # BUY
elif ratio >= sell_band:
    signal = -1  # SELL (if shorting allowed)
elif position > 0 and ratio >= (1.0 - flip_band_exit):
    signal = 0   # EXIT LONG
elif position < 0 and ratio <= (1.0 + flip_band_exit):
    signal = 0   # EXIT SHORT
```

---

### Trend Following Signal

**Moving Average Crossover:**
```
fast_ma = EMA(close, fast_period)  # or SMA
slow_ma = EMA(close, slow_period)

if fast_ma > slow_ma:
    signal = +1  # LONG
elif fast_ma < slow_ma:
    signal = -1  # SHORT (if allowed)
else:
    signal = 0   # FLAT
```

---

### Meta Strategy Regime Detection

**ADX Calculation:**
```
TR = max(high - low, abs(high - prev_close), abs(low - prev_close))
+DM = high - prev_high if (high - prev_high) > (prev_low - low) else 0
-DM = prev_low - low if (prev_low - low) > (high - prev_high) else 0

+DI = 100 × EMA(+DM, 14) / EMA(TR, 14)
-DI = 100 × EMA(-DM, 14) / EMA(TR, 14)

DX = 100 × |+DI - -DI| / (+DI + -DI)
ADX = EMA(DX, 14)
```

**Regime State:**
```
if ADX > adx_threshold:
    regime = TREND
else:
    regime = MEAN_REVERSION
```

---

## C.2 Position Sizing

### Static Mode

```
step = base_step  # Always constant
```

---

### Volatility Targeting

**Realized volatility (annualized):**
```
returns = log(close / close.shift(1))
realized_vol = returns.rolling(vol_window).std() × √(252 × bars_per_day)
```

**Step calculation:**
```
vol_ratio = target_vol / realized_vol
raw_step = base_step × vol_ratio
step = clamp(raw_step, min_step, max_step)
```

---

### Kelly Criterion

**Full Kelly:**
```
kelly = (win_rate × avg_win - (1 - win_rate) × avg_loss) / avg_win
```

**Half Kelly (recommended):**
```
kelly_half = kelly × 0.5
```

**With volatility adjustment:**
```
vol_adj = target_vol / realized_vol
step = kelly_half × vol_adj × base_step
step = clamp(step, min_step, max_step)
```

**Example:**
```
win_rate = 0.55
avg_win = 0.03  (3%)
avg_loss = 0.02 (2%)

kelly = (0.55 × 0.03 - 0.45 × 0.02) / 0.03
      = (0.0165 - 0.009) / 0.03
      = 0.25  (25%)

kelly_half = 0.125  (12.5%)
```

---

## C.3 Risk Metrics

### Drawdown

**Current Drawdown:**
```
drawdown = (HWM - current_wealth) / HWM
drawdown_pct = drawdown × 100
```

**Max Drawdown:**
```
max_dd = max(all_drawdowns_in_period)
```

---

### Daily Loss

**Fixed basis mode:**
```
daily_loss = day_start_wealth - current_wealth
limit_hit = daily_loss >= max_daily_loss_btc
```

**Dynamic mode:**
```
daily_loss_pct = (day_start_wealth - current_wealth) / day_start_wealth
limit_hit = daily_loss_pct >= max_daily_loss_frac
```

---

### Phoenix Protocol

**Reset conditions:**
```
time_passed = current_time - maxdd_hit_time
days_passed = time_passed.total_seconds() / 86400

can_reset = (
    days_passed >= drawdown_reset_days AND
    current_regime_score >= drawdown_reset_score
)
```

---

## C.4 Performance Metrics

### Sharpe Ratio

```
excess_returns = returns - risk_free_rate / 252
sharpe = mean(excess_returns) / std(returns) × √252
```

---

### Sortino Ratio

```
downside_returns = returns.where(returns < 0, 0)
downside_std = std(downside_returns)
sortino = mean(excess_returns) / downside_std × √252
```

---

### Calmar Ratio

```
calmar = CAGR / max_drawdown
```

---

### CAGR

```
years = (end_date - start_date).days / 365.25
CAGR = (final_value / initial_value)^(1/years) - 1
```

---

### Win Rate & Profit Factor

```
win_rate = winning_trades / total_trades

gross_profit = sum(positive_pnl_trades)
gross_loss = abs(sum(negative_pnl_trades))
profit_factor = gross_profit / gross_loss
```

---

### VaR and CVaR

**Value at Risk (95%):**
```
VaR_95 = percentile(daily_returns, 5)  # 5th percentile
```

**Conditional VaR (Expected Shortfall):**
```
CVaR_95 = mean(returns.where(returns <= VaR_95))
```

---

## C.5 Fee Calculations

### Maker/Taker Fees

```
fee = notional × fee_rate
notional = quantity × price

# With BNB discount
effective_rate = fee_rate × (1 - bnb_discount)
```

---

### Slippage

```
# Buy: pay more than mid
fill_price = mid_price × (1 + slippage_bps / 10000)

# Sell: receive less than mid
fill_price = mid_price × (1 - slippage_bps / 10000)

slippage_cost = abs(fill_price - mid_price) × quantity
```

---

### Total Trade Cost

```
total_cost = fee + slippage_cost
cost_bps = total_cost / notional × 10000
```

---

## C.6 Volatility Formulas

### Annualized Volatility

```
bars_per_year = bars_per_day × 252

# From returns
returns = (price - price.shift(1)) / price.shift(1)
vol_annual = std(returns) × √bars_per_year

# From log returns (more accurate)
log_returns = log(price / price.shift(1))
vol_annual = std(log_returns) × √bars_per_year
```

**15-minute bars:**
```
bars_per_day = 96
bars_per_year = 96 × 252 = 24,192
annual_factor = √24192 ≈ 155.5
```

---

*Return to: [Table of Contents](./MASTER_TABLE_OF_CONTENTS.md)*
