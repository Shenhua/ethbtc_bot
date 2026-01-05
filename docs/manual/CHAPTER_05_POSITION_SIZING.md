# Chapter 5: Position Sizing

> **Purpose:** This chapter provides exhaustive documentation of the dynamic position sizing system, covering all three modes (Static, Volatility Targeting, Kelly Criterion) with mathematical foundations, implementation details, and practical configuration examples.

---

## 5.1 Position Sizer Architecture

### 1. Concept & "The Why"

* **What it is:** A modular position sizing system that determines how much capital to allocate to each trade. Instead of always trading a fixed percentage, the system adapts based on market conditions and historical performance.

* **Purpose:** Solves three critical problems:
  1. **Over-trading in volatile markets:** Without volatility scaling, a 1% move in a volatile market causes excessive position churn
  2. **Under-trading in calm markets:** Fixed sizing leaves money on the table during low-volatility regimes
  3. **Suboptimal risk allocation:** Without Kelly Criterion, traders either bet too much (ruin risk) or too little (opportunity cost)

* **Location:** 
  - Core Module: [`core/position_sizer.py`](../../core/position_sizer.py)
  - Classes: `PositionSizerConfig`, `PositionSizer`, `RollingTradeStats`

### 2. Configuration & Parameters

#### Quick Reference Table

| Mode | Description | Best For |
|------|-------------|----------|
| `static` | Fixed step allocation | Beginners, predictable sizing |
| `volatility` | Inverse volatility scaling | Adapting to market conditions |
| `kelly` | Kelly Criterion with rolling stats | Maximizing long-term growth |

#### Shared Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `position_sizing_mode` | enum | `static`, `volatility`, `kelly` | `static` | Sizing algorithm |
| `step_allocation` / `base_step` | float | 0.0–1.0 | 0.5 | Base step size |
| `position_sizing_min_step` | float | 0.0–1.0 | 0.1 | Floor for step size |
| `position_sizing_max_step` | float | 0.0–1.0 | 1.0 | Ceiling for step size |

### 3. How It Integrates

```
Strategy Signal           Position Sizer              Final Order
    ↓                          ↓                          ↓
target_w = 0.8      →    calculate_step(vol)    →    actual_delta
                              ↓
                      step = 0.35 (example)
                              ↓
                      new_w = cur_w + step × (target_w - cur_w)
                      new_w = 0.2 + 0.35 × (0.8 - 0.2) = 0.41
```

**Hidden Logic:**
- Position sizer is called EVERY bar, not just on signal changes
- Step is applied as exponential approach: `new_w = cur_w + step × (target - cur_w)`
- This creates smooth position transitions, not instant jumps

---

## 5.2 Static Mode

### 1. Concept & "The Why"

* **What it is:** The simplest position sizing mode—uses a fixed step allocation regardless of market conditions.

* **Purpose:** Provides predictable, consistent position sizing. Best for:
  - New users learning the system
  - Strategies where volatility scaling isn't beneficial
  - Backtesting to establish baselines

* **Location:** [`core/position_sizer.py`](../../core/position_sizer.py) → `PositionSizer.calculate_step()` return `self.config.base_step`

### 2. Configuration & Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `position_sizing_mode` | string | — | Must be `"static"` |
| `step_allocation` | float | 0.5 | Position step per bar (50% toward target) |

**Defaults:** If `step_allocation` is not set, defaults to `0.5` (move 50% toward target each bar).

**Hidden Logic:**
- Static mode ignores `realized_vol` parameter entirely
- Always returns `config.base_step` unchanged
- No min/max clamping applied (assumed already within bounds)

### 3. Step-by-Step Guide

1. **Configure static sizing:**
   ```json
   {
     "strategy": {
       "position_sizing_mode": "static",
       "step_allocation": 0.33,
       "max_position": 1.0
     }
   }
   ```

2. **Understand the behavior:**
   - Signal says: target_w = 1.0 (go 100% long)
   - Current position: cur_w = 0.0 (flat)
   - Step: 0.33
   - Bar 1: new_w = 0.0 + 0.33 × (1.0 - 0.0) = 0.33 (33% long)
   - Bar 2: new_w = 0.33 + 0.33 × (1.0 - 0.33) = 0.55 (55% long)
   - Bar 3: new_w = 0.55 + 0.33 × (1.0 - 0.55) = 0.70 (70% long)
   - ...gradually approaches 100%

### 4. Real-World Use Case (The "Cookbook")

**Scenario:** Conservative trader wants predictable 3-step entry into positions.

**Configuration:**
```json
{
  "strategy": {
    "strategy_type": "mean_reversion",
    "position_sizing_mode": "static",
    "step_allocation": 0.33,
    "max_position": 1.0
  }
}
```

**Expected Outcome:**
- Takes ~3 bars to reach target (0.33 → 0.55 → 0.70 → 0.80 → ...)
- Position size is completely predictable
- No surprises from volatility spikes

### 5. Troubleshooting & Edge Cases

* **What can go wrong:**
  - **Position never reaches target:** Normal—exponential approach never fully converges. The `rebalance_threshold_w` skips tiny trades.
  - **Too slow entry:** Increase `step_allocation` to 0.5 or higher.

* **Edge Case:** With `step_allocation: 1.0`, position jumps directly to target each bar (instant, not gradual).

---

## 5.3 Volatility Targeting Mode

### 1. Concept & "The Why"

* **What it is:** Position sizing that inversely scales with realized volatility. When markets are calm, take larger positions; when volatile, take smaller positions.

* **Purpose:** Volatility scaling is the "single most important risk control" (per Theory.md). It ensures:
  - Consistent risk exposure across different volatility regimes
  - Larger positions during calm markets (more opportunity per unit risk)
  - Smaller positions during turbulent markets (capital preservation)

* **Location:** [`core/position_sizer.py`](../../core/position_sizer.py) → `PositionSizer._volatility_targeting()`

### 2. Configuration & Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `position_sizing_mode` | string | — | — | Must be `"volatility"` |
| `step_allocation` | float | 0.0–1.0 | 0.5 | Base step (before scaling) |
| `position_sizing_target_vol` | float | 0.1–2.0 | 0.5 | Target annual volatility |
| `position_sizing_min_step` | float | 0.0–1.0 | 0.1 | Minimum step (floor) |
| `position_sizing_max_step` | float | 0.0–1.0 | 1.0 | Maximum step (ceiling) |

### 3. Formula

```python
# Volatility Targeting Formula
step = base_step × (target_vol / realized_vol)
step_clamped = clamp(step, min_step, max_step)
```

**Example Calculations:**

| Realized Vol | Target Vol | Base Step | Raw Step | Clamped Step |
|--------------|------------|-----------|----------|--------------|
| 0.25 (low) | 0.5 | 0.5 | 1.0 | 1.0 (max) |
| 0.50 (normal) | 0.5 | 0.5 | 0.5 | 0.5 |
| 1.00 (high) | 0.5 | 0.5 | 0.25 | 0.25 |
| 2.00 (extreme) | 0.5 | 0.5 | 0.125 | 0.1 (min) |

**Hidden Logic:**
- Realized volatility is calculated as: `returns.rolling(vol_window).std() × sqrt(365 × 24 × 60 / bar_interval_minutes)`
- If `realized_vol <= 0` or NaN, falls back to `base_step`
- Clamping prevents extreme positions in either direction

### 4. Step-by-Step Guide

1. **Configure volatility targeting:**
   ```json
   {
     "strategy": {
       "position_sizing_mode": "volatility",
       "step_allocation": 0.5,
       "position_sizing_target_vol": 0.5,
       "position_sizing_min_step": 0.2,
       "position_sizing_max_step": 1.0
     }
   }
   ```

2. **Run backtest to see behavior:**
   ```bash
   export LOGLEVEL=DEBUG
   python core/ethbtc_accum_bot.py backtest \
     --data data/raw/BTCUSDT_15m_2021-2025_vision.csv \
     --params configs/my_vol_config.json 2>&1 | grep "PositionSizer"
   ```
   Expected output:
   ```
   [PositionSizer] Volatility targeting: rv=0.4521, target=0.5000, raw_step=0.5530, clamped=0.5530
   ```

### 5. Real-World Use Case (The "Cookbook")

**Scenario:** Trader wants aggressive sizing in calm markets, conservative in volatile.

**Configuration:**
```json
{
  "strategy": {
    "strategy_type": "meta",
    "position_sizing_mode": "volatility",
    "step_allocation": 0.5,
    "position_sizing_target_vol": 0.5,
    "position_sizing_min_step": 0.15,
    "position_sizing_max_step": 1.0,
    "vol_window": 60
  }
}
```

**Expected Outcome:**
- Bull market (low vol ~30%): Steps up to 0.83 (50% × 0.5/0.3)
- Normal market (vol ~50%): Steps at baseline 0.5
- Crash (high vol ~150%): Steps down to 0.17 (50% × 0.5/1.5)

### 6. Troubleshooting & Edge Cases

* **What can go wrong:**
  - **Always at min_step:** Volatility consistently above `target_vol × base_step / min_step`. Lower `target_vol` or `min_step`.
  - **Always at max_step:** Volatility consistently low. Consider if this is desired or adjust parameters.

* **Error Messages:**
  ```
  [PositionSizer] Invalid realized_vol=nan, using base_step
  ```
  **Cause:** Volatility calculation failed (insufficient data).
  **Fix:** Ensure data covers at least `vol_window` bars.

---

## 5.4 Kelly Criterion Mode

### 1. Concept & "The Why"

* **What it is:** Position sizing based on the Kelly Criterion—a formula that maximizes long-term geometric growth rate. Can use static parameters or dynamically calculate from rolling trade history.

* **Purpose:** 
  - Provides mathematically optimal bet sizing
  - Automatically adapts as win rate and payoff ratio change
  - Uses "Half-Kelly" (50% of optimal) to reduce variance

* **Location:** 
  - [`core/position_sizer.py`](../../core/position_sizer.py) → `PositionSizer._kelly_criterion()`
  - [`core/position_sizer.py`](../../core/position_sizer.py) → `RollingTradeStats` class

### 2. The Kelly Formula

```
f* = (p × b - q) / b

where:
  f* = optimal fraction of capital to risk
  p  = probability of winning (win rate)
  q  = probability of losing (1 - p)
  b  = ratio of average win to average loss (odds)
```

**Example Calculation:**
```
Given:
  win_rate (p) = 0.55 (55% winning trades)
  avg_win = 0.025 (2.5% per winning trade)
  avg_loss = 0.015 (1.5% per losing trade)

Step 1: Calculate odds ratio
  b = avg_win / avg_loss = 0.025 / 0.015 = 1.667

Step 2: Apply Kelly formula
  f* = (0.55 × 1.667 - 0.45) / 1.667
  f* = (0.917 - 0.45) / 1.667
  f* = 0.467 / 1.667
  f* = 0.28 (28% of capital)

Step 3: Apply Half-Kelly (kelly_fraction = 0.5)
  f_adjusted = 0.28 × 0.5 = 0.14 (14% of capital per trade)
```

### 3. Configuration & Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `position_sizing_mode` | string | — | — | Must be `"kelly"` |
| `kelly_win_rate` | float | 0.0–1.0 | 0.55 | Static win rate (fallback) |
| `kelly_avg_win` | float | 0.0+ | 0.02 | Static avg win (fallback) |
| `kelly_avg_loss` | float | 0.0+ | 0.01 | Static avg loss (fallback) |
| `kelly_fraction` | float | 0.0–1.0 | 0.5 | Fraction of full Kelly (0.5 = Half-Kelly) |
| `kelly_lookback` | int | 20–500 | 100 | Rolling trade history size |
| `kelly_min_trades` | int | 10–100 | 20 | Min trades for dynamic Kelly |

### 4. Dynamic vs Static Kelly

The system supports two modes of Kelly calculation:

#### Static Kelly (Fallback)
Uses configured `kelly_win_rate`, `kelly_avg_win`, `kelly_avg_loss` parameters.

```json
{
  "strategy": {
    "position_sizing_mode": "kelly",
    "kelly_win_rate": 0.55,
    "kelly_avg_win": 0.025,
    "kelly_avg_loss": 0.015,
    "kelly_fraction": 0.5
  }
}
```

#### Dynamic Kelly (After Sufficient Trades)
After recording `kelly_min_trades` (default 20), the system calculates win rate and payoffs from actual trade history.

**How Dynamic Kelly Works:**

```python
# RollingTradeStats tracks last N trades
class RollingTradeStats:
    def __init__(self, lookback: int = 100, min_trades: int = 20):
        self.trades = deque(maxlen=lookback)
    
    def add_trade(self, pnl: float):
        self.trades.append(pnl)
    
    def get_stats(self):
        if len(self.trades) < self.min_trades:
            return None  # Use static fallback
        
        trades = np.array(self.trades)
        wins = trades[trades > 0]
        losses = trades[trades < 0]
        
        win_rate = len(wins) / len(trades)
        avg_win = np.mean(wins) if len(wins) > 0 else 0.01
        avg_loss = abs(np.mean(losses)) if len(losses) > 0 else 0.01
        
        return (win_rate, avg_win, avg_loss)
```

**Hidden Logic:**
- Dynamic Kelly only activates after `kelly_min_trades` trades
- Trade P&L is recorded as fraction of wealth at entry
- Rolling window uses `deque(maxlen=kelly_lookback)` for O(1) additions
- Stats calculated fresh on each `calculate_step()` call
- Combined with volatility adjustment: `final = kelly × vol_adjustment`

### 5. Step-by-Step Guide

1. **Configure Kelly mode:**
   ```json
   {
     "strategy": {
       "position_sizing_mode": "kelly",
       "kelly_win_rate": 0.55,
       "kelly_avg_win": 0.02,
       "kelly_avg_loss": 0.015,
       "kelly_fraction": 0.5,
       "kelly_lookback": 100,
       "kelly_min_trades": 20
     }
   }
   ```

2. **View Kelly calculations in logs:**
   ```bash
   export LOGLEVEL=DEBUG
   python live_executor.py --params configs/my_kelly_config.json --mode dry 2>&1 | grep "Kelly"
   ```
   Early output (static fallback):
   ```
   [PositionSizer] Using STATIC Kelly: win_rate=0.55, avg_win=0.0200, avg_loss=0.0150 (need 15 more trades for dynamic)
   ```
   Later output (dynamic):
   ```
   [PositionSizer] Using DYNAMIC Kelly: win_rate=0.58, avg_win=0.0234, avg_loss=0.0142 (n=45 trades)
   ```

### 6. Real-World Use Case (The "Cookbook")

**Scenario:** Trend strategy with Kelly sizing for aggressive growth.

**Configuration:**
```json
{
  "strategy": {
    "strategy_type": "trend",
    "trend_overrides": {
      "position_sizing_mode": "kelly",
      "kelly_win_rate": 0.45,
      "kelly_avg_win": 0.035,
      "kelly_avg_loss": 0.015,
      "kelly_fraction": 0.5,
      "kelly_lookback": 50,
      "kelly_min_trades": 15
    }
  }
}
```

**Expected Outcome:**
- Initial trades use static params (45% win, 3.5% avg win, 1.5% avg loss)
- After 15 trades, switch to dynamic calculation
- Half-Kelly reduces variance while maintaining growth optimization
- Position sizes adapt as trading performance changes

### 7. Troubleshooting & Edge Cases

* **What can go wrong:**
  - **Negative Kelly:** Win rate and payoffs produce negative expected value. System falls back to `min_step`.
  - **Kelly too aggressive:** If winning streak inflates win rate, positions may become too large. Use `max_step` ceiling.
  - **Dynamic never activates:** Not enough trades recorded. Lower `kelly_min_trades` or wait longer.

* **Error Messages:**
  ```
  [PositionSizer] Negative Kelly=-0.1234 (w=0.42, avg_w=0.0150, avg_l=0.0250). Using min_step.
  ```
  **Cause:** Expected value is negative (losses > wins). Kelly formula produces negative bet.
  **Fix:** Review strategy—it may be losing money. Don't force positive sizing for a negative EV strategy.

  ```
  [PositionSizer] Invalid avg_win=0.0000, using base_step
  ```
  **Cause:** All trades in window were losses.
  **Fix:** This is protective behavior—no Kelly calculation possible without wins.

---

## 5.5 Position Sizing in Meta Strategy

When using Meta Strategy, each sub-strategy (MR and Trend) can have its own position sizing configuration via overrides:

```json
{
  "strategy": {
    "strategy_type": "meta",
    "mean_reversion_overrides": {
      "position_sizing_mode": "volatility",
      "position_sizing_target_vol": 0.5,
      "step_allocation": 0.5
    },
    "trend_overrides": {
      "position_sizing_mode": "kelly",
      "kelly_win_rate": 0.45,
      "kelly_avg_win": 0.035,
      "kelly_avg_loss": 0.015,
      "step_allocation": 1.0
    }
  }
}
```

**Behavior:**
- When in MR regime: Uses volatility targeting with 0.5 target vol
- When in Trend regime: Uses Kelly Criterion with trend-typical params
- Dynamic Kelly maintains separate stats for each regime

---

## 5.6 Visual: Position Sizing Comparison

```
Position Step Size Over Time (Example)

1.0 ┤                    ╭──── Kelly (aggressive, adapts)
    │        ╭─────╮    ╱
0.8 ┤       ╱       ╲──╱
    │      ╱
0.6 ┤─────╱─────────────────── Volatility (inverse to vol)
    │    ╱
0.4 ┤───╱
    │
0.2 ┤──────────────────────── Static (constant)
    │
0.0 ┼──────────────────────────────────────────────→ Time
        Low Vol   Normal   High Vol   Recovery
```

---

## 5.7 Choosing the Right Mode

| Criterion | Static | Volatility | Kelly |
|-----------|--------|------------|-------|
| **Complexity** | Lowest | Medium | Highest |
| **Data Required** | None | Volatility history | Trade history (20+ trades) |
| **Best For** | Beginners, testing | All-weather trading | Long-term growth optimization |
| **Risk** | Fixed | Adaptive | Optimal (with correct params) |
| **Variance** | Constant | Reduced in high vol | Lowest (at Half-Kelly) |

**Recommendation:**
1. Start with `static` to understand baseline behavior
2. Move to `volatility` for production trading
3. Use `kelly` only after understanding the math and having reliable win rate estimates

---

*Previous Chapter: [Chapter 4: Trading Strategies](./CHAPTER_04_STRATEGIES.md)*  
*Next Chapter: [Chapter 6: Risk Management](./CHAPTER_06_RISK_MANAGEMENT.md)*
