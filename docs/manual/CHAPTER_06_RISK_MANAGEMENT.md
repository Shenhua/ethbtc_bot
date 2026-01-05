# Chapter 6: Risk Management

> **Purpose:** This chapter provides exhaustive documentation of the risk management system, covering High Water Mark tracking, Maximum Drawdown protection, Daily Loss Limits, the Phoenix Protocol for automated recovery, and the two risk modes (fixed_basis vs dynamic).

---

## 6.1 Risk Management Architecture

### 1. Concept & "The Why"

* **What it is:** A multi-layered risk protection system that monitors account equity and halts trading when predefined loss limits are breached. Acts as the "circuit breaker" of the trading system.

* **Purpose:** Capital preservation is the #1 priority (per Project Philosophy). This system exists to:
  1. Prevent catastrophic loss during black swan events
  2. Limit daily losses to prevent compounding mistakes
  3. Provide automated recovery via Phoenix Protocol
  4. Track peak equity for accurate drawdown calculation

* **Location:** 
  - Core Module: [`core/risk_manager.py`](../../core/risk_manager.py)
  - State Model: [`core/models/state.py`](../../core/models/state.py) → `RiskState`
  - Configuration: [`core/config_schema.py`](../../core/config_schema.py) → `Risk` class

### 2. Configuration & Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `risk_mode` | enum | `fixed_basis`, `dynamic` | `fixed_basis` | How thresholds are calculated |
| `basis_btc` | float | 0.0–100000.0 | 0.0 | Reference capital for fixed mode |
| `max_dd_btc` | float | 0.0–100.0 | 0.0 | Max drawdown (absolute BTC) |
| `max_dd_frac` | float | 0.0–1.0 | 0.0 | Max drawdown (% of HWM) |
| `max_daily_loss_btc` | float | 0.0–100.0 | 0.0 | Daily loss limit (absolute BTC) |
| `max_daily_loss_frac` | float | 0.0–1.0 | 0.0 | Daily loss limit (% of day start) |
| `drawdown_reset_days` | float | 0.0–365.0 | 0.0 | Days before Phoenix reset (0 = disabled) |
| `drawdown_reset_score` | float | 0.0–100.0 | 25.0 | Regime score required for Phoenix reset |

### 3. State Tracking

The risk system maintains state in `RiskState` (persisted to `state.json`):

| Field | Type | Description |
|-------|------|-------------|
| `equity_high` | float | High Water Mark (peak equity) |
| `current_date` | string | Current date for daily tracking |
| `daily_start_wealth` | float | Wealth at start of current day |
| `daily_limit_hit` | bool | True if daily loss limit exceeded |
| `maxdd_hit` | bool | True if max drawdown exceeded |
| `maxdd_hit_ts` | string | Timestamp when max DD was hit |

**Example State File:**
```json
{
  "risk_equity_high": 1.2345,
  "risk_current_date": "2026-01-05",
  "risk_daily_start_wealth": 1.2100,
  "risk_daily_limit_hit": false,
  "risk_maxdd_hit": false,
  "risk_maxdd_hit_ts": null
}
```

---

## 6.2 High Water Mark (HWM) Tracking

### 1. Concept & "The Why"

* **What it is:** A running maximum of account equity used as the reference point for drawdown calculations. HWM only increases—it never decreases.

* **Purpose:** Ensures drawdown is measured from the peak, not the starting balance. This captures the true "pain" of losses after profitable periods.

* **Location:** [`core/risk_manager.py`](../../core/risk_manager.py) → `RiskManager.update()`

### 2. Configuration & Parameters

| Parameter | Description |
|-----------|-------------|
| `equity_high` (state) | The current HWM value, persisted across restarts |

**Hidden Logic:**
- HWM is updated ONLY when `maxdd_hit = False`
- If max drawdown is hit, HWM freezes until Phoenix reset
- HWM resets to current wealth after Phoenix activation

### 3. Formula

```python
# HWM Update Logic (every bar)
if not maxdd_hit:
    if wealth > equity_high:
        equity_high = wealth

# Drawdown Calculation
drawdown = equity_high - wealth
drawdown_pct = drawdown / equity_high
```

**Example:**
```
Bar 1: wealth = 1.00 BTC → equity_high = 1.00
Bar 2: wealth = 1.05 BTC → equity_high = 1.05 (new peak)
Bar 3: wealth = 1.03 BTC → equity_high = 1.05 (unchanged)
        drawdown = 0.02 BTC (1.9%)
Bar 4: wealth = 1.08 BTC → equity_high = 1.08 (new peak)
Bar 5: wealth = 0.95 BTC → equity_high = 1.08 (unchanged)
        drawdown = 0.13 BTC (12%)
```

### 4. Step-by-Step Guide

1. **View current HWM:**
   ```bash
   cat run_state/eth/state.json | jq '.risk_equity_high'
   # Output: 1.2345
   ```

2. **Manually reset HWM (advanced):**
   ```bash
   jq '.risk_equity_high = 1.0' run_state/eth/state.json > tmp.json
   mv tmp.json run_state/eth/state.json
   ```

### 5. Troubleshooting

* **What can go wrong:**
  - **HWM not updating:** Check if `maxdd_hit = true` (freezes HWM)
  - **HWM too high after account withdrawal:** Manual reset required

---

## 6.3 Maximum Drawdown Protection

### 1. Concept & "The Why"

* **What it is:** A hard limit on how much equity can decline from the High Water Mark before trading halts. When breached, `maxdd_hit = True` and all trading stops.

* **Purpose:** Prevents catastrophic loss during adverse market conditions. The 15-20% target drawdown protects against ruin.

* **Location:** [`core/risk_manager.py`](../../core/risk_manager.py) → `RiskManager.update()` lines 116-125

### 2. Configuration & Parameters

| Mode | Parameter | Example | Threshold Calculation |
|------|-----------|---------|----------------------|
| `fixed_basis` | `max_dd_btc` | 0.15 | Halt if DD > 0.15 BTC |
| `dynamic` | `max_dd_frac` | 0.20 | Halt if DD > 20% of HWM |

**Formula by Mode:**

```python
# Fixed Basis Mode
threshold_dd = max_dd_btc  # Absolute value in BTC

# Dynamic Mode
threshold_dd = equity_high * max_dd_frac  # Percentage of peak

# Trigger Check
if drawdown >= threshold_dd:
    maxdd_hit = True
    maxdd_hit_ts = current_timestamp
```

### 3. Step-by-Step Guide

1. **Configure max drawdown (dynamic 20%):**
   ```json
   {
     "risk": {
       "risk_mode": "dynamic",
       "max_dd_frac": 0.20
     }
   }
   ```

2. **Configure max drawdown (fixed 0.1 BTC):**
   ```json
   {
     "risk": {
       "risk_mode": "fixed_basis",
       "max_dd_btc": 0.1
     }
   }
   ```

3. **Check if max DD hit:**
   ```bash
   cat run_state/eth/state.json | jq '.risk_maxdd_hit, .risk_maxdd_hit_ts'
   # Output: true, "2026-01-04T15:30:00"
   ```

### 4. Real-World Use Case (The "Cookbook")

**Scenario:** Trader with 1 BTC account wants 15% max drawdown protection.

**Configuration:**
```json
{
  "risk": {
    "risk_mode": "dynamic",
    "max_dd_frac": 0.15
  }
}
```

**Behavior:**
- Initial: HWM = 1.0 BTC
- Growth phase: HWM → 1.2 BTC
- Threshold now: 1.2 × 15% = 0.18 BTC
- If wealth drops to 1.02 BTC: drawdown = 0.18 BTC → HALT

**Expected Outcome:**
- Trading halts immediately
- Alert sent (if configured)
- `maxdd_hit = true` persisted to state
- No new trades until Phoenix reset

### 5. Troubleshooting & Edge Cases

* **What can go wrong:**
  - **Trading stops unexpectedly:** Max DD hit. Check state file.
  - **Threshold too tight:** Lower `max_dd_frac` or switch to `fixed_basis`.

* **Error Messages** (in logs):
  ```
  [RISK] Max drawdown hit: DD=0.1823 BTC (18.23%), threshold=0.1800 BTC
  ```

* **Edge Case:** If both `max_dd_btc` and `max_dd_frac` are set:
  - In `fixed_basis` mode: Uses `max_dd_btc`
  - In `dynamic` mode: Uses `max_dd_frac`

---

## 6.4 Daily Loss Limits

### 1. Concept & "The Why"

* **What it is:** A per-day loss limit that resets at UTC midnight. Prevents compounding losses during bad trading days.

* **Purpose:** Bad days happen. This feature ensures one terrible day doesn't compound into catastrophe.

* **Location:** [`core/risk_manager.py`](../../core/risk_manager.py) → `RiskManager.update()` lines 127-142

### 2. Configuration & Parameters

| Mode | Parameter | Example | Threshold |
|------|-----------|---------|-----------|
| `fixed_basis` | `max_daily_loss_btc` | 0.03 | Halt if daily loss > 0.03 BTC |
| `dynamic` | `max_daily_loss_frac` | 0.03 | Halt if daily loss > 3% of day-start wealth |

**Formula:**

```python
# Calculate Daily P&L
daily_pnl = wealth - daily_start_wealth

# Threshold by Mode
if risk_mode == "dynamic":
    threshold_loss = daily_start_wealth * max_daily_loss_frac
else:
    threshold_loss = max_daily_loss_btc

# Trigger Check (note: daily_pnl is negative for losses)
if daily_pnl <= -threshold_loss:
    daily_limit_hit = True
```

**Reset Logic:**
```python
# At UTC midnight
if current_date != previous_date:
    daily_start_wealth = wealth  # Reset reference
    daily_limit_hit = False      # Clear flag
```

### 3. Step-by-Step Guide

1. **Configure 3% daily loss limit:**
   ```json
   {
     "risk": {
       "risk_mode": "dynamic",
       "max_daily_loss_frac": 0.03
     }
   }
   ```

2. **Check daily status:**
   ```bash
   cat run_state/eth/state.json | jq '.risk_current_date, .risk_daily_start_wealth, .risk_daily_limit_hit'
   # Output: "2026-01-05", 1.1234, false
   ```

3. **Calculate remaining daily budget:**
   ```bash
   # Example: daily_start = 1.1234, max_daily_loss_frac = 0.03
   # Budget = 1.1234 × 0.03 = 0.0337 BTC
   ```

### 4. Real-World Use Case (The "Cookbook")

**Scenario:** Trader wants maximum 5% daily loss on futures.

**Configuration:**
```json
{
  "risk": {
    "risk_mode": "dynamic",
    "max_daily_loss_frac": 0.05,
    "max_dd_frac": 0.20
  }
}
```

**Expected Outcome:**
- If day starts with 1.0 BTC, trading halts if wealth drops to 0.95 BTC
- At UTC midnight, limit resets regardless of prior day
- Works independently of max drawdown (both can halt trading)

### 5. Troubleshooting

* **What can go wrong:**
  - **Daily limit triggers frequently:** Threshold too tight. Increase to 5-10%.
  - **Limit doesn't reset:** Timezone mismatch. System uses UTC.

* **Edge Case:** If a session starts mid-day and `daily_start_wealth` isn't set, it initializes to current wealth (not midnight wealth).

---

## 6.5 Phoenix Protocol

### 1. Concept & "The Why"

* **What it is:** An automated recovery mechanism that resets the trading halt after max drawdown is hit. Named "Phoenix" because the bot rises from the ashes.

* **Purpose:** Max drawdown halts trading—but should it halt forever? Phoenix Protocol provides a structured path back:
  1. Wait for a cooling-off period (prevents revenge trading)
  2. Only resume when market conditions are favorable (regime score check)
  3. Reset HWM to current wealth (fresh start)

* **Location:** [`core/risk_manager.py`](../../core/risk_manager.py) → `RiskManager.can_phoenix_reset()` and `reset_phoenix()`

### 2. Configuration & Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `drawdown_reset_days` | float | 0.0–365.0 | 0.0 | Days to wait (0 = disabled) |
| `drawdown_reset_score` | float | 0.0–100.0 | 25.0 | Minimum regime score to reset |

**Reset Conditions (BOTH must be true):**
```python
# Condition 1: Time Elapsed
time_passed = current_time - maxdd_hit_ts
time_ok = time_passed >= (drawdown_reset_days × 86400 seconds)

# Condition 2: Favorable Regime
score_ok = current_regime_score >= drawdown_reset_score

# Phoenix Activates
if time_ok AND score_ok:
    reset_phoenix()
```

### 3. Step-by-Step Guide

1. **Configure Phoenix Protocol:**
   ```json
   {
     "risk": {
       "max_dd_frac": 0.15,
       "drawdown_reset_days": 7.0,
       "drawdown_reset_score": 30.0
     }
   }
   ```

2. **Monitor Phoenix status:**
   ```bash
   # Check when max DD was hit
   cat run_state/eth/state.json | jq '.risk_maxdd_hit_ts'
   # Output: "2026-01-01T12:00:00"
   
   # Calculate days elapsed
   # If today is 2026-01-08, that's 7 days → time_ok = true
   ```

3. **Manually trigger Phoenix (advanced):**
   ```bash
   jq '.risk_maxdd_hit = false | .risk_maxdd_hit_ts = null | .risk_equity_high = 1.0' \
     run_state/eth/state.json > tmp.json
   mv tmp.json run_state/eth/state.json
   ```

### 4. Real-World Use Case (The "Cookbook")

**Scenario:** Bot hit max DD during a flash crash. Want automatic recovery after 7 days if market stabilizes.

**Configuration:**
```json
{
  "risk": {
    "risk_mode": "dynamic",
    "max_dd_frac": 0.15,
    "drawdown_reset_days": 7.0,
    "drawdown_reset_score": 30.0
  }
}
```

**Timeline:**
```
Day 0 (crash):
  - Max DD hit at 15%
  - maxdd_hit = true
  - Trading halts

Days 1-6:
  - Trading remains halted
  - Phoenix checks each bar but time_ok = false

Day 7+:
  - time_ok = true
  - If regime_score >= 30 (favorable trend/ranging):
    - Phoenix activates!
    - maxdd_hit = false
    - HWM resets to current wealth
    - Trading resumes
  - If regime_score < 30:
    - Remains halted (wait for better conditions)
```

**Expected Outcome:**
- Automatic resume after 7+ days in favorable market
- No manual intervention required
- Fresh HWM prevents immediate re-trigger

### 5. Troubleshooting & Edge Cases

* **What can go wrong:**
  - **Phoenix never activates:** `regime_score` consistently below threshold. Lower `drawdown_reset_score`.
  - **Phoenix disabled:** `drawdown_reset_days = 0` disables the feature.

* **Edge Case:** If `maxdd_hit_ts` is corrupted or missing, Phoenix check returns `False` (fail-safe).

* **Edge Case:** Timezone handling—if state timestamp lacks timezone and current bar has timezone, it's coerced to match.

---

## 6.6 Risk Modes Comparison

### Fixed Basis Mode

Uses absolute BTC values for thresholds. Best for:
- Fixed allocation accounts
- Accounts where you don't want limits scaling with growth

```json
{
  "risk": {
    "risk_mode": "fixed_basis",
    "basis_btc": 1.0,
    "max_dd_btc": 0.15,
    "max_daily_loss_btc": 0.03
  }
}
```

**Behavior:**
- Max DD threshold: Always 0.15 BTC (regardless of equity)
- Daily limit: Always 0.03 BTC

### Dynamic Mode

Uses percentages calculated from peak equity. Best for:
- Compounding accounts
- Accounts where you want limits to scale with growth

```json
{
  "risk": {
    "risk_mode": "dynamic",
    "max_dd_frac": 0.15,
    "max_daily_loss_frac": 0.03
  }
}
```

**Behavior:**
- Max DD threshold: 15% of HWM (grows with success)
- Daily limit: 3% of day-start wealth

### Comparison Table

| Aspect | Fixed Basis | Dynamic |
|--------|-------------|---------|
| **Threshold Type** | Absolute (BTC) | Percentage (%) |
| **Reference Point** | `basis_btc` (constant) | HWM (growing) |
| **As Equity Grows** | Limits stay same | Limits increase proportionally |
| **Best For** | Conservative, fixed capital | Aggressive, compounding |
| **Complexity** | Simple | More nuanced |

---

## 6.7 Visual: Risk State Machine

```
                    ┌─────────────────────────────────┐
                    │         NORMAL TRADING          │
                    │  maxdd_hit = false              │
                    │  daily_limit_hit = false        │
                    └─────────────────────────────────┘
                              │            │
              Daily Loss > Threshold       DD > Max DD
                              │            │
                              ▼            ▼
    ┌──────────────────────────┐    ┌──────────────────────────┐
    │     DAILY LIMIT HIT      │    │      MAX DD HIT          │
    │  daily_limit_hit = true  │    │  maxdd_hit = true        │
    │  Trading HALTED          │    │  Trading HALTED          │
    └──────────────────────────┘    └──────────────────────────┘
              │                              │
         UTC Midnight                 Phoenix Protocol
         (Auto Reset)                (Time + Score OK)
              │                              │
              ▼                              ▼
    ┌──────────────────────────┐    ┌──────────────────────────┐
    │     DAILY RESET          │    │    PHOENIX RESET         │
    │  daily_limit_hit = false │    │  maxdd_hit = false       │
    │  daily_start = wealth    │    │  HWM = wealth            │
    └──────────────────────────┘    └──────────────────────────┘
              │                              │
              └──────────────┬───────────────┘
                             ▼
                    ┌─────────────────────────────────┐
                    │       RESUME TRADING            │
                    └─────────────────────────────────┘
```

---

## 6.8 Complete Risk Configuration Example

```json
{
  "risk": {
    "risk_mode": "dynamic",
    "max_dd_frac": 0.15,
    "max_dd_btc": 0.0,
    "max_daily_loss_frac": 0.03,
    "max_daily_loss_btc": 0.0,
    "drawdown_reset_days": 7.0,
    "drawdown_reset_score": 30.0
  }
}
```

**This configuration:**
- Uses dynamic thresholds (scales with equity)
- Halts at 15% drawdown from HWM
- Halts if daily loss exceeds 3%
- Phoenix auto-resets after 7 days if regime score ≥ 30
- Daily limit resets at UTC midnight

---

*Previous Chapter: [Chapter 5: Position Sizing](./CHAPTER_05_POSITION_SIZING.md)*  
*Next Chapter: [Chapter 7: Execution Layer](./CHAPTER_07_EXECUTION.md)*
