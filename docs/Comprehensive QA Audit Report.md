# 🔍 ETH/BTC Bot v5 — Comprehensive QA Audit Report

> **Audit Date:** December 18, 2024
> **Scope:** Full codebase logic audit with Zero Trust methodology

------

## 1. Logic vs. Goal Discrepancies

### 1.1 🔴 OBJECTIVE FLAW: No Retry Logic for API Failures

**Goal:** Documentation claims the bot handles network issues gracefully with resilience to market non-stationarity.

**Reality:** The 

live_executor.py main loop has no exponential backoff or retry logic for Binance API failures. When `adapter.get_klines()` fails (line 444), it sleeps 5 seconds and continues—no retry counter, no escalating delay.



```
except Exception as e:

    log.error("Failed to fetch klines: %s", e)

    time.sleep(5)  # Fixed delay, no backoff

    continue
```

**Severity:** **Critical**

**Fix Required:** Implement exponential backoff with max 3-5 retries before entering a degraded state. After max retries, alert via Discord and pause until next bar.

------

### 1.2 🔴 OBJECTIVE FLAW: Position Fetch Fallback Uses Stale Data

**Goal:** Accurate real-time position tracking for futures trading.

**Reality:** In 

live_executor.py (lines 524-530), when `adapter.get_position()` fails, the bot falls back to `state.get("last_known_position", 0.0)`. This stale data could be hours old, causing the bot to trade against the actual exchange state.



```
except Exception as e:

    log.error("Position fetch failed, using last known: %s", e)

    current_position = state.get("last_known_position", 0.0)  # ⚠️ STALE!
```

**Severity:** **Critical**

**Fix Required:** If position fetch fails, the bot should enter a "safe mode" (block new entries, allow exits only) and alert the user, rather than trading on stale data.

------

### 1.3 🔴 OBJECTIVE FLAW: Market Order Assumes Immediate Fill

**Goal:** Robust order execution with proper fill verification.

**Reality:** After executing a market order (line 1418), the bot only waits 1 second before checking fill status. On testnet or during high latency, this may be insufficient. Partial fills are not properly reconciled.

```
time.sleep(1.0)

is_filled, filled_qty = adapter.check_order(args.symbol, oid)

if not is_filled:

    log.warning("TAKER order %s NOT FILLED yet...")

    executed_qty += filled_qty  # Adds partial, but doesn't retry
```

**Severity:** **Major**

**Fix Required:** Implement polling loop with timeout (e.g., 10s max) to wait for full fill confirmation before proceeding.

------

### 1.4 🟡 SUBJECTIVE IMPROVEMENT: Duplicate merge_strategy_params Function

**Goal:** Clean, maintainable code without duplication.

**Reality:** 

merge_strategy_params() is defined in both 

live_executor.py

 (line 52) AND imported from `core.strategy_factory` (line 41). The local definition shadows the import.



**Severity:** Minor

**Fix Required:** Remove duplicate function definition in 

live_executor.py.



------

### 1.5 🔴 OBJECTIVE FLAW: Phoenix Protocol Requires Plan Variable

**Goal:** Phoenix Protocol auto-recovery after max drawdown is hit.

**Reality:** The Phoenix recovery logic (line 864) has a condition `if maxdd_hit and 'plan' in locals()`. If strategy calculation fails (line 860: `target_w = cur_w`), the `plan` variable won't exist, and Phoenix recovery can never trigger!

```
if maxdd_hit and 'plan' in locals():  # BUG: plan might not exist on error

    reset_days = float(getattr(cfg.risk, "drawdown_reset_days", 0.0))
```

**Severity:** **Critical**

**Fix Required:** Move regime score calculation for Phoenix outside the strategy block, or handle the case where plan doesn't exist.

------

### 1.6 🟡 SUBJECTIVE IMPROVEMENT: Missing Type Hints Across Codebase

**Goal:** Production-grade, type-safe code.

**Reality:** Most functions lack proper type annotations (e.g., 

_update_risk_state returns `None` but isn't annotated). This makes static analysis and IDE support less effective.



**Severity:** Minor

**Fix Required:** Add comprehensive type hints, especially for public APIs.

------

## 2. "Mental Sandbox" Findings

### 2.1 Workflow: **Order Execution Flow**

#### Happy Path

1. Strategy calculates `target_w = 0.7`
2. Bot computes `delta_w = 0.7 - 0.0 = 0.7`
3. Order submitted → Market fills → Metrics updated → State saved ✅

#### Destructive Scenario: API Returns 500 During Order

**Current Behavior:**

```
except Exception as e:

    msg = str(e)

    reason = "insufficient_balance" if "-2010" in msg else "order_error"

    inc_rejection(instance_name, reason)

    log.exception("Taker order rejected: %s", e)

    # Continues to next bar with no retries
```

**Expected Behavior:** Retry 2-3 times with exponential backoff. If all fail, enter degraded state and alert user. Current behavior loses the trade opportunity silently.

------

### 2.2 Workflow: **Balance Fetching (Futures Mode)**

#### Happy Path

1. `adapter.get_account_balance("USDT")` returns margin balance
2. `adapter.get_position("BTCUSDT")` returns current position
3. `cur_w` calculated correctly ✅

#### Destructive Scenario: Network Timeout Mid-Execution

**Current Behavior:** Falls back to `state.get("last_known_quote")` and continues trading.

**Problem:** If balances are stale by several bars and price moved significantly, wealth calculation (`W = quote_bal`) will be wrong, potentially causing:

- Incorrect `delta_eth` calculation
- Trading more than available margin
- Max drawdown not triggering when it should

**Expected Behavior:** If balance fetch fails AND position fetch fails, halt trading until next successful fetch. Current state mixing fresh position with stale balance is dangerous.

------

### 2.3 Workflow: **Risk State Update (_update_risk_state)**

#### Happy Path

1. New HWM recorded when `wealth > equity_high`
2. Drawdown calculated: `dd_now = equity_high - wealth`
3. If `dd_now >= threshold_dd`, `maxdd_hit = True` ✅

#### State Analysis Bug: HWM Update Race

When `maxdd_hit = True`, HWM stops updating (line 224). **This is correct.** However, after Phoenix reset (line 899), `state["risk_equity_high"] = W` sets HWM to current wealth. If current wealth is at a local low, the next minor dip could immediately re-trigger max DD.

**Recommendation:** After Phoenix reset, wait N bars before enabling max DD check, or use a smoothed HWM.

------

## 3. The "Matrix of Pain" (Test Plan)

| Component          | Scenario                          | Input Data                                          | Expected Outcome                        | Type (Unit/E2E) |
| :----------------- | :-------------------------------- | :-------------------------------------------------- | :-------------------------------------- | :-------------- |
| **FuturesAdapter** | get_position() returns empty list | `{"positions": []}`                                 | Return `0.0`, no exception              | Unit            |
| **FuturesAdapter** | get_position() times out          | Network timeout after 5s                            | Raise exception, caught by caller       | Integration     |
| **FuturesAdapter** | Hedge mode (long + short)         | `[{"positionAmt": "1.5"}, {"positionAmt": "-0.5"}]` | Net position = `1.0`                    | Unit ✅ (exists) |
| **RiskState**      | Max DD hit exactly at threshold   | `wealth=8.0, HWM=10.0, max_dd_frac=0.2`             | `maxdd_hit = True`                      | Unit            |
| **RiskState**      | Daily loss limit at midnight UTC  | Cross midnight with loss                            | Reset daily counters                    | Unit            |
| **RiskState**      | Phoenix triggers after cooldown   | `reset_days=7`, `regime_score=35`                   | `maxdd_hit = False`, HWM reset          | Unit            |
| **RiskState**      | Phoenix blocked by low score      | `reset_days=7`, `regime_score=15`                   | `maxdd_hit = True` (unchanged)          | Unit            |
| **MetaStrategy**   | Regime switch hysteresis          | ADX oscillates 24-26                                | No rapid switching (buffer=2)           | Unit            |
| **MetaStrategy**   | MR signal in Trend regime         | `regime_score=30` (>25)                             | Use Trend signal, ignore MR             | Unit            |
| **PositionSizer**  | Volatility mode, high vol         | `realized_vol=1.0, target_vol=0.5`                  | `step = 0.25` (reduced)                 | Unit            |
| **PositionSizer**  | Kelly with invalid avg_win        | `kelly_avg_win=0`                                   | Fallback to `base_step`                 | Unit            |
| **PositionSizer**  | NaN volatility input              | `realized_vol=NaN`                                  | Fallback to `base_step`                 | Unit            |
| **LiveExecutor**   | Kline fetch fails 3 times         | API returns 500                                     | Retry with backoff, then safe mode      | Integration     |
| **LiveExecutor**   | Order partially filled            | `filled_qty < qty_exec`                             | Update exposure to actual, log mismatch | Integration     |
| **LiveExecutor**   | Zero balance detected             | `W = 0.0`                                           | Log warning, skip trading, alert        | E2E             |
| **LiveExecutor**   | Gate closed (high funding)        | `funding_rate = 0.1%`                               | Block long entry, allow exits           | Integration     |
| **Backtester**     | Parity with live executor step    | Same config, same data                              | `target_w` sequences match              | Integration     |
| **Backtester**     | Funding cost at 8-hour mark       | Position held across funding                        | Deduct funding cost from balance        | Unit            |

------

## 4. Recommendations for Refactoring

### 4.1 Untestable Code

| Function                | File                | Issue                                  | Recommendation                                               |
| :---------------------- | :------------------ | :------------------------------------- | :----------------------------------------------------------- |
| main()                  | live_executor.py    | **1230 lines**, does everything        | Split into: `run_strategy_loop()`, `execute_trade()`, `update_metrics()`, `handle_risk()` |
| `Backtester.simulate()` | ethbtc_accum_bot.py | **374 lines**, mixes concerns          | Extract: `apply_funding_fees()`, `calculate_trade_costs()`, update_risk_state() |
| generate_positions()    | meta_strategy.py    | Depends on two sub-strategies + regime | Already acceptable, but add mocks in tests                   |

------

### 4.2 🔴 Hardening Steps (Must Fix Now)

1. **Add circuit breaker for consecutive API failures**

   ```
   fail_count = 0
   
   MAX_FAILS = 5
   
   if exception:
   
       fail_count += 1
   
       if fail_count >= MAX_FAILS:
   
           log.critical("Circuit breaker tripped!")
   
           alerter.send("CRITICAL: API failures, halting", level="CRITICAL")
   
           sys.exit(1)
   
   else:
   
       fail_count = 0
   ```

2. **Guard against trading on stale position data**

   ```
   position_age = bar_dt - state.get("last_position_fetch_ts")
   
   if position_age > pd.Timedelta("15min"):
   
       log.error("Position data too stale, entering safe mode")
   
       target_w = 0.0  # Force exit only
   ```

3. **Add timeout wrapper for all adapter calls**

   ```
   from functools import wraps
   
   import signal
   
   
   
   def timeout(seconds=30):
   
       def decorator(func):
   
           @wraps(func)
   
           def wrapper(*args, **kwargs):
   
               # Implementation with signal.alarm or threading
   ```

4. **Validate config before trading**

   - Add sanity_check_config.py call on startup
   - Block if `max_dd_frac > 0.5` (dangerous)
   - Block if `leverage > 10` (extreme risk)

------

### 4.3 🟡 Improvement Steps (Nice to Have)

1. Add comprehensive type hints across all modules
2. Implement structured logging (JSON format) for log aggregation
3. Add health check endpoint (`/health`) separate from metrics
4. Create integration test suite using Binance testnet
5. Add pre-commit hooks for type checking (mypy)

------

## 5. Test Coverage Gap Analysis

### Current Test Files (7 files, ~320 lines total)

| File                         | Lines | Coverage Focus                |
| :--------------------------- | :---- | :---------------------------- |
| test_fixes.py                | 72    | Hedge mode, metrics existence |
| test_metrics_risk.py         | 76    | Risk mode flags               |
| test_meta_logic.py           | ~80   | Meta strategy basic           |
| test_position_sizer.py       | ~150  | Position sizing modes         |
| test_risk_modes.py           | ~100  | Risk calculations             |
| test_volatility.py           | ~100  | Volatility calculations       |
| test_backtest_live_parity.py | ~60   | Basic parity check            |

### Critical Missing Tests

1. **No tests for live_executor.py main loop** - The most critical file has zero direct tests
2. **No tests for Phoenix Protocol edge cases**
3. **No tests for API failure scenarios**
4. **No tests for order execution with partial fills**
5. **No tests for funding gate blocking logic**
6. **No E2E tests against testnet**

### Recommended Test Additions

1. Create `test_live_executor_unit.py` - Mock adapter, test decision logic
2. Create `test_phoenix_protocol.py` - All recovery scenarios
3. Create `test_order_execution.py` - Partial fills, rejections, retries
4. Create `integration/test_binance_testnet.py` - Real API calls with testnet

------

## 6. Calibration & Reality Check

### Findings Classified

| Category                    | Count | Examples                                                     |
| :-------------------------- | :---- | :----------------------------------------------------------- |
| 🔴 OBJECTIVE FLAW (Must Fix) | 4     | No retry logic, stale position fallback, Phoenix depends on plan, market order no confirmation loop |
| 🟡 SUBJECTIVE (Nice to Have) | 3     | Duplicate function, type hints, structured logging           |

### NOT Flagged (Working as Intended)

- **Hysteresis in MetaStrategy** - Properly implements buffer to prevent switching churn
- **Risk state tracking** - Correctly separates daily loss from max DD
- **Dynamic position sizing** - Proper volatility targeting implementation
- **State persistence** - Atomic write with tmp file and replace
- **Metrics export** - Comprehensive Prometheus instrumentation

------

## Summary

The codebase is generally well-structured with good observability. However, **the lack of resilience to API failures is a critical gap** for a live trading system. The Phoenix Protocol has a subtle bug that could prevent recovery. Test coverage is minimal for the most critical paths.

**Priority fixes:**

1. Add retry/backoff for all Binance API calls
2. Fix Phoenix Protocol dependency on `plan` variable
3. Implement proper order fill confirmation loop
4. Add unit tests for live_executor.py decision logic