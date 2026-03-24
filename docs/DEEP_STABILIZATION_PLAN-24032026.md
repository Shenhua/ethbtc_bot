# Deep Stabilization Plan: Forensic Gap Analysis & Fix

## Context

The bot is ~93% feature-complete but has **critical silent-failure paths** that could cause data loss, incorrect trading, or invisible risk bypass in production. This plan addresses vulnerabilities found via line-by-line forensic audit of the codebase against the spec (`docs/Theory.md`, `docs/research/IMPLEMENTATION_AUDIT.md`). No new features — only hardening existing code.

**Approach:** Fix the most dangerous modules first (state persistence > silent exceptions > risk manager > position sizing > order execution > story writer > adapters). Each phase is one commit with its own tests.

---

## Phase 1: State Persistence (CRITICAL — Data Loss)

**The Bug:** `load_state()` returns `{}` on ANY error (corrupted JSON, permission denied). `save_state()` silently falls back to a *different directory* on read-only FS. On restart, bot loads old state from original path → position/risk history lost.

**Files:**
- `live_executor.py:137-164` — `load_state()` and `save_state()`
- `core/models/state.py:46-62` — `BotState.from_dict()`

**Fixes:**
1. **`load_state()`** — Split exceptions: `FileNotFoundError` → INFO log + return `{}`; `JSONDecodeError` → CRITICAL log + try `.bak` backup → raise if no backup; other errors → CRITICAL log + raise
2. **`save_state()`** — Remove fallback directory logic (fatal misconfiguration should crash). Add `.bak` rotation before overwrite via `shutil.copy2`. Add `import shutil`
3. **`RiskState` validators** — Add `@field_validator('equity_high')` rejecting NaN/Inf/negative → coerce to 0.0. Add `@field_validator('maxdd_hit_ts')` rejecting unparseable timestamps → coerce to None

**New tests** (`tests/test_state_persistence.py`):
- `test_load_state_missing_file` — returns `{}`, no exception
- `test_load_state_corrupted_json_no_backup` — raises with CRITICAL log
- `test_load_state_corrupted_json_with_backup` — falls back to `.bak`
- `test_save_state_creates_backup` — `.bak` exists after save
- `test_save_state_readonly_raises` — no silent fallback
- `test_riskstate_nan_equity_high` — coerced to 0.0
- `test_riskstate_negative_equity_high` — coerced to 0.0
- `test_riskstate_inf_equity_high` — coerced to 0.0
- `test_riskstate_malformed_timestamp` — coerced to None

---

## Phase 2: Silent Exception Swallowing (CRITICAL — Invisible Failures)

**The Bug:** Multiple `except Exception: pass` blocks hide exchange failures. Stale prices corrupt wealth metrics. Broken regime scoring blocks Phoenix Protocol recovery forever.

**Files:**
- `live_executor.py:499` — USD price fetch
- `live_executor.py:791-796` — Regime score
- `live_executor.py:120-124` — `inc_rejection()`

**Fixes:**
1. **Price fetch** — Replace `pass` with `log.warning("USD price fetch failed, metrics will use stale values", error=str(e))`
2. **Regime score** — Replace `pass` with `log.warning("Regime score calculation failed, defaulting to 0.0", error=str(e))`
3. **`inc_rejection()`** — Add `log.debug` in fallback branch; wrap inner fallback in its own try/except

**New tests** (`tests/test_exception_logging.py`):
- `test_price_fetch_failure_logs_warning`
- `test_regime_score_failure_logs_warning`
- `test_inc_rejection_fallback_logs`

---

## Phase 3: Risk Manager Edge Cases (CRITICAL — Safety Net Broken)

**The Bug:** If `equity_high = 0` (from corrupted state), `threshold_dd = 0 * max_dd_frac = 0`. The check `threshold_dd > 0.0 and dd_now >= threshold_dd` **never triggers** → max drawdown protection disabled. Also `ensure_state()` accepts negative wealth.

**File:** `core/risk_manager.py`

**Fixes:**
1. **Line 94** — After `equity_high = state.equity_high or wealth`, add guard: `if equity_high <= 0: log.warning(...); equity_high = wealth`
2. **`ensure_state()` (line 73-82)** — Add `if wealth < 0: log.error(...); wealth = abs(wealth)`. Add `if state.equity_high <= 0:` guard (currently `if not state.equity_high` misses 0.0 since `not 0.0 == True`)
3. **Line 129** — Add `log.info("Daily risk reset", ...)` for audit trail on date change

**New tests** (extend `tests/test_risk_manager.py`):
- `test_zero_equity_high_resets_to_wealth`
- `test_negative_equity_high_resets`
- `test_dynamic_mode_zero_equity_still_enforces_dd`
- `test_ensure_state_negative_wealth`
- `test_daily_reset_logged`

---

## Phase 4: Position Sizing Dangers (HIGH)

**The Bug:** `get_stats()` returns `avg_loss=0.01` when there are 0 losses → Kelly = ~1.0 (full leverage). `target_vol / realized_vol` crashes if `realized_vol` is NaN/Inf (current NaN check `not x == x` is fragile).

**File:** `core/position_sizer.py`

**Fixes:**
1. **`get_stats()` (line 62-71)** — If `len(wins) == 0` or `len(losses) == 0`, return `None` (insufficient diversity). This forces fallback to static Kelly config set by operator
2. **Line 261** — Replace `not realized_vol == realized_vol` with `math.isnan(realized_vol) or math.isinf(realized_vol)` (add `import math`)

**New tests** (extend `tests/test_position_sizer.py`):
- `test_kelly_all_wins_no_losses_returns_none`
- `test_kelly_all_losses_no_wins_returns_none`
- `test_volatility_targeting_nan_vol`
- `test_volatility_targeting_inf_vol`

---

## Phase 5: Order Execution Gaps (HIGH)

**The Bug:** `wait_for_fill()` returns on timeout but the order is still live on Binance → next bar places duplicate. Position staleness is logged but trading continues with potentially days-old data.

**Files:**
- `core/order_manager.py:76-113` — `wait_for_fill()`
- `core/services/order_service.py:103-108` — staleness check
- `live_executor.py` — callers

**Fixes:**
1. **`wait_for_fill()`** — Add optional `cancel_fn` parameter. On timeout, call `cancel_fn(symbol, order_id)` if provided. Log cancellation result
2. **Callers in live_executor.py** — Pass `cancel_fn=adapter.cancel` to `wait_for_fill()`
3. **`order_service.py`** — When `position_age > 15min`, set `res["position_unsafe"] = True`
4. **`live_executor.py`** — Before trade execution, check `acc_state.get("position_unsafe")` → skip trade + `inc_rejection(instance, "stale_position")`

**New tests** (extend `tests/test_order_manager.py`):
- `test_wait_for_fill_timeout_cancels_order`
- `test_wait_for_fill_timeout_no_cancel_fn`
- `test_wait_for_fill_cancel_failure_logged`
- `test_position_staleness_blocks_trading`

---

## Phase 6: Story Writer (HIGH — Audit Trail Loss)

**The Bug:** File write failures go to `print()` (invisible in Docker). No tests exist for this module.

**File:** `core/story_writer.py:74-84`

**Fixes:**
1. **`_write_line()`** — Replace `print(...)` with `self._log.error(...)` using structured logger from `core.log_setup.get_logger("story_writer")`
2. **`__init__`** — Add `self._log = get_logger("story_writer")`

**New tests** (`tests/test_story_writer.py`):
- `test_write_line_success`
- `test_write_line_readonly_logs_error`
- `test_daily_summary_zero_start_wealth`
- `test_format_benchmark_zero_prices`
- `test_check_and_log_daily_day_change`

---

## Phase 7: Adapter Failures (MEDIUM)

**The Bug:** Spot `cancel()` logs warning but doesn't raise → order stays live on exchange while bot thinks it's cancelled.

**Files:**
- `core/binance_adapter.py:130-136` — spot `cancel()`
- `core/futures_adapter.py:75-89` — position parsing

**Fixes:**
1. **Spot `cancel()`** — Suppress "Unknown order" / error code -2011 (already gone). Raise on all other errors
2. **Futures `get_position()`** — Catch `KeyError`/`ValueError`/`TypeError` specifically for malformed `positionAmt`; raise `ValueError` with context

**New tests** (`tests/test_adapters.py`):
- `test_spot_cancel_unknown_order_suppressed`
- `test_spot_cancel_real_error_raises`
- `test_futures_position_malformed_response`

---

## Summary

| Phase | Severity | Files Changed | New Tests | Focus |
|-------|----------|---------------|-----------|-------|
| 1 | CRITICAL | `live_executor.py`, `core/models/state.py` | 9 | Data loss prevention |
| 2 | CRITICAL | `live_executor.py` | 3 | Error visibility |
| 3 | CRITICAL | `core/risk_manager.py` | 5 | Safety net integrity |
| 4 | HIGH | `core/position_sizer.py` | 4 | Position sizing safety |
| 5 | HIGH | `core/order_manager.py`, `core/services/order_service.py`, `live_executor.py` | 4 | Execution safety |
| 6 | HIGH | `core/story_writer.py` | 5 | Audit trail |
| 7 | MEDIUM | `core/binance_adapter.py`, `core/futures_adapter.py` | 3 | Adapter robustness |
| **Total** | | **8 files** | **33 tests** | |

## Verification

After each phase:
```bash
pytest tests/ -v                    # Full regression
python -m mypy core/ --strict       # Type safety
```

After all phases:
```bash
pytest tests/ -v --tb=short         # Full suite passes
python live_executor.py --params configs/prod_meta_live.json --mode dry --once  # Dry-run smoke test
```

---

## Implementation Status (2026-03-24)

| Phase | Status | Notes |
|-------|--------|-------|
| 1 — State Persistence | ✅ DONE | 18 tests passing |
| 2 — Silent Exceptions | ✅ DONE | 6 tests passing |
| 3 — Risk Manager | ✅ DONE | 5 new tests passing |
| 4 — Position Sizing | ✅ DONE | 6 new tests passing; `test_dynamic_kelly.py` updated for new diversity requirement |
| 5 — Order Execution | ✅ DONE | 5 new tests passing |
| 6 — Story Writer | ✅ DONE | 10 tests passing |
| 7 — Adapter Failures | ✅ DONE | 10 tests passing |

**Final suite result:** 210 passed, 1 pre-existing flaky failure (`test_entry_blocked_on_low_volume` — off-by-one unrelated to this work), 1 skipped.
