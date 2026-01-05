# Chapter 13: Testing & Quality Assurance

> **Purpose:** This chapter provides exhaustive documentation of the testing infrastructure, including unit tests, regression testing, golden snapshots, and QA best practices.

---

## 13.1 Testing Architecture

### 1. Concept & "The Why"

* **What it is:** A comprehensive pytest-based test suite covering strategy logic, risk management, position sizing, and execution components.

* **Purpose:** 
  1. **Prevent regressions:** Catch breaking changes before deployment
  2. **Validate business logic:** Ensure strategies behave as designed
  3. **Document behavior:** Tests serve as executable specifications
  4. **Enable refactoring:** Safe code changes with test coverage

* **Location:** [`tests/`](../../tests/) directory

### 2. Test Suite Structure

```
tests/
├── test_phoenix_protocol.py      # Phoenix reset conditions
├── test_position_sizer.py        # Static, volatility, Kelly modes
├── test_risk_manager.py          # HWM, max DD, daily loss
├── test_risk_modes.py            # Fixed vs dynamic modes
├── test_meta_logic.py            # Meta strategy regime switching
├── test_live_executor_unit.py    # Executor logic mocking
├── test_resilience.py            # Retry and circuit breaker
├── test_regression.py            # Golden snapshot parity
├── test_bollinger_squeeze.py     # Bollinger filter
├── test_funding_counter.py       # Funding counter-trend
├── test_htf_filter.py            # Higher timeframe filter
├── test_rsi_filter.py            # RSI divergence filter
├── test_volume_confirm.py        # Volume confirmation
├── test_volatility.py            # Volatility calculations
├── test_dynamic_kelly.py         # Dynamic Kelly updates
├── test_order_manager.py         # Order lifecycle
├── test_backtest_live_parity.py  # Backtest vs live consistency
├── test_parity_sync.py           # State synchronization
└── golden_snapshot.json          # Regression baseline
```

---

## 13.2 Running Tests

### 1. Concept & "The Why"

* **What it is:** pytest-based test execution with various filtering and reporting options.

* **Purpose:** Fast feedback on code changes.

### 2. Step-by-Step Guide

1. **Run all tests:**
   ```bash
   pytest tests/ -v
   ```

2. **Run specific test file:**
   ```bash
   pytest tests/test_phoenix_protocol.py -v
   ```

3. **Run specific test class:**
   ```bash
   pytest tests/test_phoenix_protocol.py::TestPhoenixConditions -v
   ```

4. **Run specific test function:**
   ```bash
   pytest tests/test_phoenix_protocol.py::TestPhoenixConditions::test_phoenix_triggers_with_both_conditions_met -v
   ```

5. **Run tests matching pattern:**
   ```bash
   pytest tests/ -v -k "phoenix"
   ```

6. **Run with coverage:**
   ```bash
   pytest tests/ --cov=core --cov-report=html
   open htmlcov/index.html
   ```

7. **Run with verbose failure output:**
   ```bash
   pytest tests/ -v --tb=long
   ```

### 3. Expected Output

```
========================= test session starts ==========================
platform darwin -- Python 3.12.0, pytest-8.0.0
collected 125 items

tests/test_phoenix_protocol.py::TestPhoenixConditions::test_phoenix_not_triggered_before_cooldown PASSED
tests/test_phoenix_protocol.py::TestPhoenixConditions::test_phoenix_not_triggered_with_low_score PASSED
tests/test_phoenix_protocol.py::TestPhoenixConditions::test_phoenix_triggers_with_both_conditions_met PASSED
...
========================= 125 passed in 8.42s ==========================
```

---

## 13.3 Phoenix Protocol Tests

### 1. Concept & "The Why"

* **What it is:** Tests for the Phoenix Protocol recovery logic after max drawdown events.

* **Purpose:** Verify correct behavior of trading resumption conditions.

* **Location:** [`tests/test_phoenix_protocol.py`](../../tests/test_phoenix_protocol.py)

### 2. Test Classes

| Class | Purpose |
|-------|---------|
| `TestPhoenixConditions` | Test activation conditions |
| `TestPhoenixReset` | Test state reset logic |
| `TestPhoenixDisabled` | Test disabled state |
| `TestPhoenixEdgeCases` | Test boundary conditions |

### 3. Key Test Cases

```python
def test_phoenix_not_triggered_before_cooldown():
    """Phoenix should NOT trigger if time cooldown hasn't passed."""
    config = RiskConfig(
        drawdown_reset_days=1.0,  # 24 hour cooldown
        drawdown_reset_score=30.0
    )
    rm = RiskManager(config)
    
    crash_time = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    # Only 12 hours later (less than 24 hour cooldown)
    check_time = pd.Timestamp("2024-01-01 12:00:00", tz="UTC")
    
    state = RiskState(maxdd_hit=True, maxdd_hit_ts=crash_time.isoformat())
    
    # High score but not enough time
    can_reset = rm.can_phoenix_reset(state, check_time, current_regime_score=50.0)
    assert can_reset == False

def test_phoenix_triggers_with_both_conditions_met():
    """Phoenix SHOULD trigger when both conditions are satisfied."""
    # ... time passed AND score >= threshold
    assert can_reset == True
```

---

## 13.4 Position Sizer Tests

### 1. Concept & "The Why"

* **What it is:** Tests for all three position sizing modes (static, volatility, Kelly).

* **Purpose:** Ensure position sizing calculations are mathematically correct.

* **Location:** [`tests/test_position_sizer.py`](../../tests/test_position_sizer.py)

### 2. Test Classes

| Class | Purpose |
|-------|---------|
| `TestPositionSizerConfig` | Config validation |
| `TestStaticMode` | Static mode returns base_step |
| `TestVolatilityTargeting` | Vol scaling formulas |
| `TestKellyCriterion` | Kelly calculations |
| `TestEdgeCases` | Boundary conditions |
| `TestRealWorldScenarios` | Market scenarios |

### 3. Key Test Cases

```python
def test_low_volatility_increases_step():
    """Test step size increases in low volatility."""
    config = PositionSizerConfig(
        mode="volatility",
        base_step=0.5,
        target_vol=0.5,
        min_step=0.1,
        max_step=1.0
    )
    sizer = PositionSizer(config)
    
    # Low volatility -> larger step
    # step = 0.5 * (0.5 / 0.25) = 1.0
    assert sizer.calculate_step(0.25) == 1.0  # Capped at max_step

def test_high_volatility_decreases_step():
    """Test step size decreases in high volatility."""
    # step = 0.5 * (0.5 / 1.0) = 0.25
    assert sizer.calculate_step(1.0) == 0.25
```

---

## 13.5 Regression Testing

### 1. Concept & "The Why"

* **What it is:** Golden snapshot testing that compares current strategy output against a known-good baseline.

* **Purpose:** 
  - Detect unintended signal changes
  - Ensure refactoring doesn't alter behavior
  - Provide regression safety net

* **Location:** [`tests/test_regression.py`](../../tests/test_regression.py)

### 2. Golden Snapshot Process

```
┌─────────────────────────────────────────────────────────────────────┐
│                    GOLDEN SNAPSHOT WORKFLOW                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   1. CAPTURE (Initial or after intentional changes)                │
│      python tools/capture_snapshot.py                               │
│            │                                                        │
│            ▼                                                        │
│      tests/golden_snapshot.json                                     │
│      (Config + Input Data + Expected Signals)                       │
│                                                                     │
│   2. TEST (On every code change)                                    │
│      pytest tests/test_regression.py                                │
│            │                                                        │
│            ▼                                                        │
│      Compare current output vs golden snapshot                      │
│            │                                                        │
│            ├── MATCH ──────▶ ✅ PASS                                │
│            └── MISMATCH ───▶ ❌ FAIL (investigate!)                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 3. Step-by-Step Guide

1. **Create golden snapshot:**
   ```bash
   python tools/capture_snapshot.py \
     --config configs/prod_eth_long_wfo_robust.json \
     --data data/raw/ETHBTC_15m_sample.csv \
     --out tests/golden_snapshot.json
   ```

2. **Run regression test:**
   ```bash
   pytest tests/test_regression.py -v
   ```

3. **Expected output (success):**
   ```
   tests/test_regression.py::test_golden_regression_parity PASSED
   ✅ Regression check passed: Parity with golden snapshot confirmed.
   ```

4. **Expected output (failure):**
   ```
   AssertionError: Strategy Target Weights are different
   
   [left]:  Series([0.0, 0.5, 0.5, 1.0, ...])
   [right]: Series([0.0, 0.0, 0.5, 1.0, ...])
   ```

### 4. Troubleshooting

**Snapshot not found:**
```
SKIPPED: Golden snapshot not found. Run tools/capture_snapshot.py first.
```
**Fix:** Generate snapshot using `capture_snapshot.py`.

**Intentional signal changes:**
If you intentionally changed strategy logic:
1. Verify changes are correct
2. Re-capture snapshot: `python tools/capture_snapshot.py ...`
3. Re-run tests

---

## 13.6 Risk Management Tests

### 1. Concept & "The Why"

* **What it is:** Tests for HWM tracking, drawdown detection, and daily loss limits.

* **Purpose:** Ensure risk controls function correctly under all conditions.

* **Location:** 
  - [`tests/test_risk_manager.py`](../../tests/test_risk_manager.py)
  - [`tests/test_risk_modes.py`](../../tests/test_risk_modes.py)

### 2. Key Test Areas

| Area | Tests |
|------|-------|
| **HWM Tracking** | Peak updates, no decrease on loss |
| **Max DD Detection** | Trigger at threshold, not before |
| **Daily Loss Limit** | Reset at UTC midnight |
| **Risk Modes** | Fixed vs dynamic calculations |

### 3. Example Test

```python
def test_maxdd_triggers_at_threshold():
    """Max DD flag should trigger exactly at threshold."""
    config = RiskConfig(max_dd_frac=0.10)  # 10% max DD
    rm = RiskManager(config)
    
    state = RiskState(equity_high=1.0)
    
    # 9% DD - should NOT trigger
    rm.check_maxdd(state, wealth=0.91)
    assert state.maxdd_hit == False
    
    # 10% DD - SHOULD trigger
    rm.check_maxdd(state, wealth=0.90)
    assert state.maxdd_hit == True
```

---

## 13.7 Strategy Filter Tests

### 1. Concept & "The Why"

* **What it is:** Tests for enhanced strategy filters (Bollinger, RSI, Volume, Funding, HTF).

* **Purpose:** Verify filter logic matches design.

* **Location:** Individual test files per filter.

### 2. Filter Test Files

| Filter | Test File |
|--------|-----------|
| Bollinger Squeeze | `test_bollinger_squeeze.py` |
| RSI Divergence | `test_rsi_filter.py` |
| Volume Confirmation | `test_volume_confirm.py` |
| Funding Counter-Trend | `test_funding_counter.py` |
| Higher Timeframe | `test_htf_filter.py` |

### 3. Example: Volume Confirmation Test

```python
def test_volume_confirms_buy_signal():
    """Buy signal should only fire with above-average volume."""
    params = StratParams(volume_confirm=True, volume_mult=1.5)
    strat = EthBtcStrategy(params)
    
    # Generate signals on data with varying volume
    signals = strat.generate_positions(df_with_volume)
    
    # Verify BUY only when volume > 1.5x average
    buy_bars = signals[signals["target_w"] > 0]
    for idx in buy_bars.index:
        assert df.loc[idx, "volume"] >= avg_volume * 1.5
```

---

## 13.8 Executor Tests

### 1. Concept & "The Why"

* **What it is:** Unit tests for the live executor with mocked exchange adapters.

* **Purpose:** Test executor logic without real API calls.

* **Location:** [`tests/test_live_executor_unit.py`](../../tests/test_live_executor_unit.py)

### 2. Key Test Areas

| Area | Description |
|------|-------------|
| **Signal Processing** | Delta calculation from strategy signals |
| **Order Placement** | Maker order creation |
| **Fill Handling** | Balance updates on fills |
| **Timeout Logic** | Fallback to taker after TTL |
| **Risk Checks** | Skipping trades when risk halted |

### 3. Example: Mocked Adapter Test

```python
@pytest.fixture
def mock_adapter():
    """Create a mock exchange adapter."""
    adapter = MagicMock()
    adapter.get_balance.return_value = {"BTC": 1.0, "ETH": 10.0}
    adapter.get_book.return_value = {"bids": [[0.034, 100]], "asks": [[0.0341, 100]]}
    return adapter

def test_executor_places_limit_order(mock_adapter):
    """Executor should place limit maker order for signals."""
    executor = LiveExecutor(adapter=mock_adapter, config=test_config)
    executor.process_signal(target_w=0.5)
    
    mock_adapter.place_limit_maker.assert_called_once()
```

---

## 13.9 Running Full Test Suite

### Real-World Use Case (The "Cookbook")

**Scenario:** Pre-deployment validation.

**Step 1: Run full test suite**
```bash
pytest tests/ -v --tb=short
```

**Step 2: Run with coverage**
```bash
pytest tests/ --cov=core --cov-report=term-missing
```

**Step 3: Run regression specifically**
```bash
pytest tests/test_regression.py -v
```

**Step 4: Run risk tests**
```bash
pytest tests/test_risk*.py tests/test_phoenix*.py -v
```

**Expected Outcome:**
```
========================= test session starts ==========================
collected 125 items

tests/test_bollinger_squeeze.py ....                               [  3%]
tests/test_dynamic_kelly.py .......                                [  8%]
tests/test_funding_counter.py ......                               [ 13%]
tests/test_phoenix_protocol.py ............                        [ 22%]
tests/test_position_sizer.py ...............                       [ 34%]
tests/test_regression.py .                                         [ 35%]
tests/test_risk_manager.py ...........                             [ 44%]
...
========================= 125 passed in 12.34s =========================
```

---

## 13.10 Writing New Tests

### 1. Test File Naming

```
tests/test_{feature_name}.py
```

### 2. Test Structure Template

```python
"""
Tests for [Feature Name].

Description of what this test file covers.
"""
import pytest
import pandas as pd
from core.your_module import YourClass, YourConfig


class TestYourFeatureBasic:
    """Basic functionality tests."""
    
    def test_feature_does_x(self):
        """[Feature] should [expected behavior]."""
        # Arrange
        config = YourConfig(param=value)
        obj = YourClass(config)
        
        # Act
        result = obj.do_something(input)
        
        # Assert
        assert result == expected


class TestYourFeatureEdgeCases:
    """Edge case and boundary tests."""
    
    def test_feature_handles_zero(self):
        """[Feature] should handle zero input gracefully."""
        ...
    
    def test_feature_handles_nan(self):
        """[Feature] should fallback on NaN input."""
        ...


class TestYourFeatureIntegration:
    """Integration tests with other components."""
    ...
```

### 3. Best Practices

1. **One assertion per test (ideally)**
2. **Descriptive test names:** `test_phoenix_not_triggered_before_cooldown`
3. **Arrange-Act-Assert pattern**
4. **Use fixtures for common setup**
5. **Test edge cases explicitly**
6. **Mock external dependencies**

---

## 13.11 Troubleshooting Tests

### Common Issues

**Import errors:**
```
ModuleNotFoundError: No module named 'core'
```
**Fix:** Run from project root or set PYTHONPATH:
```bash
PYTHONPATH=. pytest tests/ -v
```

**Flaky tests:**
```
FAILED tests/test_something.py::test_timing_dependent
```
**Cause:** Time-dependent test logic.
**Fix:** Mock `time.time()` or use deterministic delays.

**Missing fixtures:**
```
fixture 'mock_adapter' not found
```
**Fix:** Define fixture in `conftest.py` or same file.

---

*Previous Chapter: [Chapter 12: Utility Tools](./CHAPTER_12_UTILITY_TOOLS.md)*  
*Next Chapter: [Appendix A: Configuration Reference](./APPENDIX_A_CONFIG_REFERENCE.md)*
