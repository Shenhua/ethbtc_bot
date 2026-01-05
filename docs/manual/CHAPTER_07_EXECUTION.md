# Chapter 7: Execution Layer

> **Purpose:** This chapter provides exhaustive documentation of the order execution system, covering exchange adapters, order types, retry logic, circuit breakers, and the complete order lifecycle from signal to fill.

---

## 7.1 Execution Architecture

### 1. Concept & "The Why"

* **What it is:** The execution layer translates trading signals into actual exchange orders. It abstracts exchange-specific APIs behind a common interface and provides resilience against API failures.

* **Purpose:** 
  1. Exchange abstraction—same code works for Spot and Futures
  2. Order quality—prefer maker orders (lower fees) with taker fallback
  3. Resilience—automatic retries and circuit breakers prevent catastrophic failures
  4. Observability—detailed logging of all order operations

* **Location:** 
  - Abstract Interface: [`core/exchange_adapter.py`](../../core/exchange_adapter.py)
  - Spot Implementation: [`core/binance_adapter.py`](../../core/binance_adapter.py)
  - Futures Implementation: [`core/futures_adapter.py`](../../core/futures_adapter.py)
  - Resilience: [`core/resilience.py`](../../core/resilience.py)
  - Main Executor: [`live_executor.py`](../../live_executor.py)

### 2. Configuration & Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `exchange_type` | enum | `spot`, `futures` | `spot` | Which adapter to use |
| `interval` | string | 1m–1d | `15m` | Bar interval for trading |
| `poll_sec` | int | 1–300 | 5 | Seconds between order status checks |
| `ttl_sec` | int | 5–600 | 30 | Order timeout before cancel |
| `taker_fallback` | bool | — | false | Use market order if maker fails |
| `max_taker_btc` | float | 0.0–1.0 | 0.002 | Max size for taker fallback |
| `max_spread_bps_for_taker` | float | 0.0–100.0 | 2.0 | Max spread for taker |
| `min_trade_btc` | float | — | null | Minimum trade size |
| `leverage` | int | 1–20 | 1 | Futures leverage |

### 3. Execution Flow Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                        Signal Generated                          │
│                     target_w = 0.5 (50% long)                    │
└──────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Position Sizing Applied                       │
│         new_w = cur_w + step × (target_w - cur_w)                │
│                      delta_qty calculated                        │
└──────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Minimum Trade Check                           │
│              trade_value >= min_trade_btc ?                      │
│                   YES: proceed / NO: skip                        │
└──────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Order Placement                               │
│            1. Place LIMIT_MAKER order at best_bid/ask            │
│            2. Poll for fill (every poll_sec)                     │
│            3. If TTL expires: cancel + taker_fallback?           │
└──────────────────────────────────────────────────────────────────┘
                                  │
                  ┌───────────────┴───────────────┐
                  ▼                               ▼
           ┌──────────┐                    ┌──────────────┐
           │  FILLED  │                    │   TIMEOUT    │
           │  (done)  │                    │  + Fallback  │
           └──────────┘                    └──────────────┘
```

---

## 7.2 Exchange Adapters

### 1. Concept & "The Why"

* **What it is:** Adapters that implement a common `ExchangeAdapter` interface for different Binance markets (Spot vs USDS-M Futures).

* **Purpose:** Allows the executor to work with any exchange without code changes. New exchanges can be added by implementing the interface.

* **Location:**
  - Abstract: [`core/exchange_adapter.py`](../../core/exchange_adapter.py)
  - Spot: [`core/binance_adapter.py`](../../core/binance_adapter.py) → `BinanceSpotAdapter`
  - Futures: [`core/futures_adapter.py`](../../core/futures_adapter.py) → `BinanceFuturesAdapter`

### 2. Common Interface Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `get_klines(symbol, interval, limit)` | Fetch OHLCV candlesticks | `List[Dict]` |
| `get_book(symbol)` | Get best bid/ask | `Book(best_bid, best_ask)` |
| `get_filters(symbol)` | Get exchange limits | `Filters(step_size, tick_size, min_notional)` |
| `get_account_balance(asset)` | Get free balance | `float` |
| `get_funding_rate(symbol)` | Get current funding rate | `float` (as percentage) |
| `place_limit_maker(symbol, side, qty, price)` | Place post-only limit order | `order_id: str` |
| `market_order(symbol, side, qty)` | Place market order | `order_id: str` |
| `cancel(symbol, order_id)` | Cancel order | `None` |
| `cancel_open_orders(symbol)` | Cancel all open orders | `List[str]` (cancelled IDs) |
| `check_order(symbol, order_id)` | Check order status | `Tuple[is_done, filled_qty]` |

### 3. Spot Adapter Specifics

```python
# BinanceSpotAdapter order placement
def place_limit_maker(self, symbol: str, side: str, quantity: float, price: float) -> str:
    """Spot POST_ONLY order using LIMIT_MAKER type."""
    resp = self.client.new_order(
        symbol=symbol,
        side=side,
        type="LIMIT_MAKER",  # ← Post-only, rejects if would take
        quantity=f"{quantity:.8f}",
        price=f"{price:.8f}"
    )
    return str(resp["orderId"])
```

**Key Differences:**
- Uses `LIMIT_MAKER` order type for post-only
- Balance from `account()["balances"]` → `free` field
- No position concept (only balances)

### 4. Futures Adapter Specifics

```python
# BinanceFuturesAdapter order placement
def place_limit_maker(self, symbol: str, side: str, quantity: float, price: float) -> str:
    """Futures Post-Only Order using GTX timeInForce."""
    resp = self.client.new_order(
        symbol=symbol,
        side=side,
        type="LIMIT",
        timeInForce="GTX",  # ← GTX = Post Only on Futures
        quantity=f"{quantity:.8f}",
        price=f"{price:.8f}"
    )
    return str(resp["orderId"])
```

**Key Differences:**
- Uses `timeInForce="GTX"` for post-only (not `LIMIT_MAKER`)
- Has `get_position(symbol)` returning signed position amount
- Balance from `account()["assets"]` → `marginBalance` field
- Supports leverage via `set_leverage(symbol, leverage)`

### 5. Step-by-Step Guide: Selecting Adapter

The adapter is selected based on `exchange_type` in config:

```python
# In live_executor.py
if cfg.execution.exchange_type == "futures":
    from core.futures_adapter import BinanceFuturesAdapter
    from binance.um_futures import UMFutures
    
    client = UMFutures(key=API_KEY, secret=API_SECRET, base_url=BASE_URL)
    adapter = BinanceFuturesAdapter(client)
else:
    from core.binance_adapter import BinanceSpotAdapter
    from binance.spot import Spot
    
    client = Spot(key=API_KEY, secret=API_SECRET, base_url=BASE_URL)
    adapter = BinanceSpotAdapter(client)
```

---

## 7.3 Order Types

### 7.3.1 LIMIT_MAKER (Post-Only)

#### 1. Concept & "The Why"

* **What it is:** A limit order that is rejected if it would immediately execute as a taker (market order). Guarantees maker fee.

* **Purpose:** 
  - Maker fees are typically 50% lower than taker fees (0.02% vs 0.04%)
  - For 1 BTC daily volume, this saves ~200 BTC annually in fees
  - Forces price improvement (order sits in book, not crosses)

* **Location:** `place_limit_maker()` in both adapters

#### 2. Configuration

| Setting | Value | Description |
|---------|-------|-------------|
| Order Type (Spot) | `LIMIT_MAKER` | Binance Spot post-only |
| Order Type (Futures) | `LIMIT` + `GTX` | Binance Futures post-only |

#### 3. Behavior

```python
# Place at best bid (for BUY) or best ask (for SELL)
book = adapter.get_book(symbol)
if side == "BUY":
    price = book.best_bid  # Join the bid
else:
    price = book.best_ask  # Join the ask

order_id = adapter.place_limit_maker(symbol, side, quantity, price)
```

**Hidden Logic:**
- If book moves against you, order won't fill until price returns
- If order would cross (execute immediately), it's REJECTED with error `-1013`

### 7.3.2 MARKET Order (Taker Fallback)

#### 1. Concept & "The Why"

* **What it is:** Market orders execute immediately at best available price. Used as fallback when maker orders timeout.

* **Purpose:** Guarantees execution when fill is critical (e.g., exiting during crash).

* **Location:** `market_order()` in both adapters

#### 2. Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `taker_fallback` | false | Enable market order fallback |
| `max_taker_btc` | 0.002 | Maximum size for taker orders |
| `max_spread_bps_for_taker` | 2.0 | Max spread to allow taker |

#### 3. Fallback Logic

```python
# After maker order times out
if taker_fallback_enabled:
    spread_bps = (book.best_ask - book.best_bid) / book.best_bid * 10000
    
    if spread_bps <= max_spread_bps_for_taker:
        if unfilled_qty <= max_taker_btc:
            adapter.market_order(symbol, side, unfilled_qty)
```

---

## 7.4 Order Lifecycle

### 1. Concept & "The Why"

* **What it is:** The complete lifecycle of an order from placement through fill or cancellation.

* **Purpose:** Understanding the lifecycle helps debug order issues and optimize execution.

* **Location:** [`live_executor.py`](../../live_executor.py)

### 2. Lifecycle States

```
┌─────────┐     place_limit_maker()     ┌─────────┐
│  INIT   │ ─────────────────────────▶  │   NEW   │
└─────────┘                             └─────────┘
                                             │
                            ┌────────────────┼────────────────┐
                            │                │                │
                            ▼                ▼                ▼
                    ┌─────────────┐   ┌──────────────┐   ┌─────────┐
                    │   FILLED    │   │ PART_FILLED  │   │ TIMEOUT │
                    └─────────────┘   └──────────────┘   └─────────┘
                            │                │                │
                            │                │         cancel()
                            │                │                │
                            ▼                ▼                ▼
                    ┌─────────────┐   ┌──────────────┐   ┌─────────┐
                    │    DONE     │   │   CANCELED   │   │ TAKER?  │
                    └─────────────┘   └──────────────┘   └─────────┘
```

### 3. Polling and Timeout

```python
# Order polling loop
t0 = time.time()
while True:
    is_done, filled_qty = adapter.check_order(symbol, order_id)
    
    if is_done:
        return filled_qty  # Success
    
    elapsed = time.time() - t0
    if elapsed >= ttl_sec:
        adapter.cancel(symbol, order_id)
        
        if taker_fallback_enabled:
            # Execute remaining as market
            remaining = original_qty - filled_qty
            adapter.market_order(symbol, side, remaining)
        
        return filled_qty
    
    time.sleep(poll_sec)
```

### 4. Step-by-Step Guide: Order Execution

1. **Signal generates delta:**
   ```
   target_w = 0.5, cur_w = 0.2
   delta_qty = (0.5 - 0.2) × wealth / price = 0.3 × 1.0 / 0.034 = 8.82 ETH
   ```

2. **Filters applied:**
   ```python
   filters = adapter.get_filters("ETHBTC")
   # step_size = 0.001, tick_size = 0.00000001, min_notional = 0.0001
   
   # Round quantity to step_size
   qty_rounded = round(delta_qty / filters.step_size) * filters.step_size
   # qty_rounded = 8.820
   
   # Round price to tick_size
   price_rounded = round(price / filters.tick_size) * filters.tick_size
   ```

3. **Order placed:**
   ```python
   order_id = adapter.place_limit_maker("ETHBTC", "BUY", 8.82, 0.03421)
   # Log: [SPOT] Placing POST_ONLY BUY order: 8.82000000 ETHBTC @ 0.03421000
   ```

4. **Polling loop:**
   ```python
   # Every 5 seconds (poll_sec)
   is_done, filled = adapter.check_order("ETHBTC", order_id)
   # Log: [SPOT] Order 12345678 status: FILLED, filled: 8.82
   ```

---

## 7.5 Exchange Filters

### 1. Concept & "The Why"

* **What it is:** Exchange-imposed constraints on order size and price precision.

* **Purpose:** Orders that violate filters are rejected. Understanding filters prevents order failures.

* **Location:** `get_filters()` in both adapters

### 2. Filter Types

| Filter | Description | Example |
|--------|-------------|---------|
| `step_size` | Minimum quantity increment | 0.001 (must trade in 0.001 multiples) |
| `tick_size` | Minimum price increment | 0.00000001 (8 decimal places) |
| `min_notional` | Minimum order value | 0.0001 BTC (~$10 at $100k BTC) |

### 3. Applying Filters

```python
def apply_filters(quantity: float, price: float, filters: Filters) -> Tuple[float, float]:
    """Round quantity and price to exchange requirements."""
    
    # Round quantity DOWN to step_size
    qty_steps = int(quantity / filters.step_size)
    qty_rounded = qty_steps * filters.step_size
    
    # Round price to tick_size (standard rounding)
    price_rounded = round(price / filters.tick_size) * filters.tick_size
    
    # Check min notional
    notional = qty_rounded * price_rounded
    if notional < filters.min_notional:
        raise ValueError(f"Order value {notional} below minimum {filters.min_notional}")
    
    return qty_rounded, price_rounded
```

### 4. Troubleshooting

* **Error Messages:**

  ```
  APIError(code=-1013): Filter failure: LOT_SIZE
  ```
  **Cause:** Quantity not a multiple of step_size.
  **Fix:** Round quantity: `qty = int(qty / step_size) * step_size`

  ```
  APIError(code=-1013): Filter failure: MIN_NOTIONAL
  ```
  **Cause:** Order value below minimum.
  **Fix:** Increase quantity or set `min_trade_btc` higher in config.

---

## 7.6 Resilience Module

### 1. Concept & "The Why"

* **What it is:** A retry and circuit breaker system that protects against transient API failures.

* **Purpose:** 
  - Network glitches shouldn't cause trading failures
  - API rate limits shouldn't cause cascade failures
  - Exponential backoff prevents hammering troubled APIs

* **Location:** [`core/resilience.py`](../../core/resilience.py)

### 2. Configuration & Parameters

#### Retry Decorator

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_attempts` | 3 | Total attempts before failure |
| `min_wait` | 1.0s | Minimum backoff time |
| `max_wait` | 30.0s | Maximum backoff time |
| `exponential_base` | 2.0 | Backoff multiplier |

**Backoff Formula:**
```
wait_time = min(max_wait, min_wait × (exponential_base ^ (attempt - 1)))

Example with min_wait=1, max_wait=30, base=2:
  Attempt 1 fails → wait 1s
  Attempt 2 fails → wait 2s
  Attempt 3 fails → wait 4s
  Attempt 4 fails → wait 8s
  ...
```

#### Circuit Breaker

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_failures` | 5 | Failures before opening circuit |
| `reset_timeout` | 60.0s | Seconds before trying again |

**States:**
- `CLOSED`: Normal operation, requests pass through
- `OPEN`: Too many failures, requests blocked
- `HALF_OPEN`: Testing if service recovered

### 3. Usage Examples

#### Decorator Usage
```python
from core.resilience import with_retry

@with_retry(max_attempts=3, min_wait=1.0, max_wait=30.0)
def fetch_klines():
    return adapter.get_klines(symbol, interval)

# Automatically retries up to 3 times on failure
klines = fetch_klines()
```

#### Functional Usage
```python
from core.resilience import retry_api_call, CircuitBreaker

breaker = CircuitBreaker(max_failures=5, reset_timeout=60)

result = retry_api_call(
    adapter.get_klines, symbol, interval,
    max_attempts=3,
    circuit_breaker=breaker
)
```

### 4. Circuit Breaker Flow

```
     ┌─────────────────────────────────────────┐
     │              CLOSED                     │
     │  (Normal: requests pass through)        │
     └─────────────────────────────────────────┘
                       │
           5 consecutive failures
                       │
                       ▼
     ┌─────────────────────────────────────────┐
     │               OPEN                      │
     │  (Blocking: requests fail immediately)  │
     │  CircuitBreakerOpen exception raised    │
     └─────────────────────────────────────────┘
                       │
           60 seconds elapsed
                       │
                       ▼
     ┌─────────────────────────────────────────┐
     │            HALF_OPEN                    │
     │  (Testing: try one request)             │
     └─────────────────────────────────────────┘
            │                     │
         Success               Failure
            │                     │
            ▼                     ▼
     ┌──────────┐          ┌──────────┐
     │  CLOSED  │          │   OPEN   │
     └──────────┘          └──────────┘
```

### 5. Troubleshooting

* **Error Messages:**

  ```
  CircuitBreakerOpen: Circuit breaker is OPEN. Will reset in 45.2s
  ```
  **Cause:** 5 consecutive API failures triggered circuit breaker.
  **Fix:** Wait for reset_timeout or investigate underlying API issue.

* **Hidden Logic:**
  - Uses `tenacity` library if available, falls back to simple retry
  - Circuit breaker state is in-memory only (resets on restart)
  - Retry logs appear at WARNING level

---

## 7.7 Minimum Trade Size

### 1. Concept & "The Why"

* **What it is:** A threshold that skips trades below a certain BTC value to avoid dust accumulation and excessive fees.

* **Purpose:**
  - Prevents accumulating many tiny positions
  - Avoids fee overhead on negligible trades
  - Implements "snap-to-zero" to exit near-zero positions cleanly

* **Location:** Configuration in `execution.min_trade_btc` or calculated from `min_trade_frac`

### 2. Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_trade_btc` | null | Absolute minimum (overrides frac) |
| `min_trade_frac` | 0.0015 | Min as fraction of portfolio |
| `min_trade_floor_btc` | 0.0 | Floor for calculated min |
| `min_trade_cap_btc` | 0.0 | Cap for calculated min |

**Calculation:**
```python
if min_trade_btc is set:
    threshold = min_trade_btc
else:
    threshold = wealth × min_trade_frac
    threshold = max(threshold, min_trade_floor_btc)
    threshold = min(threshold, min_trade_cap_btc) if min_trade_cap_btc > 0 else threshold
```

### 3. Anti-Zeno (Snap-to-Zero)

```python
# When target is zero but we hold a tiny position
if target_w == 0.0 and abs(eth_position) > 0:
    position_value = abs(eth_position) * price
    
    # Case 1: Dust cleanup (near minimum)
    if min_trade_btc < position_value < (3 × min_trade_btc):
        new_w = 0.0  # Force full exit
    
    # Case 2: Anti-Zeno (gradient too small)
    implied_delta = (new_w × wealth / price) - eth_position
    if abs(implied_delta × price) < min_trade_btc:
        new_w = 0.0  # Force full exit to avoid infinite approach
```

---

## 7.8 Complete Execution Configuration Example

```json
{
  "execution": {
    "interval": "15m",
    "poll_sec": 5,
    "ttl_sec": 30,
    "taker_fallback": true,
    "max_taker_btc": 0.005,
    "max_spread_bps_for_taker": 3.0,
    "min_trade_frac": 0.0015,
    "min_trade_btc": 0.0001,
    "exchange_type": "futures",
    "leverage": 2
  }
}
```

**This configuration:**
- Trades on 15-minute bars
- Polls order status every 5 seconds
- Cancels unfilled maker orders after 30 seconds
- Falls back to taker if spread ≤ 3 bps and size ≤ 0.005 BTC
- Minimum trade size of 0.0001 BTC
- Uses Futures adapter with 2× leverage

---

*Previous Chapter: [Chapter 6: Risk Management](./CHAPTER_06_RISK_MANAGEMENT.md)*  
*Next Chapter: [Chapter 8: Backtesting Engine](./CHAPTER_08_BACKTESTING.md)*
