# ETH/BTC Accumulation Bot — Manual (v4)

This v4 manual folds in the latest changes and fixes we applied together:
- **Prometheus metrics** on `:9109/metrics` (with safe no‑op fallback option).
- **ASCII signal bar** (ratio vs bands, gate status) in logs.
- **Idempotent logs**: one `[SIG]` per *closed bar* (not per poll).
- **Per‑bar `spread_bps`** update (optional snippet).
- **Binance adapter compatibility** for different `binance-connector` versions — fixes the
  `AttributeError: 'Spot' object has no attribute 'ticker_book_ticker'` you saw in Docker.
- **Guarded taker fallback**, **circuit‑breaker hooks**, **interval parameterization**, **healthcheck**.
- Updated **Docker Compose** and a **dev override** that mounts your source into the container so you can iterate without rebuilding the image.

> Treat this v4 as the canonical manual going forward.

---

## 0) Environment & dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install binance-connector pandas numpy pydantic prometheus-client
# optional research: python -m pip install optuna
```
`requirements.txt`:
```
binance-connector==3.12.0
python-dateutil>=2.8.2
python-dotenv>=1.0
requests>=2.32
pandas>=2.2
numpy>=2.0
pydantic>=1.10
prometheus-client>=0.17
optuna>=3
```

---

## 1) Config shapes accepted

The live runner accepts either:
- your **legacy flat JSON**, or
- the **nested** schema (`fees/strategy/execution/risk`).

It will coerce a legacy flat JSON to the nested model internally.

---

## 2) Running (dry → testnet → mainnet)

**Dry:**
```bash
python live_executor.py --params configs/selected_params_15m_final.json --symbol ETHBTC --mode dry
```

**1‑minute quick test:**
Use a copy of your config with `"execution.interval": "1m"` to see one bar per minute.

**Testnet:**
```bash
export BINANCE_BASE_URL=https://testnet.binance.vision
export MODE=testnet
export BINANCE_KEY=__TESTNET_KEY__
export BINANCE_SECRET=__TESTNET_SECRET__

STATE_FILE=state_testnet.json TRADES_CSV=trades_testnet.csv python live_executor.py --params configs/selected_params_15m_final.json --symbol ETHBTC --mode testnet
```

**Mainnet:** remove `BINANCE_BASE_URL`, set `MODE=live`, start conservatively.

---

## 3) Metrics & health

- **Prometheus**: `http://localhost:9109/metrics`  
  Custom series: `bar_latency_seconds`, `orders_submitted_total`, `fills_total`, `rejections_total`, `pnl_btc`, `exposure_eth_weight`, `spread_bps`.

- **Healthcheck**: `scripts/healthcheck.py` returns healthy when `STATE_FILE` is fresh for the interval.

If you don’t want Prometheus installed, use the **no‑op fallback** version of `core/metrics.py` so missing deps won’t crash the process.

---

## 4) ASCII signal bar & log idempotency

**ASCII bar** (ratio vs bands, gate):
```
[SIG] ratio=+0.0123  bands: -exit=0.0154  +entry=0.0295  gate=OPEN   [----|-----.........................]
```

If you need to add it manually, drop in:
```python
def level_bar(x: float, lo: float, hi: float, width: int = 41) -> str:
    if hi <= lo: lo, hi = -1.0, 1.0
    x_clamped = min(max(x, lo), hi)
    pos = int(round((x_clamped - lo) / (hi - lo) * (width - 1)))
    chars = ["-"] * width; chars[pos] = "|"
    return "[" + "".join(chars) + "]"

# after computing gate_ok and before target_w
bar_str = level_bar(cur_ratio, -exitb, entry, width=41)
log.info("[SIG] ratio=%+0.4f  bands: -exit=%0.4f  +entry=%0.4f  gate=%s   %s",
         cur_ratio, exitb, entry, ("OPEN" if gate_ok else "CLOSED"), bar_str)
```

**Print once per closed bar** (avoid repeating per poll): whenever you mark a bar processed, add:
```python
state["last_bar_close"] = bar_ts
save_state(args.state, state)
last_seen_bar = bar_ts   # important
```
(Do this in the circuit‑breaker pause, “below rebalance threshold”, “below min notional”, and end‑of‑bar sections.)

**Update `spread_bps` every bar** (optional, recommended):
```python
# right after: price = float(df["close"].iloc[-1])
book = adapter.get_book(args.symbol)
SPREAD_BPS.set(1e4 * (book.best_ask - book.best_bid) / max(price, 1e-12))
```

---

## 5) Binance adapter compatibility (fix for Docker error)

Some `binance-connector` versions expose the endpoint as **`book_ticker`** instead of **`ticker_book_ticker`**.
To be robust across versions, the adapter now tries **both**, then falls back to a tiny depth request:

```python
def get_book(self, symbol: str) -> Book:
    # Try /api/v3/ticker/bookTicker under either name
    try:
        t = self.client.ticker_book_ticker(symbol=symbol)  # preferred on some versions
        return Book(best_bid=float(t["bidPrice"]), best_ask=float(t["askPrice"]))
    except AttributeError:
        try:
            t = self.client.book_ticker(symbol=symbol)
            return Book(best_bid=float(t["bidPrice"]), best_ask=float(t["askPrice"]))
        except AttributeError:
            d = self.client.depth(symbol=symbol, limit=5)
            return Book(best_bid=float(d["bids"][0][0]), best_ask=float(d["asks"][0][0]))
```

**You have two ways to apply this fix:**

1) **Drop‑in file replacement** (easiest): replace `core/binance_adapter.py` with the version I attached below.  
2) **Patch** (if you prefer unified diffs): apply `patch_binance_adapter_book.diff` I attached.

**Important for Docker:** after you change code, **rebuild without cache** so the new Python files are baked into the image:
```bash
docker compose down
docker compose build --no-cache
docker compose up -d
docker logs -f bot
```

### Dev‑mode alternative (no rebuild)
Use the `docker-compose.dev.yml` I attached to **mount your local source** into `/app`, so code changes are reflected immediately in the container:
```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d
```
Now editing `core/binance_adapter.py` locally takes effect instantly inside the container.

---

## 6) Docker Compose

**Production‑style** compose (bakes code into the image; rebuild after changes):
```yaml
version: "3.9"
services:
  bot:
    build:
      context: .
      dockerfile: Dockerfile
    image: ethbtc-bot:latest
    environment:
      MODE: ${MODE:-dry}            # dry | testnet | live
      SYMBOL: ${SYMBOL:-ETHBTC}
      BINANCE_KEY: ${BINANCE_KEY}
      BINANCE_SECRET: ${BINANCE_SECRET}
      BINANCE_BASE_URL: ${BINANCE_BASE_URL:-}
      INTERVAL: ${INTERVAL:-15m}
      STATE_FILE: /data/state.json
      TRADES_CSV: /data/trades_log.csv
      METRICS_PORT: 9109
      LOGLEVEL: ${LOGLEVEL:-INFO}
    command: ["python","live_executor.py",
              "--params","configs/selected_params_15m_final.json",
              "--mode","${MODE:-dry}","--symbol","${SYMBOL:-ETHBTC}"]
    volumes:
      - ./data:/data
    ports:
      - "9109:9109"
    restart: unless-stopped
    healthcheck:
      test: ["CMD","python","scripts/healthcheck.py"]
      interval: 30s
      timeout: 5s
      retries: 3
    logging:
      driver: json-file
      options:
        max-size: "100m"
        max-file: "5"
```

**Dev override** (mounts your working tree as `/app` so edits are live):
```yaml
# docker-compose.dev.yml
version: "3.9"
services:
  bot:
    volumes:
      - ./:/app
```

> On Linux/macOS (case‑sensitive FS), ensure your build file is named **Dockerfile** (capital D).

---

## 7) Ops checklist

- One runner per `STATE_FILE`/`TRADES_CSV` (avoid two processes on the same files).
- Rotate any keys that were ever committed; store real keys in env/secret manager; restrict by IP/permissions.
- Start testnet with relaxed bands or disabled gate for path testing; revert before mainnet.
- Watch `/metrics` for `orders_submitted_total`, `fills_total`, `rejections_total`, `bar_latency_seconds`, and exposure gauges.

---

## 8) Quick workflows (copy/paste)

**Dry → Testnet → Mainnet** and **Backtest → Optimize → Pick → Verify** are unchanged from v3;
see prior sections for commands. The only new operational step is **rebuild with `--no-cache`**
(or use the dev override) after code changes so Docker doesn’t serve an old cached layer.

---

**End of v4.**
