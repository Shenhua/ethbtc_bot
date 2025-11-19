# ETH/BTC Accumulation Bot — Manual (v3, updated after enhancements)

This v3 manual consolidates all changes we implemented during setup and testing:
- **Metrics** via Prometheus on `:9109/metrics` (with optional no‑op fallback).
- **ASCII signal bar** in logs (ratio vs bands, gate status).
- **“Print once per closed bar”** fix for logs.
- **`spread_bps` updated every bar** (optional snippet below).
- **Binance adapter compatibility** for `bookTicker` across connector versions.
- **Guarded taker fallback**, **circuit breaker hooks** (daily loss / drawdown), **interval parameterization**, and **healthcheck** script.
- Updated **Docker Compose** with healthcheck and metrics.

> If you're following the earlier v1 manual, treat this as the canonical, corrected version.

---

## 0) Environment & Dependencies

Create / use a virtualenv and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate               # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install binance-connector pandas numpy pydantic prometheus-client
# optional (research): optuna
```

**requirements.txt additions** (if you keep one):
```
binance-connector>=3
pandas>=1.5
numpy>=1.24
pydantic>=1.10
prometheus-client>=0.17
# optuna>=3
```

> **Security:** never commit real keys. Rotate any keys that were ever committed; lock real keys by **IP** and **least privilege**.

---

## 1) Config shapes accepted

The runner accepts either your **legacy flat JSON** or the **nested** schema with sections `fees`, `strategy`, `execution`, `risk`. It coerces legacy fields automatically.

Example (`configs/selected_params_15m_final.json` excerpt):
```json
{
  "fees": {
    "maker_fee": 0.0002, "taker_fee": 0.0004, "slippage_bps": 1.0,
    "bnb_discount": 0.25, "pay_fees_in_bnb": true
  },
  "strategy": {
    "trend_kind": "roc", "trend_lookback": 120,
    "flip_band_entry": 0.0295, "flip_band_exit": 0.0154,
    "vol_window": 45, "vol_adapt_k": 0.0025,
    "target_vol": 0.3, "cooldown_minutes": 240,
    "step_allocation": 0.33, "max_position": 0.8,
    "gate_window_days": 30, "gate_roc_threshold": 0.02,
    "rebalance_threshold_w": 0.03,
    "vol_scaled_step": true, "profit_lock_dd": 0.02
  },
  "execution": {
    "interval": "15m", "poll_sec": 5, "ttl_sec": 30,
    "taker_fallback": true, "max_taker_btc": 0.002, "max_spread_bps_for_taker": 2.0,
    "min_trade_frac": 0.0015, "min_trade_floor_btc": 0.0002, "min_trade_cap_btc": 0.002,
    "min_trade_btc": null
  },
  "risk": { "basis_btc": 0.16, "max_daily_loss_btc": 0.0, "max_dd_btc": 0.0 }
}
```

---

## 2) Running the bot (dry → testnet → mainnet)

**Dry run (fast validation):**
```bash
python live_executor.py   --params configs/selected_params_15m_final.json   --symbol ETHBTC --mode dry
```

**Quick iteration (1‑minute bars):**
Create `configs/quick_1m.json` identical to your main config but with `"execution.interval": "1m"` to get one bar per minute in testing.

**Testnet:**
```bash
export BINANCE_BASE_URL=https://testnet.binance.vision
export MODE=testnet
export BINANCE_KEY=__TESTNET_KEY__
export BINANCE_SECRET=__TESTNET_SECRET__

STATE_FILE=state_testnet.json TRADES_CSV=trades_testnet.csv python live_executor.py --params configs/selected_params_15m_final.json   --symbol ETHBTC --mode testnet
```

**Mainnet:**
- Remove `BINANCE_BASE_URL`
- Set `MODE=live`
- Start conservatively: longer TTL, consider disabling taker fallback at first.

---

## 3) Metrics and health

- **Prometheus** is exposed at `http://localhost:9109/metrics`.
  Key series: `bar_latency_seconds`, `orders_submitted_total`, `fills_total`,
  `rejections_total`, `pnl_btc`, `exposure_eth_weight`, `spread_bps`.

- **Healthcheck** (optional, used by Docker):
  `scripts/healthcheck.py` returns healthy when your `STATE_FILE` is fresh for the expected bar interval.

> If you prefer the bot to run **without** Prometheus installed, use the no‑op fallback version of `core/metrics.py` (provided earlier) so missing deps don’t crash the process.

---

## 4) ASCII signal bar (ratio vs bands + gate)

We reintroduced a small ASCII line to help you see signal proximity:

```
[SIG] ratio=+0.0123  bands: -exit=0.0154  +entry=0.0295  gate=OPEN   [----|-----.........................]
```

If you haven’t applied it yet, use the patch file I provided (`patch_ascii_visual.diff`) **or** add the two blocks manually:
- Helper under imports:
  ```python
  def level_bar(x: float, lo: float, hi: float, width: int = 41) -> str:
      if hi <= lo: lo, hi = -1.0, 1.0
      x_clamped = min(max(x, lo), hi)
      pos = int(round((x_clamped - lo) / (hi - lo) * (width - 1)))
      chars = ["-"] * width; chars[pos] = "|"
      return "[" + "".join(chars) + "]"
  ```
- Log line right after `gate_ok` is computed:
  ```python
  bar_str = level_bar(cur_ratio, -exitb, entry, width=41)
  log.info("[SIG] ratio=%+0.4f  bands: -exit=%0.4f  +entry=%0.4f  gate=%s   %s",
           cur_ratio, exitb, entry, ("OPEN" if gate_ok else "CLOSED"), bar_str)
  ```

**Emit once per closed bar** (important):
Wherever you mark the bar as processed (we save `state["last_bar_close"] = bar_ts`), also set:
```python
last_seen_bar = bar_ts
```
so the `[SIG]` line prints **once** per bar instead of once per poll.

**Update `spread_bps` every bar** (optional but recommended):
Right after you compute `price` add:
```python
book = adapter.get_book(args.symbol)
SPREAD_BPS.set(1e4 * (book.best_ask - book.best_bid) / max(price, 1e-12))
```

---

## 5) Binance adapter compatibility (bookTicker)

Different `binance-connector` versions expose the best‑bid/ask endpoint under different names.
The adapter was updated to try, in order:

1. `self.client.ticker_book_ticker(symbol=...)`
2. `self.client.book_ticker(symbol=...)`
3. Fallback to `self.client.depth(symbol=..., limit=5)` and take top levels.

If you prefer not to modify code, you can also simply:
```bash
python -m pip install --upgrade binance-connector
```

---

## 6) Docker Compose (new)

A **single‑service** compose that builds your local repo, exposes metrics, and uses the healthcheck:

```yaml
version: "3.9"
services:
  bot:
    build:
      context: .
      dockerfile: Dockerfile        # Make sure the file name is capitalized on Linux
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

> **Note:** Put `scripts/healthcheck.py` in your repo (same one we used during setup). On Linux/macOS (case‑sensitive FS), ensure the build file is named **`Dockerfile`**, not `dockerfile`.

---

## 7) Ops checklist

- Use **distinct** `STATE_FILE`/`TRADES_CSV` per environment (dry/testnet/live).  
- Never run two instances on the **same** state/log.  
- Rotate any committed keys; store real keys only in env or a secret manager.  
- Start testnet with relaxed bands and disabled gate if you just need to validate execution path; revert before mainnet.
- Monitor `/metrics`: watch `orders_submitted_total`, `fills_total`, `rejections_total`, `bar_latency_seconds` and your exposure gauges.

---

## 8) Common workflows (quick copy/paste)

**Backtest (verify a JSON):**
```bash
python ethbtc_accum_bot.py backtest   --data data/ETHBTC_15m.csv --bnb-data data/BNBBTC_15m.csv   --basis-btc 0.16 --config configs/selected_params_15m_final.json   --start 2024-07-01 --end 2025-06-30   --excel-out out/final_backtest.xlsx
```

**Optimizer (robust CLI):**
```bash
python optimizer_cli.py   --data data/ETHBTC_15m.csv --bnb-data data/BNBBTC_15m.csv   --train-start 2021-01-01 --train-end 2024-06-30   --test-start  2024-07-01  --test-end  2025-06-30   --n-random 600 --threads 0 --chunk-size 32   --lambda-turns 2.0 --gap-penalty 0.35 --turns-scale 800   --lambda-fees 2.0 --lambda-turnover 1.0   --out out/opt_runs_seed1.csv --excel-out out/opt_runs_seed1.xlsx   --emit-config out/selected_params_seed1.json
```

**Walk‑forward family picker:**
```bash
python wf_pick.py   --runs out/opt_runs_seed1.csv out/opt_runs_seed2.csv out/opt_runs_seed3.csv   --q-fees 0.75 --q-turnover 0.75   --penalty-turns 0.0 --penalty-fees 1.0 --penalty-turnover 0.5   --lb-bucket 40 --cd-bucket 60 --band-round 4 --step-round 2   --top-k 12 --min-occurs 2 --require-costs --require-var   --emit-config out/selected_params_15m_final.json   --out-csv out/wf_ranked_families.csv --excel-out out/wf_summary.xlsx
```

**Live (dry → testnet → mainnet):**
```bash
# dry
python live_executor.py --params configs/selected_params_15m_final.json --mode dry -v

# testnet
BINANCE_BASE_URL=https://testnet.binance.vision MODE=testnet python live_executor.py --params configs/selected_params_15m_final.json --mode testnet -vv

# mainnet
MODE=live python live_executor.py --params configs/selected_params_15m_final.json --mode live -v
```

---

## 9) Appendix — Tiny snippets you might want to keep

**No‑op Prometheus fallback (if library missing):**
```python
# core/metrics.py (fallback)
try:
    from prometheus_client import Counter, Gauge, Summary, start_http_server
except Exception:
    class _Noop:
        def labels(self,*a,**k): return self
        def inc(self,*a,**k): pass
        def set(self,*a,**k): pass
        def time(self):
            class _C: 
                def __enter__(self): return None
                def __exit__(self,*a): return False
            return _C()
    def Counter(*a,**k): return _Noop()
    def Gauge(*a,**k): return _Noop()
    def Summary(*a,**k): return _Noop()
    def start_http_server(*a,**k): pass
```

**Book ticker compatibility inside `core/binance_adapter.py`:**
```python
def get_book(self, symbol: str) -> Book:
    try:
        t = self.client.ticker_book_ticker(symbol=symbol)
        return Book(best_bid=float(t["bidPrice"]), best_ask=float(t["askPrice"]))
    except AttributeError:
        try:
            t = self.client.book_ticker(symbol=symbol)
            return Book(best_bid=float(t["bidPrice"]), best_ask=float(t["askPrice"]))
        except AttributeError:
            d = self.client.depth(symbol=symbol, limit=5)
            return Book(best_bid=float(d["bids"][0][0]), best_ask=float(d["asks"][0][0]))
```

---

**End of v3 manual.**
