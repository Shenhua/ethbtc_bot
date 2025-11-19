# ETHBTC Accumulation Bot — Updated Manual (v2)

_Last updated: 2025‑11‑12_

This manual reflects the **current** state of your app after the recent changes we made together: a clearer **ASCII signal meter**, a **human‑readable /status server**, richer **Prometheus metrics**, strict **spot long‑only execution**, and docker fixes.

---

## 0) TL;DR Quick Start

```bash
# 1) Install
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) Set keys (spot; use testnet base URL for test)
export BINANCE_API_KEY=...
export BINANCE_API_SECRET=...

# 3) Run live executor (testnet)
python live_executor.py   --params configs/selected_params_15m_final.json   --symbol ETHBTC   --mode testnet

# 4) Watch
open http://localhost:9109/metrics    # Prometheus metrics
open http://localhost:9110/status     # Human JSON snapshot
```

**Docker** (after fixing `entrypoint.sh` perms and compose):  
```bash
docker compose build --no-cache
docker compose up --force-recreate
```

---

## 1) Architecture & Flow

The app has two main layers:

**A) Strategy Engine (“bot”)**  
- Computes the **ratio** signal and compares it to **four band edges**:
  - `-entry < -exit < 0 < +exit < +entry`
  - Enter BUY when ratio ≤ `-entry`, exit when ratio ≥ `-exit`
  - Enter SELL when ratio ≥ `+entry`, exit when ratio ≤ `+exit`
- Turns that into a **target ETH weight** (`target_w ∈ [0,1]` for spot long‑only).

**B) Live Executor (`live_executor.py`)**  
- Polls new bars, **recomputes the same target** with current config/inputs.
- Applies **step‑toward‑target** (`step_allocation`) and **rebalance threshold**.
- Enforces **spot long‑only**: clamps `target_w` and `new_w` to `[0,1]` (no short).
- Checks **balances**, **min‑notional**, and **filters**.
- Places market order via **Binance Spot API** (testnet or main) when eligible.
- Exposes **Prometheus** on `:9109/metrics` and a **human JSON snapshot** on `:9110/status`.

> **Important:** The executor is **not** “doing its own math”—it recomputes the strategy decision **exactly** as in backtests, then applies execution constraints (step, thresholds, spot‑only clamps, balances, min‑notional).

---

## 2) What’s New vs. v1

- **Clearer ASCII meter** (external module `ascii_levelbar.py`):
  - Landmarks: `B=−entry, b=−exit, 0=zero, s=+exit, S=+entry, |=current`.
  - `dist→BUY/SELL` bps printed each bar.
- **Prometheus metrics extended**:
  - `signal_ratio`, `dist_to_buy_bps`, `dist_to_sell_bps`,
  - `gate_state{open|closed}`, `signal_zone{buy_band|sell_band|neutral}`,
  - `trade_decision{exec_buy|exec_sell|skip_*}`, `delta_w`, `delta_eth`,
  - `balance_free{asset="btc|eth"}`, `wealth_btc_total`, `price_mid`, `spread_bps`.
- **/status JSON server** (port `9110`):
  - Single snapshot per bar with price, wealth, zone/gate, plan deltas, and the ASCII meter string.
- **Spot long‑only execution**:
  - `target_w` and `new_w` are clamped to `[0,1]` (no short).  
  - SELL requires **ETH balance**; BUY requires **BTC balance**.
- **Skip reasons labeled**:
  - `skip_threshold`, `skip_delta_zero`, `skip_balance`, `skip_min_notional`, `skip_gate_closed`.
- **Docker reliability**:
  - Compose `version:` removed (deprecated).  
  - `entrypoint.sh` is baked into image, `chmod +x`, CRLF stripped.
- **Binance connector naming**:
  - Use `BinanceSpotAdapter` and the Spot API methods appropriate to your installed `binance-connector` version.

---

## 3) Installation & Setup

### 3.1 Prereqs
- Python 3.11+ recommended.
- `pip install -r requirements.txt`
- Binance API keys (spot). For **testnet**, same keys but different base URL (see 6.5).

### 3.2 Environment
```bash
export BINANCE_API_KEY=...         # required
export BINANCE_API_SECRET=...      # required
export METRICS_PORT=9109           # optional
export STATUS_PORT=9110            # optional
export STATE_FILE=./state.json     # optional (default inside docker is /data/state.json)
```

### 3.3 File Layout (key parts)
```
core/
  binance_adapter.py          # Spot adapter
  exchange_adapter.py         # Rounding helpers, filters
  metrics.py                  # Prometheus + (now) /status server
ascii_levelbar.py             # New ASCII meter module
ethbtc_accum_bot.py           # Bot CLI (backtest/optimize/selftest)
optimizer_cli.py              # Alternative optimizer CLI
wf_pick.py                    # Post‑processing: pick robust parameter families
live_executor.py              # Live trading loop (testnet/main/dry)
configs/
  selected_params_15m_final.json     # Sample config (bands/step/threshold/etc)
dockerfile
docker-compose.yml
entrypoint.sh
```

---

## 4) Data & Backtests

### 4.1 Historical data
You can use your existing CSVs (e.g., `vision_data/ETHBTC_15m_....csv`) or your own downloader. The backtest CLI expects timestamped OHLCV with a close column.

### 4.2 Backtest CLI
```bash
python ethbtc_accum_bot.py backtest   --data vision_data/ETHBTC_15m_2021-2025_vision.csv   --params configs/selected_params_15m_final.json   --start 2023-01-01 --end 2025-10-31
```

**Tip:** If your `ethbtc_accum_bot.py` uses subcommands, make sure the arguments match (`--params` vs `--config`, etc.). Run `-h` if in doubt.

### 4.3 Optimize (grid/study)
Two options (depending on which CLI you want to use):

**A) Built‑in optimizer**
```bash
python ethbtc_accum_bot.py optimize   --data vision_data/ETHBTC_15m_2021-2025_vision.csv   --start 2020-01-01 --end 2024-06-30   --val 2024-07-01:2024-12-31   --study out/opt_2020-2024__val_2024H2.csv
```

**B) `optimizer_cli.py`**
```bash
python optimizer_cli.py   --data vision_data/ETHBTC_15m_2021-2025_vision.csv   --train 2020-01-01:2024-06-30   --val   2024-07-01:2024-12-31   --out   opt_2020-2024__val_2024H2.csv
# optional: if the CLI supports parallelism
# --jobs 8    (or similar)
```

> If your optimizer lacks a `--jobs` flag, you can shard searches into multiple processes with different random seeds or parameter ranges.

### 4.4 Robust param selection (`wf_pick.py`)
Pick diverse “families” that perform consistently:

```bash
python wf_pick.py   --runs opt_2020-2024__val_2024H2.csv   --q-fees 0.75 --q-turnover 0.75   --penalty-fees 1.0 --penalty-turnover 0.5   --band-round 3 --step-round 1   --lb-bucket 100 --cd-bucket 120   --min-occurs 1 --require-var   --top-k 12   --emit-config configs/selected_params_15m_longonly.json   --out-csv wf_ranked_families.csv   --excel-out wf_ranked_families.xlsx
```

This will write a ready‑to‑trade config JSON with long‑only constraints (bands, step, thresholds, cooldown if used).

---

## 5) Config JSON (key fields)

A typical config (`configs/selected_params_15m_final.json`) contains:

```jsonc
{
  "strategy": {
    "entry_bps": 160,                  // +entry (and -entry) in bps of ratio
    "exit_bps":  91,                   // +exit (and -exit) in bps
    "step_allocation": 1.0,            // fraction of (target - current) per bar
    "vol_scaled_step": false,          // if true, step scales with |z| = ratio/rv
    "rebalance_threshold_w": 0.0010,   // min |Δw| to act
    "cooldown_sec": 0,                 // optional: min seconds between execs
    "trend_kind": "none"               // optional: if you gate by trend
  },
  "execution": {
    "poll_sec": 2.0,                   // loop sleep
    "min_notional_btc": 0.0008,        // lower bound to avoid dust trades
    "slippage_bps": 0                   // accounting (optional)
  }
}
```

**Spot long‑only guard:** the executor clamps `target_w` & `new_w` to `[0,1]`. If you backtest shorting, re‑optimize with long‑only assumptions.

---

## 6) Live Trading

### 6.1 Run modes
```bash
# Dry: compute & log, never hit the exchange
python live_executor.py --params configs/selected_params_15m_final.json --symbol ETHBTC --mode dry

# Testnet (Binance Spot testnet)
python live_executor.py --params configs/selected_params_15m_final.json --symbol ETHBTC --mode testnet

# Main
python live_executor.py --params configs/selected_params_15m_final.json --symbol ETHBTC --mode main
```

### 6.2 What you’ll see
- **Logs** each bar: `[SIG]` (ASCII meter + distances), then `[STATUS]` (human summary).
- **Prometheus** at `http://localhost:9109/metrics`
- **JSON snapshot** at `http://localhost:9110/status`

### 6.3 Skip reasons
- `skip_delta_zero` — `target_w == cur_w` (no-op)
- `skip_threshold` — `|Δw| < rebalance_threshold_w`
- `skip_balance` — not enough BTC (for BUY) or ETH (for SELL)
- `skip_min_notional` — order would be below exchange min notional
- `skip_gate_closed` — your external gate or cooldown blocked it

### 6.4 Testnet seeding (helper)
If you want to force SELL/BUY activity in testnet, you can seed tiny positions. Example helper script:

```python
# seed_testnet_order.py (uses binance-connector Spot)
from binance.spot import Spot
import os, argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="ETHBTC")
    ap.add_argument("--side", choices=["BUY","SELL"], required=True)
    ap.add_argument("--qty", type=float, required=True)
    ap.add_argument("--base-url", default="https://testnet.binance.vision")
    args = ap.parse_args()

    api_key = os.getenv("BINANCE_API_KEY"); api_secret = os.getenv("BINANCE_API_SECRET")
    if not api_key or not api_secret:
        raise RuntimeError("Set BINANCE_API_KEY and BINANCE_API_SECRET")

    c = Spot(api_key=api_key, api_secret=api_secret, base_url=args.base_url)
    print(c.new_order(symbol=args.symbol, side=args.side, type="MARKET", quantity=args.qty))

if __name__ == "__main__":
    main()
```

### 6.5 Binance adapter notes
- Use the **Spot** endpoints that exist in your installed `binance-connector` version (we use `.new_order`, and for the book either `book_ticker` or `ticker_book_ticker` depending on library version). Your `core/binance_adapter.py` already abstracts this.
- Testnet base URL: `https://testnet.binance.vision` (Spot testnet).

### 6.6 State file
- Path defaults to `/data/state.json` in docker. On bare metal, you can set:
  ```bash
  --state ./state.json
  ```
- The executor writes the last seen bar and last target to resume smoothly.

---

## 7) Reading the New ASCII Meter

Example:
```
[----B--------b----------0-----|----s--------S----]
```

- `B=−entry`, `b=−exit`, `0=zero`, `s=+exit`, `S=+entry`, `|=current`
- Left = BUY side; Right = SELL side.
- Distances printed next to it:
  - `dist→BUY` = bps until you reach **buy entry** (0 if already inside)
  - `dist→SELL` = bps until you reach **sell entry** (0 if already inside)

If your terminal is a TTY, the whole bar turns **green** in BUY zone and **yellow** in SELL zone.

---

## 8) Docker

**Dockerfile** key bits:
```dockerfile
COPY entrypoint.sh /entrypoint.sh
RUN sed -i 's/\r$//' /entrypoint.sh && chmod 0755 /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
# CMD [...]  # optional
```

**docker-compose.yml** (no `version:` key):
```yaml
services:
  bot:
    build: .
    image: ethbtc-bot:latest
    environment:
      BINANCE_API_KEY: ${BINANCE_API_KEY}
      BINANCE_API_SECRET: ${BINANCE_API_SECRET}
      METRICS_PORT: "9109"
      STATUS_PORT: "9110"
      STATE_FILE: "/data/state.json"
    volumes:
      - ./configs:/app/configs:ro
      - ./data:/data
    ports:
      - "9109:9109"
      - "9110:9110"
    # command: ["python","/app/live_executor.py","--params","configs/selected_params_15m_final.json","--symbol","ETHBTC","--mode","testnet"]
```

Rebuild & run:
```bash
docker compose build --no-cache
docker compose up --force-recreate
```

---

## 9) Troubleshooting (most common)

- **`Permission denied: /entrypoint.sh`** → ensure `chmod +x`, strip CRLF, don’t bind‑mount over it.
- **`AttributeError: 'Spot' object has no attribute ...`** → mismatch with `binance-connector` method names. Use the adapter functions we added; keep `requirements.txt` pinned.
- **`ValueError: counter metric is missing label values`** → only call `.inc()` on **unlabelled** counters, and use `.labels(reason).inc()` for labelled ones (we fixed these in code).
- **`Read-only file system: '/data'`** → set `--state ./state.json` locally, or mount a writeable volume in docker (`./data:/data`).

---

## 10) Strategy Changes & Re‑Optimization for Long‑Only

Your older parameter sets assumed the ability to **short**. With spot long‑only clamps, re‑optimize:

1) **Run optimizer** on the **same data** but with the executor’s constraints (no short, clamp to `[0,1]`).  
2) Use `wf_pick.py` to select stable families (see §4.4).  
3) Validate out‑of‑sample (recent months) and only then deploy.

Consider:
- Slightly **lower entries** and **higher exits** often work better in long‑only.
- Use a realistic **rebalance threshold** and **min_notional** to avoid dust churn.
- If you enable `vol_scaled_step`, keep a **min step** (e.g. 0.1) to avoid freezing.

---

## 11) UX Add‑Ons (optional)

- Add `/hud` plaintext endpoint combining the first line summary + meter + balances.
- Grafana dashboard using the new metrics (ratio, distances, skips by reason, orders, balances, pnl).

---

## 12) Reference

### 12.1 Live Executor CLI
```
usage: live_executor.py --params CONFIG --symbol SYMBOL --mode {dry,testnet,main} [--state STATE_FILE]
env:
  BINANCE_API_KEY, BINANCE_API_SECRET
  METRICS_PORT (default 9109)
  STATUS_PORT  (default 9110)
  STATE_FILE   (default /data/state.json in docker)
```

### 12.2 Prometheus Metrics (non‑exhaustive)
- `signal_ratio`
- `dist_to_buy_bps`, `dist_to_sell_bps`
- `gate_state{open|closed}`, `signal_zone{buy_band|sell_band|neutral}`
- `trade_decision{exec_buy|exec_sell|skip_*}`
- `delta_w`, `delta_eth`
- `price_mid`, `spread_bps`
- `wealth_btc_total`, `balance_free{asset=...}`
- `orders_submitted_total`, `fills_total`, `rejections_total{reason}`
- `bar_latency_seconds`

### 12.3 ASCII Module
```python
from ascii_levelbar import ascii_level_bar, dist_to_buy_sell_bps
```

---

## 13) Changelog (recent)

- Externalized & redesigned ASCII meter (B, b, 0, s, S + distances).
- Added /status JSON server returning last bar snapshot.
- Standardized skip reasons & metrics.
- Spot long‑only enforcement and balance‑aware execution.
- Docker/compose fixes to entrypoint and deprecations.
- Adapter cleanups for binance-connector compatibility.
