# ETH/BTC Accumulation Bot — In‑Depth Analysis & Complete User Manual

**Repo artifact analyzed:** `/mnt/data/ethbtc_bot.zip`  
**Verified key files:** `ethbtc_accum_bot.py`, `optimizer_cli.py`, `wf_pick.py`, `multi_interval_opt.py`, `live_executor.py`, `download_vision.py`, `vision_fix_timestamps.py`, `ascii_levelbar.py`, `requirements.txt`, `docker-compose.yml`, `dockerfile`, `entrypoint.sh`, `selected_params_15m_final.json`, `.env`.

> **What this document is:** a complete, end‑to‑end manual (plus my analysis) for the app, with ready‑to‑run CLI workflows for: data download → backtest → optimization → walk‑forward selection → testnet → mainnet. It reflects the **actual code** in the zip (I verified filenames, flags, and behaviors).

---

## 🔍 Pre‑Analysis Summary (and quick questions)

### What’s in the app
- A research & execution stack for an **ETH/BTC rotation (accumulation)** strategy.
- **Backtesting & optimization**: parameter search (random search, penalties for costs/turns), train/test splits, Excel summaries.
- **Walk‑forward “family” picker** to choose robust parameter buckets that repeat across runs.
- **Live executor** for **dry**, **testnet**, and **mainnet** with **15‑minute bar cadence** and **idempotent per bar** execution (`state.json`). Uses **LIMIT_MAKER** (post‑only) with **TTL** and optional **taker fallback**.
- A **single JSON** parameter file flows from optimization to backtest to live (`selected_params_15m_final.json` schema).

### Security note (important)
- The archive contains an `.env` with real‑looking `BINANCE_KEY` / `BINANCE_SECRET` values. **Treat them as compromised** and **rotate immediately**. Do not commit real keys. Restrict real keys by **IP** and **permissions**.

### Minor build gotcha
- `docker-compose.yml` references **`Dockerfile`** (capital “D”), but the archive contains **`dockerfile`** (lowercase). On a **case‑sensitive** filesystem (most Linux servers), this will fail. **Fix** by either renaming `dockerfile → Dockerfile` or changing `docker-compose.yml: build.dockerfile` to `dockerfile`.

### Optional confirmations (I proceeded with sensible defaults)
1) **Pair/interval**: Code defaults to `ETHBTC` at **15m** cadence (live code pulls 15m klines).  
2) **Fee modeling**: Backtest simulates **taker buys** (price↑ with slippage + taker fee) and **maker sells** (price↓ with slippage − maker fee)—close to live behavior with post‑only + TTL.  
3) **Fallback**: Live supports `--taker-fallback` after TTL cancel; default examples enable it only on testnet.  
4) **Walk‑forward routine**: I assume you’ll run several optimizer runs (different seeds/splits), then use `wf_pick.py` to select a family and emit the final JSON.

---

## 🧠 Strategy Overview

**Goal:** accumulate BTC by tilting between **BTC** and **ETH** based on a **trend ratio** with **hysteresis bands**, **volatility‑aware band widening**, **regime gating**, and **anti‑churn** controls.

**Signal (from `ethbtc_accum_bot.py`)**
- `trend_kind`: `"sma"` or `"roc"`  
  - SMA: `ratio = close / SMA(L) - 1`  
  - ROC: `ratio = close / close.shift(L) - 1`
- `trend_lookback` = `L`

**Hysteresis**
- Two bands: `flip_band_entry`, `flip_band_exit` (entry wider than exit to avoid churn).

**Volatility adaptation**
- `vol_window` (realized vol) + `vol_adapt_k` → widens both entry/exit bands by `k × realized_vol`.

**Regime gate (daily ROC)**
- `gate_window_days`, `gate_roc_threshold`: when the **absolute** daily ROC is below threshold, ETH flips can be **blocked** (stay in BTC).

**Execution smoothing & anti‑churn**
- `step_allocation` ∈ [0,1]: step part of the distance toward the target weight each bar.  
- `cooldown_minutes`: min time between flips.  
- `max_position`: cap ETH target weight.  
- `rebalance_threshold_w`: skip small trades.  
- `min_trade_*`: BTC‑notional minimums to pass exchange filters / noise.

**Fees & slippage (backtest)**
- `maker_fee`, `taker_fee`, `slippage_bps`, `bnb_discount`, `pay_fees_in_bnb`.  
- **Backtest side model:** **buys = taker**, **sells = maker**, with slippage applied to effective price in each direction.

---

## 🗂️ Repository Map & Synergies

| File | Role | Used by |
|---|---|---|
| `download_vision.py` | Download and merge Binance Vision kline CSVs | Backtests/optimizers |
| `vision_fix_timestamps.py` | Normalize candle timestamps to exact boundaries | Use if Vision files misalign |
| `ethbtc_accum_bot.py` | Strategy, backtester, simple optimizer | Core engine |
| `optimizer_cli.py` | Fast/parallel random search, robust scoring, emits JSON | Research → live |
| `wf_pick.py` | Walk‑forward family grouping/ranking; emits final JSON | Consolidation |
| `multi_interval_opt.py` | Compare multiple timeframes in one run | Interval vetting |
| `live_executor.py` | Live runner (dry/testnet/live), 15m cadence, post‑only+TTL | Production |
| `ascii_levelbar.py` | Console viz (bars/sparks) | Live UX |
| `selected_params_15m_final.json` | Example final config | Backtest & live |
| `requirements.txt` | Dependencies | All |
| `docker-compose.yml` + `dockerfile` + `entrypoint.sh` | Packaging & run | Docker users |

**Key synergy:** the **same JSON** config (from optimization/walk‑forward) is used for **backtest verification** and **live trading**.

---

## 🛠️ Installation

### Local (Python 3.11 recommended)
```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Docker
> **Case‑sensitive fix:** ensure the file is named **Dockerfile** or update compose to point at `dockerfile`.
```bash
# Option A: rename
mv dockerfile Dockerfile

# Build & run (after fixing compose or filename)
docker build -t ethbtc-bot:dev .
docker compose up -d
docker logs -f ethbtc_bot
```

### `.env` (never commit real keys)
```ini
# For TESTNET (remove BINANCE_BASE_URL for mainnet)
BINANCE_BASE_URL=https://testnet.binance.vision
BINANCE_KEY=***YOUR_TESTNET_KEY***
BINANCE_SECRET=***YOUR_TESTNET_SECRET***

SYMBOL=ETHBTC
MODE=testnet           # dry | testnet | live
STATE_FILE=state.json  # persisted state
TRADES_CSV=trades_log.csv
```
**Mainnet:** omit `BINANCE_BASE_URL`, set `MODE=live`, and use real keys with **IP restriction** + **minimal perms**.

---

## 📦 Data Pipeline (Binance Vision)

Download ETHBTC and (optionally) BNBBTC for fee‑in‑BNB modeling.

```bash
# ETHBTC 15m, 2021‑01 .. 2025‑10
python download_vision.py --symbol ETHBTC --intervals 15m \
  --start 2021-01 --end 2025-10 --out-dir data/vision

# BNBBTC 15m for fee modeling
python download_vision.py --symbol BNBBTC --intervals 15m \
  --start 2021-01 --end 2025-10 --out-dir data/vision
```

If timestamps are off, normalize:
```bash
python vision_fix_timestamps.py \
  --in  data/vision/ETHBTC_15m_2021-01_2025-10_vision.csv \
  --out data/ETHBTC_15m.csv --freq 15T

python vision_fix_timestamps.py \
  --in  data/vision/BNBBTC_15m_2021-01_2025-10_vision.csv \
  --out data/BNBBTC_15m.csv --freq 15T
```

> Loaders in `ethbtc_accum_bot.py` can read the Vision CSVs directly; the fixer is only for misaligned files.

---

## 📊 Backtesting

### A) Using a JSON config (recommended)
```bash
python ethbtc_accum_bot.py backtest \
  --data     data/ETHBTC_15m.csv \
  --bnb-data data/BNBBTC_15m.csv \
  --basis-btc 0.16 \
  --config  selected_params_15m_final.json \
  --start   2023-01-01 --end 2025-06-30 \
  --out     out/equity_backtest.csv \
  --excel-out out/backtest_report.xlsx
```
Outputs:
- Console summary (params, fees, window).
- `out/equity_backtest.csv` (wealth series, targets, etc.).
- `out/backtest_report.xlsx` (equity chart).

### B) Overriding params on CLI (CLI wins by default)
```bash
python ethbtc_accum_bot.py backtest \
  --data data/ETHBTC_15m.csv --bnb-data data/BNBBTC_15m.csv --basis-btc 0.16 \
  --strategy roc --trend 120 --flip-band-entry 0.0295 --flip-band-exit 0.0154 \
  --vol-window 45 --vol-adapt-k 0.0025 --target-vol 0.3 \
  --cooldown-minutes 240 --step-allocation 0.33 --max-position 0.8 \
  --gate-window-days 30 --gate-roc-threshold 0.02 \
  --rebalance-threshold-w 0.03 --min-trade-frac 0.0015 \
  --maker-fee 0.0002 --taker-fee 0.0004 --bnb-discount 0.25 --slippage-bps 1.0 \
  --out out/equity_backtest.csv --excel-out out/backtest_report.xlsx
```

---

## 🔎 Optimization Options

### A) Built‑in optimizer (simple random search)
```bash
python ethbtc_accum_bot.py optimize \
  --data data/ETHBTC_15m.csv --bnb-data data/BNBBTC_15m.csv \
  --train-start 2021-01-01 --train-end 2024-06-30 \
  --test-start  2024-07-01  --test-end  2025-06-30 \
  --n-random 120 \
  --maker-fee 0.0002 --taker-fee 0.0004 --slippage-bps 1.0 --bnb-discount 0.25 \
  --out out/opt_simple.csv --excel-out out/opt_simple.xlsx
```

### B) `optimizer_cli.py` (fast/parallel + robust scoring) ✅
Key extras:
- Threads/chunking, early stop, penalties: `--lambda-turns`, `--gap-penalty`, `--turns-scale`, `--lambda-fees`, `--lambda-turnover`.
- Emits a **ready‑to‑use JSON** via `--emit-config`.

```bash
python optimizer_cli.py \
  --data data/ETHBTC_15m.csv \
  --bnb-data data/BNBBTC_15m.csv \
  --train-start 2021-01-01 --train-end 2024-06-30 \
  --test-start  2024-07-01  --test-end  2025-06-30 \
  --n-random 600 --threads 0 --chunk-size 32 \
  --lambda-turns 2.0 --gap-penalty 0.35 --turns-scale 800 \
  --lambda-fees 2.0 --lambda-turnover 1.0 \
  --out out/opt_runs_seed1.csv --excel-out out/opt_runs_seed1.xlsx \
  --emit-config out/selected_params_seed1.json
```

> Tip: run multiple seeds/date‑slices to feed the walk‑forward picker.

---

## 🚶 Walk‑Forward Family Selection (`wf_pick.py`)

This groups similar parameter sets (bands/step/lookback/cooldown), filters by cost quantiles, requires repetition across runs, ranks by test performance with penalties, and **emits a final JSON**.

```bash
python wf_pick.py \
  --runs out/opt_runs_seed1.csv out/opt_runs_seed2.csv out/opt_runs_seed3.csv \
  --q-fees 0.75 --q-turnover 0.75 \
  --penalty-turns 0.0 --penalty-fees 1.0 --penalty-turnover 0.5 \
  --lb-bucket 40 --cd-bucket 60 --band-round 4 --step-round 2 \
  --top-k 12 --min-occurs 2 --require-costs --require-var \
  --out-csv  out/wf_ranked_families.csv \
  --excel-out out/wf_summary.xlsx \
  --emit-config out/selected_params_15m_final.json --family-index 0
```

Then verify the pick:
```bash
python ethbtc_accum_bot.py backtest \
  --data data/ETHBTC_15m.csv --bnb-data data/BNBBTC_15m.csv \
  --basis-btc 0.16 --config out/selected_params_15m_final.json \
  --start 2024-07-01 --end 2025-06-30 \
  --excel-out out/final_backtest.xlsx
```

---

## 🧪 Multi‑Interval Comparison (optional)

```bash
python multi_interval_opt.py \
  --data data/ETHBTC_5m.csv data/ETHBTC_15m.csv data/ETHBTC_30m.csv data/ETHBTC_1h.csv \
  --bnb-data data/BNBBTC_15m.csv \
  --train-start 2021-01-01 --train-end 2024-06-30 \
  --test-start  2024-07-01  --test-end  2025-06-30 \
  --n-random 200 \
  --excel-out out/multi_interval_summary.xlsx
```

> Note: live executor is hard‑coded to **15m** cadence; trading other intervals live would require code changes.

---

## ⚙️ Live Execution (dry → testnet → mainnet)

**Core behavior (from `live_executor.py`):**
- Pulls **15m klines**, computes target weight from the JSON config.
- Enforces **idempotency per closed 15m bar** via `state.json` (`last_bar_close`).
- Places **LIMIT_MAKER** (post‑only) orders; after `--ttl-sec`, can **cancel** and optionally send a **MARKET** taker fallback (`--taker-fallback`).
- Respects `LOT_SIZE`, `PRICE_FILTER`, and `MIN_NOTIONAL`, rounds accordingly.
- Logs fills to `trades_log.csv` with `maker_like` = 1 for maker, 0 for taker fallback.

**State & logs**
- `state.json`: do **not** delete mid‑session (prevents duplicate bar trades).  
- `trades_log.csv`: append‑only trade ledger.

### Dry‑run (no orders)
```bash
python live_executor.py \
  --params out/selected_params_15m_final.json \
  --mode dry --poll-sec 5 --ttl-sec 30 -vv
```

### Testnet
`.env` (example):
```ini
BINANCE_BASE_URL=https://testnet.binance.vision
MODE=testnet
BINANCE_KEY=***
BINANCE_SECRET=***
```
Run:
```bash
python live_executor.py \
  --params out/selected_params_15m_final.json \
  --mode testnet --taker-fallback --poll-sec 5 --ttl-sec 30 -vv
```

### Mainnet
- Remove `BINANCE_BASE_URL` from `.env`, set `MODE=live`, and use **real keys** with **IP restrictions**.
- Consider starting **without** `--taker-fallback` (or with longer `--ttl-sec`).

```bash
python live_executor.py \
  --params out/selected_params_15m_final.json \
  --mode live --poll-sec 10 --ttl-sec 90 -v
```

### Docker Compose (testnet or live)
```bash
# Ensure Dockerfile/compose mismatch is fixed (see earlier note)
docker build -t ethbtc-bot:dev .
docker compose up -d
docker logs -f ethbtc_bot
```

---

## 🧾 JSON Config Anatomy (example)

`selected_params_15m_final.json` (from the repo):
```json
{
  "trend_kind": "roc",
  "trend_lookback": 120,
  "flip_band_entry": 0.02946117452092501,
  "flip_band_exit": 0.015429849009990178,
  "vol_window": 45,
  "vol_adapt_k": 0.0025,
  "target_vol": 0.3,
  "min_mult": 0.5,
  "max_mult": 1.5,
  "cooldown_minutes": 240,
  "step_allocation": 0.33,
  "max_position": 0.8,
  "gate_window_days": 30,
  "gate_roc_threshold": 0.02,
  "rebalance_threshold_w": 0.03,
  "maker_fee": 0.0002,
  "taker_fee": 0.0004,
  "slippage_bps": 1.0,
  "bnb_discount": 0.25,
  "pay_fees_in_bnb": true,
  "basis_btc": 0.16,
  "min_trade_frac": 0.0015,
  "min_trade_floor_btc": 0.0002,
  "min_trade_cap_btc": 0.002,
  "min_trade_btc": null
}
```
Notes:
- The live loader tolerates flat or nested shapes (`fees`/`strategy`/`execution`) and coerces types.
- If `min_trade_btc` is `null`, dynamic min trade is derived from `min_trade_frac × basis_btc`, then floor/cap; live also checks the exchange’s `MIN_NOTIONAL`.

---

## 🧰 CLI Quick Reference

### `ethbtc_accum_bot.py backtest`
- Data/config: `--data` (req), `--bnb-data`, `--config`, `--basis-btc`, `--start`, `--end`, `--prefer-cli` (defaults **True**).
- Strategy: `--strategy {sma,roc}`, `--trend`, `--flip-band-entry`, `--flip-band-exit`, `--vol-window`, `--vol-adapt-k`, `--target-vol`, `--min-mult`, `--max-mult`, `--cooldown-minutes`, `--step-allocation`, `--max-position`, `--rebalance-threshold-w`, `--gate-window-days`, `--gate-roc-threshold`, `--min-trade-btc` or `--min-trade-frac`.
- Fees: `--maker-fee`, `--taker-fee`, `--slippage-bps`, `--bnb-discount`, `--no-bnb`.
- Output: `--out`, `--excel-out`.

### `ethbtc_accum_bot.py optimize`
- Windows: `--train-start`, `--train-end`, `--test-start`, `--test-end`.
- Search: `--n-random`.
- Fees: same as backtest.
- Output: `--out`, `--excel-out`.

### `optimizer_cli.py`
- Robust scoring: `--lambda-turns`, `--gap-penalty`, `--turns-scale`, `--lambda-fees`, `--lambda-turnover`.
- Performance: `--threads 0`, `--chunk-size`, and early‑stop (`--early-stop`, `--patience`, `--min-improve`) if implemented in your copy.
- Output: `--out`, `--excel-out`, **`--emit-config`**.

### `wf_pick.py`
- Inputs: one or more `--runs` CSVs.
- Cost filters: `--q-fees`, `--q-turnover`.
- Family bucketing: `--lb-bucket`, `--cd-bucket`, `--band-round`, `--step-round`.
- Robustness: `--min-occurs`, `--require-costs`, `--require-var`.
- Output: `--out-csv`, `--excel-out`, **`--emit-config`**, choose with `--family-index`.

### `multi_interval_opt.py`
- Inputs: multiple `--data` CSVs, optional `--bnb-data`.
- Windows: same as optimizer.
- Output: `--excel-out`.
- Penalties: `--lambda-turns`, `--gap-penalty`, `--turns-scale`, `--lambda-fees`, `--lambda-turnover`.

### `live_executor.py`
- Inputs: `--params` (req).
- Mode: `--mode {dry,testnet,live}` (or via `MODE` in `.env`).
- Runtime: `--poll-sec`, `--ttl-sec`, `--klines-limit`.
- Orders: `--taker-fallback` for MARKET after TTL cancel.
- I/O: `--state`, `--trades-csv`.
- Verbosity: `-v`, `-vv`.

---

## 🧭 End‑to‑End Workflows (copy/paste)

### Workflow A — *Download → backtest → optimize → pick → verify*
```bash
# 1) Data
python download_vision.py --symbol ETHBTC --intervals 15m --start 2021-01 --end 2025-10 --out-dir data/vision
python download_vision.py --symbol BNBBTC --intervals 15m --start 2021-01 --end 2025-10 --out-dir data/vision

# (optional) timestamp normalization
python vision_fix_timestamps.py --in data/vision/ETHBTC_15m_2021-01_2025-10_vision.csv --out data/ETHBTC_15m.csv --freq 15T
python vision_fix_timestamps.py --in data/vision/BNBBTC_15m_2021-01_2025-10_vision.csv --out data/BNBBTC_15m.csv --freq 15T

# 2) Quick backtest using example params
python ethbtc_accum_bot.py backtest \
  --data data/ETHBTC_15m.csv --bnb-data data/BNBBTC_15m.csv \
  --basis-btc 0.16 --config selected_params_15m_final.json \
  --start 2023-01-01 --end 2025-06-30 --excel-out out/backtest_report.xlsx

# 3) Robust optimizer (seed 1)
python optimizer_cli.py \
  --data data/ETHBTC_15m.csv --bnb-data data/BNBBTC_15m.csv \
  --train-start 2021-01-01 --train-end 2024-06-30 \
  --test-start  2024-07-01  --test-end  2025-06-30 \
  --n-random 600 --threads 0 --chunk-size 32 \
  --lambda-turns 2.0 --gap-penalty 0.35 --turns-scale 800 --lambda-fees 2.0 --lambda-turnover 1.0 \
  --out out/opt_runs_seed1.csv --excel-out out/opt_runs_seed1.xlsx

# (repeat optimizer for more seeds/windows)

# 4) Walk-forward picker
python wf_pick.py \
  --runs out/opt_runs_seed1.csv out/opt_runs_seed2.csv out/opt_runs_seed3.csv \
  --q-fees 0.75 --q-turnover 0.75 \
  --penalty-turns 0.0 --penalty-fees 1.0 --penalty-turnover 0.5 \
  --lb-bucket 40 --cd-bucket 60 --band-round 4 --step-round 2 \
  --top-k 12 --min-occurs 2 --require-costs --require-var \
  --emit-config out/selected_params_15m_final.json \
  --out-csv out/wf_ranked_families.csv --excel-out out/wf_summary.xlsx

# 5) Final verification backtest (test window)
python ethbtc_accum_bot.py backtest \
  --data data/ETHBTC_15m.csv --bnb-data data/BNBBTC_15m.csv \
  --basis-btc 0.16 --config out/selected_params_15m_final.json \
  --start 2024-07-01 --end 2025-06-30 \
  --excel-out out/final_backtest.xlsx
```

### Workflow B — *Live: dry → testnet → mainnet*

**Dry**
```bash
python live_executor.py \
  --params out/selected_params_15m_final.json \
  --mode dry --poll-sec 5 --ttl-sec 30 -vv
```

**Testnet**
```bash
# .env contains BINANCE_BASE_URL=https://testnet.binance.vision, MODE=testnet, and testnet keys
python live_executor.py \
  --params out/selected_params_15m_final.json \
  --mode testnet --taker-fallback --poll-sec 5 --ttl-sec 30 -vv
```

**Mainnet**
```bash
# Remove BINANCE_BASE_URL from .env, set MODE=live, use real keys with IP restrictions
python live_executor.py \
  --params out/selected_params_15m_final.json \
  --mode live --poll-sec 10 --ttl-sec 90 -v
```

**Docker Compose**
```bash
# Fix Dockerfile name mismatch first if on Linux
docker build -t ethbtc-bot:dev .
docker compose up -d
docker logs -f ethbtc_bot
```

---

## 🧯 Troubleshooting & Gotchas

- **`.env` keys in repo**: rotate immediately; treat as compromised. Add `.env` to `.gitignore` going forward.  
- **Dockerfile name mismatch**: `docker-compose.yml` expects `Dockerfile` but repo has `dockerfile`. Fix before building on Linux.  
- **Timestamps/Timezones**: Loaders use UTC and index by `close_time`. Normalize via `vision_fix_timestamps.py` if needed.  
- **MIN_NOTIONAL/LOT_SIZE**: If orders are rejected, your rounded `qty × price` may be too small; increase `min_trade_*` or target delta.  
- **BNB fee discount**: Fund BNB if `pay_fees_in_bnb=true` (live), or set it false to avoid discount modeling mismatch.  
- **Idempotency**: Don’t delete `state.json` mid‑run; the next loop could re‑trade the last closed bar.  
- **Dry mode**: Shows heartbeats/targets; no orders placed.  
- **Excel writer**: `--excel-out` produces an equity chart when wealth is present.

---

## 💡 My Advice & Future Enhancements

**Security & Ops**
- **Rotate committed keys** immediately; restrict real keys by **IP** and **least privilege**.  
- Add a **/health** endpoint or simple HTTP status for Compose/K8s healthchecks (last bar time, last order status, rolling error counter).  
- Implement **daily loss / drawdown caps** that **pause** the bot when breached.

**Execution & Realism**
- Improve the backtest slippage model with **spread/liquidity‑aware** estimates and **vol‑state** sensitivity.  
- Add **smart maker price maintenance** (rebid if book drifts) and consider **IOC** for tiny remainders instead of full MARKET fallback.  
- Make **dynamic min trade** react to `MIN_NOTIONAL` and balances each loop (you already load filters; use them to bump mins).

**Strategy & Robustness**
- Add a second‑stage search (e.g., **Bayesian optimization** or **CMA‑ES**) around top families.  
- Experiment with **volatility regime** gating or **drawdown clamps** alongside daily ROC.  
- Generalize to other BTC‑quoted symbols for robustness validation (research only), even if live remains ETHBTC.

**Engineering**
- Package as an installable module with **console scripts** (`ethbtc ...`), plus **pydantic** schema for configs.  
- Add **unit tests** for loaders, rounding/filters, gating, idempotency, and trade accounting.  
- Emit **structured logs** (JSON) and push metrics to **Prometheus** (bar latency, order outcomes, fills, errors).  
- Auto‑export **per‑trade ledgers** and **drawdowns** into the backtest Excel report.

---

## 🔚 Closing Notes

- Live executor is **15m‑fixed**; trading 5m/30m/1h live requires small code changes (interval parameterization + state logic).  
- Use one **JSON config** as the **single source of truth** from research to live to avoid drift.  
- For parity between backtest and live fees, include **BNBBTC** data if you intend to pay fees in BNB.

If you want, I can tailor a short **operations runbook** (alerts, dashboards, crontabs) around your infra. Otherwise, this manual is ready to use as‑is.
