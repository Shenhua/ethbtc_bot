This is the **Official Operations Manual** for the ETH/BTC Accumulation Bot v3.

It is designed for **Quants, DevOps Engineers, and Traders**. It covers the entire lifecycle: from downloading raw data and optimizing parameters to deploying a production-grade Docker container.

------



# 📘 ETH/BTC Accumulation Bot v3 — The Complete Manual





## 1. System Architecture & Logic





### What is this?



This is a **Volatility-Adaptive Mean-Reversion** system. It trades the **ETH/BTC** ratio.

- **Logic:** When ETH is statistically "cheap" (below trend), it swaps BTC for ETH. When ETH returns to the mean or becomes "expensive," it swaps ETH back to BTC.
- **Goal:** Increase the total amount of BTC held over time (Accumulation).



### Core Scripts Explained



- **`live_executor.py` (The Engine):**
  - Runs an infinite loop (Polling).
  - Fetches real-time price/balance from Binance.
  - Loads the strategy logic dynamically.
  - Manages state (persistence), risk checks (drawdown), and sends orders.
  - Exposes a Prometheus metrics server on port `9109`.
- **`ethbtc_accum_bot.py` (The Brain):**
  - Contains the `EthBtcStrategy` class.
  - Calculates indicators (SMA/ROC, Volatility, Bands) using Vectorized Pandas (fast).
  - Used by *both* the Backtester and the Live Engine to ensure logic parity.
- **`optimizer_cli.py` (The Lab):**
  - Runs thousands of backtests using randomized parameters to find the best configuration for a specific dataset.

------



## 2. Environment Setup





### Prerequisites



- **Docker** & **Docker Compose** installed.
- **Python 3.10+** (for local research).
- **Binance Account** (Mainnet) and **Binance Testnet Account**.



### Folder Structure



Create this exact structure to ensure all scripts work as intended:

Plaintext

```
ethbtc_bot/
├── configs/                 # JSON configuration files
│   └── prod_dynamic.json
├── data/                    # Data storage
│   ├── raw/                 # Zips from Binance Vision
│   └── clean/               # Processed CSVs
├── run_state/               # Persistent state (DO NOT DELETE unless resetting)
│   └── state.json
├── .env                     # API Keys (Secrets)
├── docker-compose.yml       # Deployment config
├── Dockerfile               # Container definition
├── requirements.txt         # Python dependencies
└── *.py                     # All python scripts (live_executor, optimizer, etc.)
```



### The `.env` File (Crucial)



Create a file named `.env` in the root. **Never commit this to Git.**

Ini, TOML

```
# --- API KEYS ---
# Get these from https://testnet.binance.vision for testing
BINANCE_KEY=your_api_key_here
BINANCE_SECRET=your_secret_key_here

# --- NETWORK SETTINGS ---
# Uncomment for TESTNET. Comment out for REAL MONEY (Live).
BINANCE_BASE_URL=https://testnet.binance.vision

# --- BOT SETTINGS ---
SYMBOL=ETHBTC
MODE=testnet        # Options: dry | testnet | live
STATE_FILE=/data/state.json
```

------



## 3. The Research Workflow (Quant Phase)



*Scenario: You want to find the best parameters for the current market.*



### Step 1: Acquire High-Fidelity Data



Use `download_vision.py` to get official tick-perfect data from Binance.

Bash

```
# Download 15-minute candles for 2023-2024
python download_vision.py \
  --symbol ETHBTC \
  --intervals 15m \
  --start 2023-01-01 \
  --end 2024-06-01 \
  --out-dir data/raw
```

*Result:* You will get a merged CSV in `data/raw/`.



### Step 2: Run the Optimizer



Use `optimizer_cli.py` to brute-force find profitable parameters using random search.

Bash

```
python optimizer_cli.py \
  --data data/raw/ETHBTC_15m_2023-01-01_2024-06-01_vision.csv \
  --train-start 2023-01-01 \
  --train-end 2023-12-31 \
  --test-start 2024-01-01 \
  --test-end 2024-06-01 \
  --n-random 1000 \
  --out opt_results_run1.csv
```

- **`--n-random 1000`**: Will simulate 1,000 different strategy variations.
- **`--out`**: Saves the metrics (PnL, Drawdown, Sharpe) to a CSV.



### Step 3: Pick the Best Family



Don't just pick the highest return (it might be luck). Use `wf_pick.py` to find stable parameter "families."

Bash

```
python wf_pick.py \
  --runs opt_results_run1.csv \
  --emit-config configs/best_strategy.json \
  --top-k 5
```

*Result:* A JSON file `configs/best_strategy.json` containing the optimal strategy parameters.

------



## 4. Configuration (JSON Schema)



You need to construct a full config file (e.g., configs/prod_live.json).

Combine the Strategy (from Step 3) with your Execution and Risk preferences.

**Sample `prod_live.json`:**

JSON

```
{
  "strategy": {
    "trend_kind": "sma",           // Trend type: "sma" or "roc"
    "trend_lookback": 200,         // Length of trend
    "flip_band_entry": 0.025,      // Buy at -2.5% deviation
    "flip_band_exit": 0.015,       // Stop buying at -1.5%
    "bar_interval_minutes": 15,    // Candle size
    "gate_window_days": 60,        // Regime filter lookback
    "gate_roc_threshold": 0.0      // Filter threshold
  },
  "execution": {
    "interval": "15m",             // MUST match bar_interval_minutes
    "poll_sec": 5,                 // How often to check price
    "min_trade_btc": 0.001         // Minimum order size
  },
  "risk": {
    "basis_btc": 1.0,              // Starting capital (for dry run calculations)
    "risk_mode": "dynamic",        // "dynamic" (scales with equity) or "fixed_basis"
    "max_dd_frac": 0.15,           // Stop trading if 15% drawdown from peak
    "max_daily_loss_frac": 0.03    // Pause trading if 3% loss in one day
  }
}
```

------



## 5. Docker Deployment (DevOps Phase)





### The `docker-compose.yml`



Ensure your compose file passes the environment variables correctly.

YAML

```
version: "3.8"
services:
  ethbtc_bot:
    build: .
    container_name: ethbtc_bot_v3
    environment:
      - BINANCE_KEY=${BINANCE_KEY}
      - BINANCE_SECRET=${BINANCE_SECRET}
      - BINANCE_BASE_URL=${BINANCE_BASE_URL} # Important for Testnet
    command: >
      python /app/live_executor.py
      --params configs/prod_live.json
      --mode ${MODE}
      --symbol ${SYMBOL}
    volumes:
      - ./run_state:/data   # Persist the state.json outside container
      - .:/app              # Bind mount code (optional, for dev)
    ports:
      - "9109:9109"         # Prometheus metrics
    restart: unless-stopped
```



### Deployment Workflow





#### Use Case A: The "Testnet Verification" (Recommended)



1. **Seed Wallet:** Testnet accounts start with 0 ETH. You need ETH to test selling.

   Bash

   ```
   # Install connector locally first
   pip install binance-connector
   # Run seeder (Buy 0.5 ETH)
   python seed_testnet_order.py --qty 0.5 --side BUY
   ```

2. **Configure .env:** Set `MODE=testnet` and uncomment `BINANCE_BASE_URL`.

3. **Launch:**

   Bash

   ```
   docker-compose up -d --build --force-recreate
   ```

4. **Verify:** Check logs.

   Bash

   ```
   docker-compose logs -f
   ```

   *Look for:* `[BALANCE] Successfully fetched...`



#### Use Case B: The "Live" Deployment (Real Money)



1. **Configure .env:**

   - Set `MODE=live`.
   - **Comment out** `BINANCE_BASE_URL` (so it defaults to Mainnet).
   - Update keys to your **Real** API keys.

2. **Clean State:** Remove old Testnet memory.

   Bash

   ```
   rm run_state/state.json
   ```

3. **Launch:**

   Bash

   ```
   docker-compose up -d --build --force-recreate
   ```

------



## 6. Observability (Monitoring)



The bot is a "Black Box" without Grafana.



### 1. Metrics Endpoint



The bot serves raw metrics at `http://localhost:9109`.



### 2. Grafana Dashboard



- **Install Grafana & Prometheus.**
- **Import Dashboard:** Use the JSON provided in Release v3.0.1.
- **Key Panels:**
  - **Blocker Timeline:** Tells you *why* the bot isn't trading (e.g., `skip_threshold`).
  - **Proximity:** Tells you how close (in basis points) the price is to a buy/sell level.

------



## 7. Troubleshooting Guide



| **Issue**                | **Error Message**                                  | **Solution**                                                 |
| ------------------------ | -------------------------------------------------- | ------------------------------------------------------------ |
| **Invalid Key**          | `Error -2015: Invalid API-key, IP, or permissions` | You are sending Testnet keys to Mainnet URL (or vice-versa). Check `BINANCE_BASE_URL` in `.env` and `docker-compose.yml`. |
| **Insufficient Balance** | `Error -2010: Account has insufficient balance`    | The bot didn't reserve fees. **Update code** to use `0.999` buffer in `live_executor.py` logic. |
| **Fake Profits**         | `pnl_btc` shows massive gains (e.g. +0.8 BTC)      | The bot started with a default `0.16` BTC before connecting. **Fix:** `rm run_state/state.json` and restart. |
| **Stuck Orders**         | Target Weight line diverges from Current Weight    | The bot is trying to trade but failing validation (`min_notional`). Increase `min_trade_btc` in config. |
| **Updates Ignored**      | Code changes not appearing in logs                 | Docker is using a cached build. Run `docker-compose up -d --build`. |

------



## 8. Tool Reference





### `seed_testnet_order.py`



- **Purpose:** Manually buys/sells assets on Testnet to set up a specific portfolio scenario (e.g., "Start with 50% ETH").
- **Usage:** `python seed_testnet_order.py --side BUY --qty 1.0`



### `download_vision.py`



- **Purpose:** Downloads massive datasets. Handles the complex logic of "Monthly Zips" + "Daily Zips" fallback automatically.
- **Pro Tip:** Use `--prefer-daily` if you need the absolute latest data from the current month.



### `state.json` (The Brain's Memory)



Located in `run_state/`. It stores:

- `session_start_W`: Your wealth when you first turned the bot on (for PnL).
- `risk_maxdd_hit`: Boolean flag. If `true`, the bot acts as a "Circuit Breaker" and refuses to trade.
- **To Reset Risk:** Manually edit this file to `false` or delete the file.