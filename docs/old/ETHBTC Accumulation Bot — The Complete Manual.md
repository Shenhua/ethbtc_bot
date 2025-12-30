# 📘 ETH/BTC Bot v4 — Operations Manual (Fleet Edition)





## 1. System Overview



This is a **Volatility-Adaptive Mean-Reversion** system designed to accumulate the **Quote Asset** of any crypto pair.

- **Pairs:** Can trade `ETH/BTC` (Accumulate BTC), `BNB/USDC` (Accumulate USDC), etc.
- **Logic:** Buys Base Asset (e.g., ETH) when oversold; Sells back to Quote Asset (e.g., BTC) when overbought.
- **Safety:** Includes "Funding Rate" monitoring to detect market euphoria/panic.
- **Maintenance:** Includes a "Dust Sweeper" service to clean up untradeable balances.

------



## 2. Environment Setup





### Folder Structure



Plaintext

```
ethbtc_bot/
├── configs/                 # Strategy configs (e.g., prod_eth.json, prod_bnb.json)
├── data/                    # Historical data
│   └── raw/                 # CSVs from Binance Vision
├── run_state/               # Persistent State
│   ├── eth/                 # State file for ETH bot
│   └── bnb/                 # State file for BNB bot
├── .env                     # API Keys & Global Settings
├── docker-compose.yml       # Fleet Definition
└── *.py                     # Python Source Code
```



### The `.env` File



Ini, TOML

```
# --- API KEYS (Required) ---
BINANCE_KEY=your_api_key
BINANCE_SECRET=your_secret_key

# --- NETWORK (Comment out BINANCE_BASE_URL for Live) ---
# BINANCE_BASE_URL=https://testnet.binance.vision
MODE=testnet  # Options: dry | testnet | live

# --- SWEEPER CONFIG ---
# Assets to NEVER sell for dust (Moon bags + Gas + Quote Assets)
SWEEPER_IGNORE=BNB,BTC,ETH,USDT,USDC,SHIB
```

------



## 3. Phase 1: Research & Optimization



*How to find parameters for a new pair (e.g., BNB/USDC).*



### Step A: Download Data



Use `download_vision.py` to get official history.

Bash

```
python download_vision.py --symbol BNBUSDC --intervals 15m --start 2023-01 --end 2024-10 --out-dir data/raw
```



### Step B: Run Optimizer



Use `optimizer_cli.py`. **Note:** The output column `test_final_btc` actually represents "Test Final Quote Asset" (USDC in this case).

Bash

```
python optimizer_cli.py \
  --data data/raw/BNBUSDC_15m_....csv \
  --train-start 2023-01-01 --train-end 2023-12-31 \
  --test-start 2024-01-01 --test-end 2024-10-01 \
  --n-random 1000 \
  --out opt_bnb.csv
```



### Step C: Select Stable Parameters



Use `wf_pick.py` to find the most robust cluster of parameters.

Bash

```
python wf_pick.py --runs opt_bnb.csv --emit-config configs/prod_bnb.json --top-k 5
```

------



## 4. Phase 2: Configuration (JSON)



You need one config file per bot.

**Example `configs/prod_bnb.json` (for BNB/USDC):**

JSON

```
{
  "strategy": {
    "trend_kind": "sma",
    "trend_lookback": 180,
    "flip_band_entry": 0.03,       // BNB might need wider bands than ETH
    "flip_band_exit": 0.01,
    "bar_interval_minutes": 15,
    "funding_limit_long": 0.05,    // Stop buying if funding > 0.05%
    "funding_limit_short": -0.05
  },
  "execution": {
    "interval": "15m",
    "poll_sec": 5,
    "min_trade_btc": 10.0          // IMPORTANT: For USDC pairs, this means 10 USDC!
  },
  "risk": {
    "basis_btc": 1000.0,           // IMPORTANT: This means 1000 USDC
    "risk_mode": "dynamic",
    "max_dd_frac": 0.15
  }
}
```

------



## 5. Phase 3: Docker Deployment (The Fleet)





### `docker-compose.yml` Setup



Define one service per pair. Map unique ports and state folders.

YAML

```
services:
  # --- Bot 1: ETH/BTC ---
  bot_eth:
    image: ethbtc_bot_v4
    build: .
    container_name: ethbtc_bot
    restart: unless-stopped
    environment:
      - BINANCE_KEY=${BINANCE_KEY}
      - BINANCE_SECRET=${BINANCE_SECRET}
      - BINANCE_BASE_URL=${BINANCE_BASE_URL}
      - MODE=${MODE}
      - SYMBOL=ETHBTC
    volumes:
      - ./run_state/eth:/data
    command: >
      python /app/live_executor.py 
      --params configs/prod_eth.json 
      --mode ${MODE} --symbol ETHBTC --state /data/state.json
    ports:
      - "9100:9109"

  # --- Bot 2: BNB/USDC ---
  bot_bnb:
    image: ethbtc_bot_v4
    container_name: bnbusdc_bot
    restart: unless-stopped
    environment:
      - BINANCE_KEY=${BINANCE_KEY}
      - BINANCE_SECRET=${BINANCE_SECRET}
      - BINANCE_BASE_URL=${BINANCE_BASE_URL}
      - MODE=${MODE}
      - SYMBOL=BNBUSDC
    volumes:
      - ./run_state/bnb:/data
    command: >
      python /app/live_executor.py 
      --params configs/prod_bnb.json 
      --mode ${MODE} --symbol BNBUSDC --state /data/state.json
    ports:
      - "9101:9109"

  # --- Sweeper Service ---
  sweeper:
    build: .
    environment:
      - BINANCE_KEY=${BINANCE_KEY}
      - BINANCE_SECRET=${BINANCE_SECRET}
      - BINANCE_BASE_URL=${BINANCE_BASE_URL}
      - MODE=${MODE}
      - SWEEPER_IGNORE=${SWEEPER_IGNORE}
    command: python /app/dust_sweeper.py --mode auto --schedule weekly
```



### Launching



Bash

```
docker-compose up -d --build
```

------



## 6. Phase 4: Observability



**Grafana Dashboard v4.2 ("Admiral")** allows switching between bots.

1. **Prometheus Config:** Ensure `prometheus.yml` scrapes `localhost:9100` (labeled `eth`) and `localhost:9101` (labeled `bnb`).
2. **Dashboard:** Import the v4.2 JSON. Use the dropdown at the top left to switch views.