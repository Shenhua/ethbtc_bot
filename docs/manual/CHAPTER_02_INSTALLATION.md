# Chapter 2: Installation & Deployment

> **Purpose:** This chapter provides exhaustive, step-by-step instructions for installing and deploying the ETH/BTC Algorithmic Trading Bot in all supported environments: local development, Docker containers, and production servers.

---

## 2.1 Prerequisites

### 1. Concept & "The Why"

* **What it is:** A checklist of required software, accounts, and credentials needed before installing the bot.

* **Purpose:** Ensures you have everything ready before starting installation. Missing prerequisites cause 90% of "it doesn't work" issues.

* **Location:** Requirements defined in [`requirements.txt`](../requirements.txt) and [`Dockerfile`](../Dockerfile).

### 2. Configuration & Parameters

#### System Requirements

| Requirement | Minimum | Recommended | Notes |
|-------------|---------|-------------|-------|
| **Operating System** | Linux, macOS, Windows (WSL2) | Ubuntu 22.04 LTS | Windows native not supported; use WSL2 |
| **Python** | 3.10 | 3.11 | Dockerfile uses `python:3.11-slim` |
| **RAM** | 512 MB | 2 GB | More RAM needed for backtesting |
| **Disk Space** | 500 MB | 5 GB | Historical data consumes space |
| **Docker** | 20.10+ | 24.0+ | Required for production deployment |
| **Docker Compose** | 2.0+ | 2.20+ | Compose V2 required |

#### Python Dependencies (from `requirements.txt`)

| Package | Version | Purpose |
|---------|---------|---------|
| `binance-connector` | 3.12.0 | Binance Spot API client |
| `binance-futures-connector` | 4.1.0 | Binance Futures API client |
| `python-dotenv` | ≥1.0.0 | Load `.env` files |
| `pandas` | 2.2.0 | Data manipulation |
| `numpy` | 1.26.4 | Numerical operations |
| `pydantic` | ≥2.0 | Configuration validation |
| `prometheus-client` | ≥0.17 | Metrics export |
| `optuna` | ≥3.0 | Hyperparameter optimization |
| `tenacity` | ≥8.2.0 | Retry logic |
| `structlog` | ≥24.0.0 | Structured logging |
| `rich` | ≥14.2.0 | Rich console output |

#### Binance Account Requirements

| Account Type | Purpose | How to Create |
|--------------|---------|---------------|
| **Spot Testnet** | Testing without real money | https://testnet.binance.vision |
| **Futures Testnet** | Testing futures without real money | https://testnet.binancefuture.com |
| **Live Spot** | Real trading on spot market | https://www.binance.com |
| **Live Futures** | Real trading on USDS-M futures | Enable in Binance Futures settings |

### 3. Step-by-Step Guide: Getting Your API Keys

#### Spot Testnet Keys

1. **Navigate to Binance Spot Testnet:**
   ```
   https://testnet.binance.vision
   ```

2. **Authenticate with GitHub:**
   - Click "Log In with GitHub"
   - Authorize the application

3. **Generate API Keys:**
   - Click "Generate HMAC_SHA256 Key"
   - **Copy both values immediately** (they won't be shown again):
     - API Key → `SPOT_TESTNET_KEY`
     - Secret Key → `SPOT_TESTNET_SECRET`

#### Futures Testnet Keys

1. **Navigate to Binance Futures Testnet:**
   ```
   https://testnet.binancefuture.com
   ```

2. **Create Account:**
   - Click "Register" and complete signup
   - Verify email

3. **Generate API Keys:**
   - Go to API Management (top right → API Management)
   - Click "Create API"
   - **Copy both values immediately**:
     - API Key → `FUTURES_TESTNET_KEY`
     - Secret Key → `FUTURES_TESTNET_SECRET`

#### Live Production Keys

1. **Log into Binance:**
   ```
   https://www.binance.com
   ```

2. **Navigate to API Management:**
   - Profile → API Management

3. **Create API Key:**
   - Click "Create API"
   - Select "System generated"
   - Complete 2FA verification

4. **Configure Permissions:**
   - ✅ Enable Spot & Margin Trading
   - ✅ Enable Futures (if using futures)
   - ❌ Disable Withdrawals (for safety)
   - Set IP whitelist (recommended)

5. **Copy Keys:**
   - API Key → `BINANCE_KEY`
   - Secret Key → `BINANCE_SECRET`

### 4. Real-World Use Case (The "Cookbook")

**Scenario:** User wants to verify all prerequisites are met.

**Checklist Command:**
```bash
# Check Python version
python3 --version
# Expected: Python 3.10.x or 3.11.x

# Check Docker
docker --version
# Expected: Docker version 24.x.x or higher

# Check Docker Compose
docker compose version
# Expected: Docker Compose version v2.20.x or higher

# Check disk space
df -h .
# Expected: At least 5GB available

# Test Binance API connectivity
curl -s https://api.binance.com/api/v3/ping
# Expected: {}
```

**Expected Outcome:** All commands return expected values without errors.

### 5. Troubleshooting & Edge Cases

* **What can go wrong:**
  - Python version mismatch (using 3.9 instead of 3.10+)
  - Docker not running (`Cannot connect to Docker daemon`)
  - Firewall blocking Binance API

* **Error Messages:**

  ```
  ModuleNotFoundError: No module named 'pydantic'
  ```
  **Cause:** Dependencies not installed.
  **Fix:** `pip install -r requirements.txt`

  ```
  binance.error.ClientError: APIError(code=-2015): Invalid API-key
  ```
  **Cause:** Wrong API keys or using testnet keys against production.
  **Fix:** Verify keys match the environment (testnet vs live).

---

## 2.2 Local Development Setup

### 1. Concept & "The Why"

* **What it is:** Installing the bot directly on your development machine without Docker, for rapid iteration and debugging.

* **Purpose:** Enables faster development cycles, easier debugging with breakpoints, and direct access to logs.

* **Location:** No specific file—this is a workflow.

### 2. Configuration & Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Virtual environment | `.venv/` | Isolated Python environment |
| PYTHONPATH | Project root | Auto-set when using venv |

### 3. Step-by-Step Guide

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-org/ethbtc_bot_3.git
   cd ethbtc_bot_3
   ```

2. **Create Virtual Environment:**
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   # OR
   .venv\Scripts\activate     # Windows PowerShell
   ```

3. **Install Dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Create Environment File:**
   ```bash
   cp .env.example .env  # If example exists
   # OR create from scratch:
   touch .env
   ```

5. **Configure `.env` for Testnet:**
   ```bash
   # .env file contents:
   SPOT_TESTNET_KEY=your_spot_testnet_api_key_here
   SPOT_TESTNET_SECRET=your_spot_testnet_secret_here
   FUTURES_TESTNET_KEY=your_futures_testnet_api_key_here
   FUTURES_TESTNET_SECRET=your_futures_testnet_secret_here
   BINANCE_BASE_URL=https://testnet.binance.vision
   MODE=testnet
   LOGLEVEL=DEBUG
   ```

6. **Create State Directory:**
   ```bash
   mkdir -p run_state/eth run_state/btc
   ```

7. **Run First Test:**
   ```bash
   python -m pytest tests/ -v --tb=short
   ```
   **Expected Output:**
   ```
   ========================= test session starts ==========================
   collected 92 items
   ...
   ========================= 92 passed in 12.34s ==========================
   ```

8. **Run Bot in Dry Mode (No Real Trades):**
   ```bash
   python live_executor.py \
     --params configs/prod_meta_live.json \
     --mode dry \
     --symbol ETHBTC \
     --once
   ```

### 4. Real-World Use Case (The "Cookbook")

**Scenario:** Developer wants to debug signal generation.

**Configuration:**
```bash
# Set debug logging
export LOGLEVEL=DEBUG

# Run with debugger (VS Code launch.json)
{
  "name": "Debug Live Executor",
  "type": "python",
  "request": "launch",
  "program": "${workspaceFolder}/live_executor.py",
  "args": [
    "--params", "configs/prod_meta_live.json",
    "--mode", "dry",
    "--symbol", "ETHBTC",
    "--once"
  ],
  "env": {
    "PYTHONPATH": "${workspaceFolder}"
  }
}
```

**Expected Outcome:**
- Breakpoints work in VS Code/PyCharm
- Full debug logs printed to console
- Single bar processed, then exit (due to `--once`)

### 5. Troubleshooting & Edge Cases

* **What can go wrong:**
  - Virtual environment not activated (wrong Python used)
  - Missing `PYTHONPATH` causing import errors
  - `.env` file not loaded

* **Error Messages:**

  ```
  ModuleNotFoundError: No module named 'core'
  ```
  **Cause:** `PYTHONPATH` not set or not in project root.
  **Fix:** Ensure you're in the project root directory, or set:
  ```bash
  export PYTHONPATH="${PYTHONPATH}:$(pwd)"
  ```

  ```
  FileNotFoundError: [Errno 2] No such file or directory: 'configs/prod_meta_live.json'
  ```
  **Cause:** Running from wrong directory.
  **Fix:** `cd` to project root before running.

---

## 2.3 Docker Deployment

### 1. Concept & "The Why"

* **What it is:** Containerized deployment using Docker and Docker Compose, providing isolated, reproducible environments with automatic restarts.

* **Purpose:** 
  - Consistent deployment across any host
  - Automatic restart on failure (`restart: unless-stopped`)
  - Isolated state per bot instance
  - Integrated observability stack (Prometheus + Grafana)

* **Location:** 
  - [`Dockerfile`](../Dockerfile) — Container image definition
  - [`docker-compose.yml`](../docker-compose.yml) — Multi-container orchestration
  - [`entrypoint.sh`](../entrypoint.sh) — Container initialization script

### 2. Configuration & Parameters

#### Dockerfile Breakdown

```dockerfile
# Base Image: Python 3.11 on Debian Slim (small footprint)
FROM python:3.11-slim

# Working directory inside container
WORKDIR /app

# Prevent Python from buffering stdout (important for Docker logs)
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="${PYTHONPATH}:/app"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl tini && \
    rm -rf /var/lib/apt/lists/*

# tini: Proper init process for containers (handles signals)
ENTRYPOINT ["/usr/bin/tini", "--", "./entrypoint.sh"]

# Default command (overridden by docker-compose)
CMD ["python", "live_executor.py", "--params", "configs/prod_meta_live.json"]
```

**Hidden Logic:**
- `tini` is used as PID 1 to properly handle SIGTERM/SIGINT signals
- `PYTHONUNBUFFERED=1` ensures logs appear immediately in `docker logs`
- `__pycache__` directories are cleaned during build to reduce image size

#### Docker Compose Services

| Service | Container Name | Purpose | Ports |
|---------|---------------|---------|-------|
| `bot_eth` | `ethbtc_bot` | ETH/BTC Spot Trading | 9109 (metrics) |
| `bot_btc` | `btcusdt_bot` | BTC/USDT Futures Trading | 9102→9109 |
| `sweeper` | `global_sweeper` | Dust conversion utility | — |
| `prometheus` | `ethbtc_prometheus` | Metrics collection | 9999→9090 |
| `grafana` | `ethbtc_grafana` | Metrics visualization | 3000 |

#### Environment Variables in Docker Compose

| Variable | Service | Description |
|----------|---------|-------------|
| `BINANCE_KEY` | bot_eth, bot_btc | API key (from `.env`) |
| `BINANCE_SECRET` | bot_eth, bot_btc | API secret (from `.env`) |
| `MODE` | All bots | `dry`, `testnet`, or `live` |
| `EXCHANGE_TYPE` | bot_eth, bot_btc | `spot` or `futures` |
| `SYMBOL` | bot_eth, bot_btc | Trading pair (ETHBTC, BTCUSDT) |
| `DISCORD_WEBHOOK_URL` | bot_eth, bot_btc | Alert webhook |

#### Volume Mounts

| Mount | Container Path | Purpose |
|-------|---------------|---------|
| `./run_state/eth:/data` | `/data` | ETH bot state persistence |
| `./run_state/btc:/data` | `/data` | BTC bot state persistence |
| `.:/app` | `/app` | Live code mount (dev mode) |
| `./prometheus.yml:/etc/prometheus/prometheus.yml` | Config | Prometheus scrape config |
| `./prometheus_data:/prometheus` | Data | Prometheus time-series data |
| `./grafana_data:/var/lib/grafana` | Data | Grafana dashboards & settings |

### 3. Step-by-Step Guide

#### First-Time Setup

1. **Create Docker Network:**
   ```bash
   docker network create trading-net
   ```
   This network allows containers to communicate by name.

2. **Create Data Directories:**
   ```bash
   mkdir -p run_state/eth run_state/btc prometheus_data grafana_data
   ```

3. **Set Directory Permissions (Linux only):**
   ```bash
   # Prometheus runs as nobody:nogroup (UID 65534)
   sudo chown -R 65534:65534 prometheus_data
   
   # Grafana runs as grafana (UID 472)
   sudo chown -R 472:472 grafana_data
   ```

4. **Configure Environment:**
   ```bash
   # Edit .env with your API keys
   nano .env
   ```
   
   Minimum required for testnet:
   ```bash
   SPOT_TESTNET_KEY=your_key
   SPOT_TESTNET_SECRET=your_secret
   FUTURES_TESTNET_KEY=your_futures_key
   FUTURES_TESTNET_SECRET=your_futures_secret
   MODE=testnet
   ```

5. **Build and Start:**
   ```bash
   docker compose up -d --build
   ```

6. **Verify All Services Running:**
   ```bash
   docker compose ps
   ```
   **Expected Output:**
   ```
   NAME                 STATUS              PORTS
   ethbtc_bot           Up 2 minutes        0.0.0.0:9109->9109/tcp
   btcusdt_bot          Up 2 minutes        0.0.0.0:9102->9109/tcp
   global_sweeper       Up 2 minutes
   ethbtc_prometheus    Up 2 minutes        0.0.0.0:9999->9090/tcp
   ethbtc_grafana       Up 2 minutes        0.0.0.0:3000->3000/tcp
   ```

7. **Check Bot Logs:**
   ```bash
   # ETH bot logs
   docker logs -f ethbtc_bot
   
   # BTC bot logs
   docker logs -f btcusdt_bot
   ```

8. **Access Dashboards:**
   - Prometheus: http://localhost:9999
   - Grafana: http://localhost:3000 (admin/admin)

#### Switching to Production

1. **Update `.env` for Live:**
   ```bash
   # Comment out testnet URL (uses production by default)
   # BINANCE_BASE_URL=https://testnet.binance.vision
   
   # Set live keys
   BINANCE_KEY=your_live_api_key
   BINANCE_SECRET=your_live_api_secret
   
   # CRITICAL: Change mode
   MODE=live
   ```

2. **Restart Services:**
   ```bash
   docker compose down
   docker compose up -d
   ```

### 4. Real-World Use Case (The "Cookbook")

**Scenario:** Deploy BTC futures bot only (no ETH bot).

**Configuration:** Create `docker-compose.override.yml`:
```yaml
# docker-compose.override.yml
services:
  bot_eth:
    profiles:
      - disabled  # Won't start unless explicitly requested
```

**Commands:**
```bash
# Start only BTC bot + observability
docker compose up -d bot_btc prometheus grafana

# Verify
docker compose ps
# Only bot_btc, prometheus, grafana should be running
```

**Expected Outcome:** Only BTC bot trading, no ETH bot running.

### 5. Troubleshooting & Edge Cases

* **What can go wrong:**
  - Network `trading-net` doesn't exist
  - Permission denied on data directories
  - Port already in use

* **Error Messages:**

  ```
  network trading-net declared as external, but could not be found
  ```
  **Cause:** Docker network not created.
  **Fix:** `docker network create trading-net`

  ```
  Error response from daemon: Ports are not available: exposing port TCP 0.0.0.0:3000
  ```
  **Cause:** Another service using port 3000.
  **Fix:** Change Grafana port in `docker-compose.yml`:
  ```yaml
  grafana:
    ports:
      - "3001:3000"  # Use 3001 instead
  ```

  ```
  prometheus  | level=error msg="Opening storage failed" err="open /prometheus/lock: permission denied"
  ```
  **Cause:** Wrong ownership on `prometheus_data`.
  **Fix:** `sudo chown -R 65534:65534 prometheus_data`

---

## 2.4 Directory Structure Reference

### 1. Concept & "The Why"

* **What it is:** Complete map of the project directory structure with explanations.

* **Purpose:** Helps new users understand where to find things and where to put new files.

* **Location:** Project root.

### 2. Configuration & Parameters

```
ethbtc_bot_3/
├── .env                      # Environment variables (API keys, mode)
├── .gitignore                # Files excluded from git
├── .pre-commit-config.yaml   # Pre-commit hooks configuration
├── Dockerfile                # Container image definition
├── docker-compose.yml        # Multi-container orchestration
├── entrypoint.sh             # Container initialization script
├── live_executor.py          # 🚀 MAIN ENTRY POINT (1,373 lines)
├── prometheus.yml            # Prometheus scrape configuration
├── requirements.txt          # Python dependencies
├── mypy.ini                  # Type checking configuration
│
├── configs/                  # 📁 Strategy Configuration Files
│   ├── prod_btc_meta_live.json       # Production BTC config
│   ├── prod_eth_long_wfo_robust.json # Production ETH config
│   ├── test_*.json                   # Test configurations
│   └── archive/                      # Old/deprecated configs
│
├── core/                     # 📁 Core Trading Logic (26 files)
│   ├── __init__.py
│   ├── ethbtc_accum_bot.py   # Mean Reversion strategy + Backtester
│   ├── trend_strategy.py     # Trend Following strategy
│   ├── meta_strategy.py      # Ensemble strategy (regime-switching)
│   ├── position_sizer.py     # Dynamic position sizing
│   ├── risk_manager.py       # Risk management + Phoenix Protocol
│   ├── config_schema.py      # Pydantic configuration validation
│   ├── binance_adapter.py    # Spot exchange adapter
│   ├── futures_adapter.py    # Futures exchange adapter
│   ├── resilience.py         # Circuit breaker + retry logic
│   ├── metrics.py            # Prometheus metrics
│   ├── alert_manager.py      # Discord/Telegram alerts
│   ├── story_writer.py       # Trading narrative logger
│   ├── backtest_report.py    # Backtest report generation
│   ├── regime.py             # ADX regime detection
│   └── ...
│
├── tools/                    # 📁 CLI Utilities & Optimization (29 files)
│   ├── run_optimization.py   # WFO orchestrator
│   ├── optimizer_cli.py      # Optuna optimizer
│   ├── wfo_select_best.py    # WFO window selection
│   ├── dust_sweeper.py       # Dust conversion utility
│   ├── download_vision.py    # Historical data downloader
│   └── ...
│
├── tests/                    # 📁 Unit & Integration Tests (25 files)
│   ├── test_position_sizer.py
│   ├── test_risk_manager.py
│   ├── test_phoenix_protocol.py
│   └── ...
│
├── data/                     # 📁 Data Storage
│   ├── raw/                  # Historical OHLCV CSVs
│   │   ├── BTCUSDT_15m_2021-2025_vision.csv
│   │   └── BTCUSDT_funding_2021-2025.csv
│   └── db/                   # Databases
│       └── optuna.db         # Optimization history
│
├── run_state/                # 📁 Bot State (per-instance)
│   ├── eth/
│   │   └── state.json        # ETH bot state
│   └── btc/
│       └── state.json        # BTC bot state
│
├── docs/                     # 📁 Documentation
│   ├── Theory.md             # System specification
│   ├── OPTIMIZER_MANUAL.md   # Optimization guide
│   └── manual/               # Reference manual chapters
│
├── grafana/                  # 📁 Grafana Dashboard Exports
│   ├── ethbtc_bot_grafana_legacy.json
│   └── ethbtc_bot_grafana_live.json
│
├── scripts/                  # 📁 Shell Scripts
│   ├── check_parity.sh       # Backtest/live parity check
│   ├── migrate_volumes.sh    # Docker volume migration
│   └── purge_prometheus.sh   # Reset Prometheus data
│
├── results/                  # 📁 Backtest Output (gitignored)
├── logs/                     # 📁 Log Files (gitignored)
├── prometheus_data/          # 📁 Prometheus Data (gitignored)
└── grafana_data/             # 📁 Grafana Data (gitignored)
```

### 3. Key Files Quick Reference

| Need to... | Look at... |
|------------|------------|
| Run the bot | `live_executor.py` |
| Change trading parameters | `configs/*.json` |
| Add a new strategy | `core/meta_strategy.py` |
| Modify risk rules | `core/risk_manager.py` |
| Change position sizing | `core/position_sizer.py` |
| Add Prometheus metrics | `core/metrics.py` |
| Run optimization | `tools/run_optimization.py` |
| Download historical data | `tools/download_vision.py` |
| Debug API issues | `core/binance_adapter.py` or `core/futures_adapter.py` |

---

## 2.5 First Run Checklist

### 1. Concept & "The Why"

* **What it is:** A pre-flight checklist before running the bot for the first time.

* **Purpose:** Prevents common first-run failures and ensures the system is correctly configured.

* **Location:** N/A — this is a workflow.

### 2. The Checklist

```bash
# 1. ✅ Virtual environment activated (local) OR Docker running
source .venv/bin/activate  # Local
docker compose ps          # Docker

# 2. ✅ API keys configured
grep "TESTNET_KEY" .env    # Should show your keys (not empty)

# 3. ✅ State directories exist
ls -la run_state/eth run_state/btc

# 4. ✅ Historical data available (for backtesting)
ls -la data/raw/
# Should contain BTCUSDT_15m_*.csv files

# 5. ✅ Tests pass
python -m pytest tests/ -v --tb=short -x
# Should end with "X passed"

# 6. ✅ Dry run succeeds
python live_executor.py \
  --params configs/prod_meta_live.json \
  --mode dry \
  --symbol ETHBTC \
  --once

# 7. ✅ Prometheus metrics accessible (Docker only)
curl http://localhost:9109/metrics | head -20
# Should show Prometheus metrics
```

### 3. Real-World Use Case (The "Cookbook")

**Scenario:** First-time user wants to run bot on testnet.

**Complete Command Sequence:**
```bash
# 1. Clone and enter directory
git clone <repo> && cd ethbtc_bot_3

# 2. Create virtual environment
python3.11 -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cat > .env << 'EOF'
SPOT_TESTNET_KEY=your_testnet_key_here
SPOT_TESTNET_SECRET=your_testnet_secret_here
BINANCE_BASE_URL=https://testnet.binance.vision
MODE=testnet
LOGLEVEL=INFO
EOF

# 5. Create state directory
mkdir -p run_state/eth

# 6. Run on testnet (single bar, then exit)
python live_executor.py \
  --params configs/prod_eth_long_wfo_robust.json \
  --mode testnet \
  --symbol ETHBTC \
  --state run_state/eth/state.json \
  --once
```

**Expected Outcome:**
```
2026-01-05 10:20:00 [INFO] Bot starting...
2026-01-05 10:20:01 [INFO] Loaded config: prod_eth_long_wfo_robust.json
2026-01-05 10:20:02 [INFO] Fetched 500 bars for ETHBTC
2026-01-05 10:20:02 [INFO] Signal: target_w=0.33, current_w=0.00, delta_w=0.33
2026-01-05 10:20:02 [INFO] Mode=testnet, executing trade...
2026-01-05 10:20:03 [INFO] Order placed: BUY 0.5 ETH @ 0.03421
2026-01-05 10:20:05 [INFO] Order filled successfully
2026-01-05 10:20:05 [INFO] State saved to run_state/eth/state.json
2026-01-05 10:20:05 [INFO] --once flag set, exiting.
```

### 4. Troubleshooting & Edge Cases

* **What can go wrong:**
  - API keys not set → Connection refused
  - Wrong `MODE` value → No trades executed
  - State file permission issues → Cannot save state

* **Error Messages:**

  ```
  binance.error.ClientError: APIError(code=-1021): Timestamp for this request is outside of the recvWindow
  ```
  **Cause:** System clock is out of sync.
  **Fix:** 
  ```bash
  # Linux
  sudo timedatectl set-ntp on
  
  # macOS
  sudo sntp -sS time.apple.com
  ```

  ```
  KeyError: 'BINANCE_KEY'
  ```
  **Cause:** `.env` file not loaded or variable not set.
  **Fix:** Ensure `.env` exists and contains `BINANCE_KEY=...`

---

*Previous Chapter: [Chapter 1: Introduction & Overview](./CHAPTER_01_INTRODUCTION.md)*  
*Next Chapter: [Chapter 3: Configuration Reference](./CHAPTER_03_CONFIGURATION.md)*
