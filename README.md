# ETH/BTC Accumulation Bot

This repository contains a modular, automated trading bot designed for the **ETH/BTC** pair (or other pairs) on Binance. It supports both **Spot** (Accumulation) and **Futures** (USDS-M) trading, with advanced features like regime switching, walk-forward optimization, and risk management.

## 🚀 Key Features

*   **Dual-Engine Strategy**:
    *   **Mean Reversion**: Uses dynamic Bollinger Bands and volatility scaling for chopping markets.
    *   **Trend Following**: Uses moving average crossovers (EMA/SMA) for trending markets.
    *   **Meta-Strategy**: Automatically switches between Mean Reversion and Trend engines based on the **ADX** (Average Directional Index) regime score.
*   **Execution**:
    *   **Maker Chase**: Tries to post limit orders at the best bid/ask to save on fees, with Taker fallback.
    *   **Smart Risk Management**: Enforces daily loss limits and maximum drawdown thresholds.
    *   **Phoenix Protocol**: Automatically halts trading after a major drawdown and resumes only when the market regime becomes favorable.
*   **Infrastructure**:
    *   **Prometheus/Grafana Integration**: Exposes real-time metrics for visualization.
    *   **Story Logging**: Generates a human-readable narrative log (`story_ethbtc.txt`) of all actions.
    *   **Optimization**: Includes tools for Walk-Forward Optimization (WFO) using **Optuna**.

---

## 🛠️ Setup

### Prerequisites
*   Python 3.9+
*   Binance Account (Spot or Futures)
*   Docker (Optional, for Prometheus/Grafana)

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-repo/ethbtc-bot.git
    cd ethbtc-bot
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Environment Configuration**:
    Create a `.env` file in the root directory:
    ```bash
    # Binance Credentials
    BINANCE_KEY=your_api_key
    BINANCE_SECRET=your_api_secret

    # Optional: Futures-specific keys
    BINANCE_FUTURES_KEY=your_futures_key
    BINANCE_FUTURES_SECRET=your_futures_secret

    # Notifications (Optional)
    DISCORD_WEBHOOK_URL=your_discord_webhook
    TELEGRAM_TOKEN=your_telegram_bot_token
    TELEGRAM_CHAT_ID=your_chat_id

    # Configuration
    SYMBOL=ETHBTC
    MODE=live  # dry, testnet, or live
    LOGLEVEL=INFO
    ```

---

## 🏃 Usage

### 1. Live Trading
Run the main executor:
```bash
python live_executor.py --params configs/prod_meta_live.json --mode live --symbol ETHBTC
```
*   `--mode`: `dry` (paper trading), `testnet` (Binance Testnet), `live` (Real money).
*   `--params`: Path to the strategy configuration JSON.

### 2. Backtesting
Run a backtest on historical data:
```bash
python core/ethbtc_accum_bot.py backtest \
    --data data/ETHBTC_15m.csv \
    --config configs/prod_meta_live.json \
    --out results/backtest_results.csv
```

### 3. Optimization
Run parameter optimization using Optuna:
```bash
python tools/optimizer_cli.py \
    --data data/ETHBTC_15m.csv \
    --train-start 2023-01-01 --train-end 2023-06-30 \
    --test-start 2023-07-01 --test-end 2023-12-31 \
    --n-trials 100
```

### 4. Data Downloading
Download historical data from Binance Vision:
```bash
python tools/download_vision.py --symbol ETHBTC --start 2023-01-01 --end 2023-12-31 --intervals 15m
```

---

## 📂 Project Structure

*   **`core/`**: Core trading logic.
    *   `ethbtc_accum_bot.py`: Main entry for strategy logic.
    *   `meta_strategy.py`: Logic for switching between Trend and MR.
    *   `trend_strategy.py`: Trend following implementation.
    *   `live_executor.py`: Main loop for live trading execution.
    *   `binance_adapter.py` / `futures_adapter.py`: Exchange interfaces.
*   **`tools/`**: Utilities for data, optimization, and maintenance.
    *   `optimizer_cli.py`: Main optimization script.
    *   `download_vision.py`: Data downloader.
    *   `dust_sweeper.py`: Cleans up dust balances.
*   **`configs/`**: JSON configuration files.

---

## 📊 Monitoring

The bot starts a lightweight HTTP server (default port 9110) exposing status and metrics.
*   **Status**: `http://localhost:9110/status` (JSON state)
*   **Story**: `http://localhost:9109/story` (Human-readable log)
*   **Metrics**: `http://localhost:9109/metrics` (Prometheus)

---

## ⚠️ Disclaimer

This software is for educational purposes only. Do not trade with money you cannot afford to lose. The authors are not responsible for any financial losses.
