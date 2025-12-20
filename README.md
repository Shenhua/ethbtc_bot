# ethbtc_bot

A production-grade cryptocurrency trading bot for Binance, implementing Mean Reversion, Trend Following, and Meta Strategy with comprehensive risk management.

## Features

- **Multi-Strategy Support:** Mean Reversion, Trend Following, and Meta (ensemble) strategies
- **Dynamic Regime Switching:** ADX-based regime detection for automatic strategy selection
- **Risk Management:** Max drawdown limits, daily loss limits, and Phoenix Protocol for auto-recovery
- **Execution Modes:** Spot and Futures (USDS-M) with maker/taker order support
- **Observability:** Prometheus metrics, story logging, and real-time status endpoint
- **Resilience:** Circuit breaker pattern, API retry logic with exponential backoff

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set environment variables (see below)
export BINANCE_KEY="your_api_key"
export BINANCE_SECRET="your_api_secret"

# 3. Run in dry mode (no real trades)
python live_executor.py --params configs/prod_meta_live.json --mode dry --once

# 4. Run tests
python -m pytest tests/ -v
```

## Environment Variables

| Variable | Required | Description |
|:---------|:---------|:------------|
| `BINANCE_KEY` | Yes | Binance API key |
| `BINANCE_SECRET` | Yes | Binance API secret |
| `BINANCE_FUTURES_KEY` | Futures | Futures-specific API key (optional) |
| `BINANCE_FUTURES_SECRET` | Futures | Futures-specific API secret (optional) |
| `INSTANCE_NAME` | No | Prometheus metrics instance label |
| `STATE_FILE` | No | Path to state.json file |
| `LOGLEVEL` | No | Logging level (DEBUG, INFO, WARNING) |

## Configuration

Configs are JSON files in `configs/`. Key sections:

```json
{
  "fees": { "maker_fee": 0.0002, "taker_fee": 0.0004 },
  "strategy": { "strategy_type": "meta", "adx_threshold": 10.0 },
  "execution": { "interval": "15m", "exchange_type": "spot" },
  "risk": { "max_dd_frac": 0.20, "drawdown_reset_days": 7 }
}
```

## Project Structure

```
ethbtc_bot_3/
├── live_executor.py     # Main entry point
├── core/
│   ├── meta_strategy.py    # Ensemble strategy
│   ├── ethbtc_accum_bot.py # Mean Reversion + Backtester
│   ├── trend_strategy.py   # Trend Following
│   ├── risk_manager.py     # Risk management + Phoenix Protocol
│   ├── resilience.py       # Circuit breaker + retry logic
│   └── config_schema.py    # Pydantic config validation
├── tests/               # 92 unit tests
└── configs/             # JSON configuration files
```

## Risk Management

### Max Drawdown Protection
- **Fixed Basis:** Absolute BTC limit (e.g., max 0.2 BTC loss)
- **Dynamic:** Percentage of peak equity (e.g., 20% of HWM)

### Phoenix Protocol
Automatic recovery after max drawdown:
1. Halt trading when max DD hit
2. Wait for cooldown (e.g., 7 days)
3. Resume when regime score is favorable

## Testing

```bash
# Full test suite (92 tests)
python -m pytest tests/ -v

# Specific test file
python -m pytest tests/test_phoenix_protocol.py -v

# With coverage
python -m pytest tests/ --cov=core
```

## License

Private - All rights reserved.