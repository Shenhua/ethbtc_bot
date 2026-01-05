# Appendix B: Glossary

> **Purpose:** Definitions of key terms, abbreviations, and concepts used throughout this manual.

---

## Trading & Strategy Terms

| Term | Definition |
|------|------------|
| **ADX** | Average Directional Index. Measures trend strength (0-100). Values > 25 indicate strong trend. |
| **Alpha** | Excess return above benchmark (HODL). Positive alpha = outperformance. |
| **Backtest** | Simulating strategy performance on historical data. |
| **Basis Points (bps)** | 1/100th of a percent. 100 bps = 1%. |
| **Calmar Ratio** | CAGR ÷ Maximum Drawdown. Risk-adjusted return measure. |
| **CAGR** | Compound Annual Growth Rate. Annualized return. |
| **Cooldown** | Minimum time between trades to prevent rapid flipping. |
| **CVaR** | Conditional Value at Risk. Expected loss in worst N% of scenarios. |
| **Drawdown (DD)** | Peak-to-trough decline in portfolio value. |
| **EMA** | Exponential Moving Average. Recent prices weighted more heavily. |
| **Flip Band** | Deviation threshold from trend that triggers entry/exit. |
| **Funding Rate** | Periodic payment between long/short positions in perpetual futures. |
| **Gate** | Trading filter based on long-term trend direction. |
| **Half-Kelly** | Using 50% of Kelly Criterion recommendation for safety. |
| **HODL** | "Hold On for Dear Life." Buy-and-hold benchmark. |
| **HWM** | High Water Mark. Peak portfolio value for drawdown calculation. |
| **Kelly Criterion** | Optimal bet sizing formula based on win rate and payoffs. |
| **Limit Maker** | Post-only order that guarantees maker fee. |
| **Long** | Buying asset expecting price increase. |
| **Maker** | Order that adds liquidity to order book. Lower fees. |
| **Mean Reversion** | Strategy betting price returns to moving average. |
| **Meta Strategy** | Ensemble strategy switching between MR and Trend based on regime. |
| **OHLCV** | Open, High, Low, Close, Volume. Standard candle data. |
| **OOS** | Out-of-Sample. Data not used for optimization. |
| **Phoenix Protocol** | Automatic trading resumption after max DD cooldown. |
| **Profit Factor** | Gross profit ÷ Gross loss. > 1 = profitable. |
| **Regime** | Market condition (trending vs. ranging). |
| **ROC** | Rate of Change. Percentage price change over period. |
| **RSI** | Relative Strength Index. Momentum oscillator (0-100). |
| **Sharpe Ratio** | Risk-adjusted return. (Return - Rf) ÷ StdDev. |
| **Short** | Selling asset expecting price decrease. |
| **Slippage** | Difference between expected and actual fill price. |
| **SMA** | Simple Moving Average. Equal weight to all prices. |
| **Sortino Ratio** | Like Sharpe but only penalizes downside volatility. |
| **Spread** | Difference between best bid and ask prices. |
| **Step Allocation** | Fraction of capital traded per signal. |
| **Taker** | Order that removes liquidity. Higher fees. |
| **Trend Following** | Strategy trading in direction of prevailing trend. |
| **TTL** | Time To Live. Order expiration period. |
| **VaR** | Value at Risk. Maximum expected loss at confidence level. |
| **Volatility** | Annualized standard deviation of returns. |
| **WFO** | Walk-Forward Optimization. Rolling window parameter search. |
| **Win Rate** | Percentage of trades that are profitable. |

---

## Technical Terms

| Term | Definition |
|------|------------|
| **API** | Application Programming Interface. Exchange communication. |
| **Circuit Breaker** | Pattern that stops retries after too many failures. |
| **Container** | Isolated runtime environment (Docker). |
| **Exponential Backoff** | Increasing wait time between retries. |
| **Gauge** | Prometheus metric that can increase or decrease. |
| **gRPC** | High-performance RPC framework (Binance Future). |
| **JSON** | JavaScript Object Notation. Config/data format. |
| **Pydantic** | Python library for data validation. |
| **REST** | RESTful API. HTTP-based exchange communication. |
| **SQLite** | Lightweight embedded database. |
| **Structlog** | Python structured logging library. |
| **TPE** | Tree-structured Parzen Estimator. Optuna's sampler. |
| **WebSocket** | Persistent connection for real-time data. |

---

## Abbreviations

| Abbr | Full Term |
|------|-----------|
| **BNB** | Binance Coin |
| **BTC** | Bitcoin |
| **DD** | Drawdown |
| **ETH** | Ethereum |
| **HTF** | Higher Timeframe |
| **MR** | Mean Reversion |
| **PnL** | Profit and Loss |
| **USD** | US Dollar |
| **USDT** | Tether (stablecoin) |
| **UTC** | Coordinated Universal Time |

---

## File Extensions

| Extension | Description |
|-----------|-------------|
| `.json` | Configuration files |
| `.csv` | Historical data files |
| `.jsonl` | Trade log (JSON Lines) |
| `.py` | Python source code |
| `.md` | Markdown documentation |
| `.yml` | Docker Compose / Prometheus config |
| `.db` | SQLite database |

---

*Return to: [Table of Contents](./MASTER_TABLE_OF_CONTENTS.md)*
