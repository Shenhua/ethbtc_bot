# BTC Algorithmic Trading Agent: Practical Specification

This document provides a realistic, implementable specification for a BTC algorithmic trading agent designed for solo or small-team operation. It prioritizes robustness, capital preservation, and achievable infrastructure over theoretical optimality.

---

## Part 1: Core Objectives and Realistic Targets

### I. Project Philosophy

The goal is to develop a **robust, profitable Bitcoin trading agent** that prioritizes:
1. **Capital Preservation** — Surviving drawdowns is more important than maximizing returns
2. **Simplicity** — Fewer moving parts means fewer failure modes
3. **Adaptability** — Markets change; the system must adapt without constant manual intervention
4. **Operational Reliability** — Uptime and monitoring matter more than microsecond latency

### II. Target Metrics (Realistic)

| Metric                     | Target Specification                               | Rationale                                                    |
| -------------------------- | -------------------------------------------------- | ------------------------------------------------------------ |
| **Sharpe Ratio**           | **1.0 – 2.0**                                      | A Sharpe > 1.5 is excellent for systematic trading. Targets above 2.5 are rarely sustainable in live conditions. |
| **Annual Return**          | **20–50%** (depending on market regime)            | Outperforming buy-and-hold over full market cycles is the goal, not maximizing bull-market gains. |
| **Maximum Drawdown (MDD)** | **< 15%** (hard stop at 20%)                       | Capital preservation is paramount. The system should pause trading before catastrophic loss. |
| **Win Rate**               | **> 45%** with positive expectancy                 | Win rate alone is meaningless; what matters is (Avg Win × Win Rate) > (Avg Loss × Loss Rate). |
| **Recovery Factor**        | **> 2.0** (Net Profit / Max Drawdown)              | Measures ability to recover from drawdowns. A factor > 2 indicates healthy risk-reward. |
| **Execution Quality**      | Slippage < 0.1% per trade; > 80% maker fills       | Limit orders and patient execution preserve alpha. |

### III. Non-Goals (Explicitly Out of Scope)

The following are **intentionally excluded** as they require institutional infrastructure or provide marginal benefit:

- ❌ **High-Frequency Trading (< 100ms latency)** — Not necessary for swing/trend strategies
- ❌ **Cross-Exchange Arbitrage / Smart Order Routing** — Requires complex infrastructure and legal considerations
- ❌ **Deep Learning / Transformer Models** — Adds latency, cost, and fragility vs. simpler approaches
- ❌ **Reinforcement Learning (DQN, PPO)** — Multi-year research project with high overfit risk
- ❌ **Multimodal Sentiment Analysis (TikTok, etc.)** — Expensive, noisy, and unreliable for alpha
- ❌ **Full L2/L3 Historical Order Book Data** — Costs $1,000s/month; marginal benefit for swing trading

---

## Part 2: System Architecture

### Document 2.1: Infrastructure and Deployment

| Component                 | Specification                                                | Rationale                                                    |
| ------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Core Architecture**     | Event-driven Python application with modular components (Signal → Position Sizing → Execution → Monitoring). | Separation of concerns enables independent testing, debugging, and enhancement of each layer. |
| **Processing Paradigm**   | Single-threaded async/await with periodic batch processing. Avoid multi-threading unless necessary. | Reduces complexity and race conditions. Crypto markets don't require HFT-level parallelism. |
| **Target Latency**        | **< 5 seconds** from signal to order submission is acceptable. | Alpha in swing/trend trading comes from signal quality, not execution speed. |
| **Deployment Platform**   | Docker containers with health checks, auto-restart, and persistent state storage. | Ensures uptime and reproducibility. Containers survive host restarts and enable easy migration. |
| **Monitoring Stack**      | Prometheus for metrics collection + Grafana for visualization + Alertmanager for notifications. | Industry-standard observability. Enables real-time tracking and historical analysis. |
| **State Persistence**     | JSON state files with atomic writes and backup rotation. | Simple, debuggable, and survives process restarts. |
| **Configuration**         | External JSON config files with validation on startup. | Allows parameter changes without code modification. Enables multiple bot instances with different configs. |

---

### Document 2.2: Data Pipeline and Signal Generation

| Component                   | Specification                                                | Rationale                                                    |
| --------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Primary Data Source**     | Exchange WebSocket for real-time OHLCV and order book snapshots; REST API for historical backfills. | WebSocket provides low-latency updates; REST ensures data completeness after reconnection. |
| **Historical Data**         | OHLCV at 1-minute resolution, stored locally for backtesting. Minimum 3 years of data covering bull, bear, and sideways regimes. | Enables Walk-Forward Optimization and out-of-sample validation across all market conditions. |
| **On-Chain Data (Optional)**| Exchange flow metrics (net inflows/outflows) from Glassnode or CryptoQuant API. | Provides uncorrelated signal: large outflows often precede accumulation phases. **Cost: ~$40-300/month.** |
| **Signal Architecture**     | Rule-based indicators with optimizable parameters (e.g., moving average crossovers, trend filters, volatility regimes). | Interpretable, debuggable, and less prone to overfitting than black-box ML models. |
| **Regime Detection**        | Classify market state as Trending, Mean-Reverting, or High-Volatility using rolling metrics (ADX, Bollinger Band width, ATR percentile). | Different regimes require different strategies; knowing the regime prevents misapplication. |
| **Stationarity Validation** | Apply Augmented Dickey-Fuller (ADF) test to verify mean-reversion assumptions when applicable. | Prevents trading mean-reversion signals on non-stationary (trending) data. |
| **Volatility Forecasting**  | GARCH(1,1) or EWMA volatility estimation for position sizing and risk scaling. | Captures volatility clustering (high-vol follows high-vol) for adaptive risk management. |

---

### Document 2.3: Position Sizing and Capital Allocation

| Component                   | Specification                                                | Rationale                                                    |
| --------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Sizing Algorithm**        | **Kelly Criterion** with fractional scaling (typically 0.25× to 0.5× Kelly). | Kelly maximizes geometric growth but is aggressive. Fractional Kelly reduces variance and drawdowns. |
| **Dynamic Kelly**           | Calculate Kelly parameters (win rate, avg win/loss) using a rolling window of recent trades (e.g., last 50-100 trades). | Adapts to changing market conditions without requiring re-optimization. |
| **Volatility Scaling**      | Position size inversely proportional to current ATR or GARCH volatility. Larger positions in calm markets; smaller in volatile ones. | The single most important risk control. Volatility-adjusted sizing accounts for ~90% of portfolio performance variability. |
| **Maximum Position Size**   | Hard cap at a percentage of portfolio (e.g., 50% for aggressive, 25% for conservative). | Prevents overconcentration regardless of what Kelly suggests. |
| **Minimum Trade Size**      | Skip trades where calculated size is below exchange minimum or below a practical threshold (e.g., $50). | Avoids dust trades that incur fees without meaningful profit potential. |

---

### Document 2.4: Risk Management Framework

| Component                   | Specification                                                | Rationale                                                    |
| --------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Drawdown Control**        | **Phoenix Protocol**: Pause trading when drawdown exceeds threshold (e.g., 10%). Auto-resume when equity recovers or after cooling-off period. | Prevents catastrophic loss and emotional decision-making during adverse conditions. |
| **Per-Trade Risk Limit**    | Maximum loss per trade capped at 1-2% of portfolio.          | Ensures no single trade can cause significant damage. |
| **Daily Loss Limit**        | Pause trading for the day if cumulative daily loss exceeds threshold (e.g., 3%). | Prevents compounding losses during bad days. |
| **Tail Risk Metric**        | Calculate **Conditional Value-at-Risk (CVaR)** at 95% confidence level. | CVaR (Expected Shortfall) measures average loss in worst 5% of scenarios—superior to VaR for fat-tailed crypto returns. |
| **Correlation Monitoring**  | Track rolling 30-day correlation between strategy returns and BTC buy-and-hold. | High correlation means the strategy isn't adding value beyond simple holding. |
| **Stress Testing**          | Monte Carlo simulations with bootstrapped historical returns to estimate drawdown distributions and ruin probability. | Quantifies probability of hitting various drawdown levels; informs position sizing and leverage decisions. |
| **Regime-Aware Risk**       | Reduce position size or pause trading when volatility exceeds 2× historical average. | Extreme volatility regimes often violate model assumptions and increase execution risk. |

---

### Document 2.5: Execution Layer

| Component                | Specification                                                | Rationale                                                    |
| ------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Order Type**           | **Limit orders with post-only flag** as default. Market orders only for urgent exits. | Maker orders receive better fees (often rebates) and avoid slippage. |
| **Fill Confirmation**    | Polling-based fill verification with timeout and retry logic. | Ensures positions are correctly opened/closed before updating state. |
| **Order Timeout**        | Cancel and re-price limit orders if unfilled after configurable duration (e.g., 30-60 seconds). | Prevents stale orders from sitting indefinitely in a moving market. |
| **Slippage Tracking**    | Log and monitor expected vs. actual fill price for every trade. | Enables detection of execution quality degradation over time. |
| **Fee Tracking**         | Record all fees paid; include in P&L calculations and backtest simulations. | Fees compound significantly; accurate tracking prevents P&L surprises. |
| **Rate Limiting**        | Respect exchange rate limits with exponential backoff on 429 errors. | Prevents API bans and ensures reliable connectivity. |
| **Reconnection Logic**   | Auto-reconnect WebSocket with state recovery on disconnect. | Network issues are inevitable; graceful recovery maintains uptime. |

---

### Document 2.6: Backtesting and Optimization

| Component                | Specification                                                | Rationale                                                    |
| ------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Backtesting Engine**   | Event-driven or vectorized backtest with realistic fill assumptions (slippage, fees, latency). | Simulation must approximate live conditions to produce meaningful results. |
| **Walk-Forward Optimization (WFO)** | Rolling window optimization: optimize on N months, validate on M months, roll forward. | Prevents overfitting by always validating on out-of-sample data. |
| **Robustness Testing**   | Parameter sensitivity analysis: profitable across a range of parameter values, not just optimal point. | Fragile parameters indicate curve-fitting; robust parameters indicate genuine edge. |
| **Regime Validation**    | Separate performance metrics for bull, bear, and sideways markets. | A strategy that only works in bull markets provides no edge—it's just leveraged beta. |
| **Transaction Costs**    | Include realistic fees (maker/taker), funding rates (for perpetuals), and slippage in all backtests. | Strategies that look profitable before costs often become losers after costs. |
| **Benchmark Comparison** | Compare strategy returns to: (1) Buy-and-Hold BTC, (2) Risk-free rate, (3) Simple moving average baseline. | Provides context for whether the strategy adds value. |

---

### Document 2.7: Monitoring and Alerting

| Component                | Specification                                                | Rationale                                                    |
| ------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Real-Time Metrics**    | Expose via Prometheus: portfolio value, position size, unrealized PnL, drawdown, signal state, execution latency. | Enables real-time dashboards and historical trend analysis. |
| **Health Checks**        | Regular heartbeat metrics; alert if bot stops reporting.     | Detects crashes, freezes, or network issues immediately. |
| **Drawdown Alerts**      | Notify (email, Telegram, etc.) when drawdown exceeds warning thresholds (e.g., 5%, 10%). | Enables human oversight and intervention if needed. |
| **Execution Alerts**     | Alert on: failed orders, repeated fill timeout, slippage exceeding threshold. | Catches execution issues before they compound. |
| **Daily Summary**        | Automated daily report: trades executed, P&L, current position, key metrics. | Keeps operator informed without requiring constant dashboard watching. |
| **Anomaly Detection**    | Simple threshold-based alerts (not ML): unusual trade frequency, unexpected position size, API errors. | Catches bugs or unexpected behavior. Complex ML anomaly detection is overkill. |

---

## Part 3: Implementation Priorities

### Phase 1: Core Foundation ✅ (Completed)
- [x] Signal generation with configurable parameters
- [x] Position sizing with Kelly Criterion
- [x] Basic risk limits (max position, drawdown control)
- [x] Limit order execution with post-only
- [x] State persistence and recovery
- [x] Prometheus/Grafana monitoring

### Phase 2: Robustness Hardening ✅ (Completed)
- [x] Walk-Forward Optimization framework
- [x] Fill confirmation with polling
- [x] Dynamic Kelly with rolling statistics
- [x] Decimal precision for monetary calculations
- [x] Phoenix Protocol for drawdown protection
- [x] Comprehensive test coverage

### Phase 3: Advanced Risk Metrics (Recommended Next)
- [ ] Implement CVaR (Conditional Value-at-Risk) calculation
- [ ] Add Monte Carlo stress testing for drawdown estimation
- [ ] Track rolling Sharpe and Sortino ratios in monitoring
- [ ] Implement regime-aware position scaling (reduce size in high-volatility regimes)

### Phase 4: Execution Optimization (Optional)
- [ ] TWAP execution for large orders (if trading significant size)
- [ ] Slippage analysis and execution quality tracking dashboard
- [ ] Adaptive order timeout based on market conditions

### Phase 5: Alpha Enhancement (Optional, Requires Subscription)
- [ ] Integrate on-chain exchange flow data (Glassnode/CryptoQuant)
- [ ] Add funding rate as signal input for perpetual futures
- [ ] Multi-timeframe signal confirmation

---

## Part 4: What NOT to Build

The following features are explicitly **out of scope** to avoid complexity creep:

| Feature                           | Why Not                                                      |
| --------------------------------- | ------------------------------------------------------------ |
| **LLM/Transformer-based Signals** | Adds latency, cost, and fragility. Simpler indicators perform comparably with 1/100th the complexity. |
| **Reinforcement Learning (DQN)** | Requires years of research to get right. Prone to overfitting and catastrophic failure modes. |
| **Cross-Exchange Arbitrage**      | Requires capital on multiple exchanges, complex rebalancing, and legal considerations. |
| **Social Sentiment Analysis**     | Twitter/X API is expensive; sentiment signals are noisy and unreliable. |
| **Ultra-Low Latency (<100ms)**    | Only matters for HFT. For swing trading, signal quality beats execution speed. |
| **Custom Neural Network Models**  | Require massive data, compute, and expertise to outperform simple baselines. |

---

## Appendix A: Key Formulas

### Kelly Criterion
```
f* = (p × b - q) / b

where:
  f* = optimal fraction of capital to bet
  p  = probability of winning (win rate)
  q  = probability of losing (1 - p)
  b  = ratio of average win to average loss
```

### Conditional Value-at-Risk (CVaR)
```
CVaR_α = E[Loss | Loss > VaR_α]

Interpretation: Average loss in the worst (1-α)% of scenarios.
Example: CVaR_95 is the average loss in the worst 5% of cases.
```

### Sharpe Ratio
```
Sharpe = (R_p - R_f) / σ_p

where:
  R_p = portfolio return
  R_f = risk-free rate (often assumed 0 for crypto)
  σ_p = portfolio standard deviation
```

### Volatility-Scaled Position Size
```
Position = Base_Size × (Target_Volatility / Current_Volatility)

where:
  Target_Volatility = desired portfolio volatility (e.g., 15% annualized)
  Current_Volatility = ATR-based or GARCH-estimated volatility
```

---

## Appendix B: Recommended Reading

1. **Position Sizing**: Van Tharp, "Trade Your Way to Financial Freedom"
2. **Systematic Trading**: Robert Carver, "Systematic Trading"
3. **Risk Management**: Nassim Taleb, "The Black Swan" (for understanding tail risk)
4. **Quant Fundamentals**: Ernest Chan, "Quantitative Trading"
5. **Backtesting Pitfalls**: David Aronson, "Evidence-Based Technical Analysis"
