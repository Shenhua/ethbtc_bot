- Gotcha — you want a *product-level* spec, not a metric-by-metric wiring diagram. Something that says **what** we want to see and **roughly how**, but leaves the engineer responsible for:

  

  - Mapping to real metric names
  - Checking what already exists
  - Deciding what extra metrics/instrumentation are needed

  

  

  Here’s a spec you can drop in Notion / Jira.

  

  ------

  

  

  # **Hybrid Spot & Futures – Multi-Bot Grafana Dashboard**

  

  

  **Specification (High-Level, Dev-Driven Wiring)**

  

  

  ## **1. Purpose & Scope**

  

  

  A single Grafana dashboard that:

  

  - Monitors **multiple trading bots** across **Spot and Futures** accounts.

  - Gives a **one-glance view** of:

    

    - Total equity
    - PnL (today, recent)
    - Exposure/leverage
    - Health (bots & exchanges)

    

  - Allows **deep dive** into:

    

    - Per-bot performance
    - Risk (per symbol, per venue)
    - Execution quality (slippage, errors)

    

  - Works across **environments** (prod / paper / dev).

  

  

  The dashboard should be driven by **labels/tags** like:

  

  - env (prod, paper, …)
  - venue (e.g. binance_spot, binance_futures, …)
  - account (subaccount or wallet ID)
  - bot (bot ID / name)
  - symbol (e.g. ETHBTC, BTCUSDT)

  

  

  **Developer note:**

  For each panel below, you’re expected to:

  

  - Reuse existing metrics where possible
  - Identify missing metrics and add instrumentation in the bot / recon tools
  - Prefer pre-aggregated/recorded series if queries get heavy

  

  

  ------

  

  

  ## **2. Layout Overview**

  

  

  Recommended structure, top → bottom:

  

  1. **Filter bar (templating variables)**
  2. **Top Summary Row:** key KPIs & status (stat/gauge panels)
  3. **Bot Fleet Overview:** table, one row per bot
  4. **Equity & PnL:** account and per-bot curves
  5. **Risk & Exposure:** positions, leverage, liquidation distance
  6. **Execution:** trades, slippage, fees/funding
  7. **Health / Infra:** heartbeats, errors, resource usage
  8. **Per-Bot Deep Dive:** everything filtered by a bot variable

  

  

  ------

  

  

  ## **3. Dashboard Variables (Filters)**

  

  

  Create Grafana variables (no strict metric names; use whatever the data source provides):

  

  - **env**

    

    - Description: Environment (prod, paper, dev)
    - Used to filter *all* panels.

    

  - **venue**

    

    - Examples: binance_spot, binance_futures, …
    - Used for exchange-specific views (margin, funding, rate limits).

    

  - **account**

    

    - Map to exchange account / subaccount identifiers.

    

  - **bot**

    

    - Human-readable bot ID / strategy name.

    

  - **symbol**

    

    - Trading pair, from your market data / positions streams.

    

  

  

  > **Dev task:**

  

  - > Make sure every metric used by this dashboard has env, and ideally venue, account, bot, symbol labels where applicable.

  - > Some panels will not render correctly if these labels are missing.

  

  

  ------

  

  

  ## **4. Top Summary Row (At-a-Glance)**

  

  

  Each card here is a **Stat** or **Gauge** panel. No hard metric names, just concepts.

  

  

  ### **4.1 Total Account Equity (Spot + Futures)**

  

  

  - **Panel type:** Stat (with tiny sparkline if easy)

  - **Value:** Current account equity in main quote currency (e.g. USDT or BTC).

  - **Scope:** Sum across selected account + venue in current env.

  - **Tooltip copy:**

    > Total mark-to-market value (Spot + Futures) for the selected environment and accounts.

  - **Dev notes:**

    

    - Reuse existing equity metric(s) from reconciliation / balance polling.
    - If Spot & Futures are separate, aggregate them in query.

    

  

  

  ------

  

  

  ### **4.2 24h PnL**

  

  

  - **Panel type:** Stat

  - **Value:** Change in equity over the last 24 hours (realized + unrealized).

  - **Tooltip copy:**

    > Change in total equity over the last 24 hours, including realized and unrealized PnL, fees and funding.

  - **Dev notes:**

    

    - Either:

      

      - Provide a dedicated “24h PnL” metric, **or**
      - Derive from equity timeseries via Grafana (value now − value 24h ago).

      

    - Ensure time zone definition is clear (UTC).

    

  

  

  ------

  

  

  ### **4.3 Today’s PnL % (Daily Return)**

  

  

  - **Panel type:** Stat

  - **Value:** (equity_today / equity_yesterday_close - 1) * 100.

  - **Tooltip copy:**

    > Percentage change in equity since yesterday’s close. Normalizes PnL by account size.

  - **Dev notes:**

    

    - You can either:

      

      - Emit an explicit “daily return” metric, or
      - Let Grafana compute it via transformations / recording rules.

      

    

  

  

  ------

  

  

  ### **4.4 Current Exposure / Leverage**

  

  

  - **Panel type:** Gauge

  - **Value:** Gross or net exposure vs equity (e.g. gross_notional / equity).

  - **Tooltip copy:**

    > Current exposure relative to equity, across Spot and Futures. Higher leverage means higher liquidation risk.

  - **Dev notes:**

    

    - Need a notional exposure metric per symbol:

      

      - Spot: size × price
      - Futures: contract size × price

      

    - Aggregate to account level, divide by equity.

    

  

  

  ------

  

  

  ### **4.5 Margin Utilization (Futures)**

  

  

  - **Panel type:** Gauge

  - **Value:** Used margin / total margin % on futures venues.

  - **Tooltip copy:**

    > Portion of available margin currently in use on futures accounts.

  - **Dev notes:**

    

    - Hook into exchange margin stats (from API polling or existing exporter).

    

  

  

  ------

  

  

  ### **4.6 Bot Status Overview**

  

  

  - **Panel type:** Stat

  - **Value:** “X running / Y total (Z in error)”.

  - **Tooltip copy:**

    > Status of all bots in the selected environment (running, stopped, error).

  - **Dev notes:**

    

    - Ensure each bot periodically emits a status/heartbeat metric or log that can be scraped.

    

  

  

  ------

  

  

  ### **4.7 PnL Drift (Recon Check)**

  

  

  - **Panel type:** Stat (colored)

  - **Value:** % difference between exchange wallet equity and internal bot equity.

  - **Tooltip copy:**

    > Difference between exchange wallet equity and the bot’s internal equity estimate. Large drift may indicate missing fills or accounting bugs.

  - **Dev notes:**

    

    - Feed this from your reconcile_pnl.py process as a single numeric series per account.
    - Align with whatever threshold you use for Discord alerts (e.g. >1%).

    

  

  

  ------

  

  

  ## **5. Bot Fleet Overview (Table)**

  

  

  A single **Table** panel summarizing all bots.

  

  **Each row = one bot instance** (bot+symbol+venue+account).

  

  Suggested columns (no metric names, just what they represent):

  

  1. **Bot Name**

     

     - Text label (ID / strategy name).
     - Tooltip: “Logical name of the bot/strategy instance.”

     

  2. **Venue & Product**

     

     - Text fields such as Binance / Spot, Binance / Futures.
     - Tooltip: “Exchange & product type (spot, futures, or hybrid).”

     

  3. **Symbol**

     

     - Trading pair.
     - Tooltip: “Primary market the bot trades.”

     

  4. **Status**

     

     - e.g. Running / Stopped / Error.
     - Tooltip: “Latest reported health status from bot heartbeat.”
     - Dev: derive from heartbeat age, or explicit state metric.

     

  5. **Net Position – Spot**

     

     - Value in base units (e.g. ETH).
     - Tooltip: “Current net spot position in base units.”

     

  6. **Net Position – Futures**

     

     - Value in base units (positive = long, negative = short).
     - Tooltip: “Current net futures position in base units.”

     

  7. **Net Delta (Combined)**

     

     - Net effective exposure in quote terms (spot + hedge).
     - Tooltip: “Mark-to-market delta exposure, combining spot and futures.”

     

  8. **Today PnL**

     

     - PnL since start of day.
     - Tooltip: “Bot’s PnL since midnight UTC (realized + unrealized).”

     

  9. **MTD PnL**

     

     - Month-to-date PnL.
     - Tooltip: “Cumulative PnL for the current calendar month.”

     

  10. **Win Rate (Recent)**

      

      - Win rate over recent N trades or last X days.
      - Tooltip: “Fraction of profitable trades in the recent window.”

      

  11. **Avg Reward / Risk (Recent)**

      

      - Reward:Risk ratio, average profit / average loss.
      - Tooltip: “Average reward-to-risk ratio on recent trades.”

      

  12. **Last Trade Time**

      

      - Timestamp.
      - Tooltip: “Time of the most recent trade for this bot.”

      

  13. **Error Count (24h)**

      

      - Number of critical errors in logs.
      - Tooltip: “Number of errors reported by this bot in the last 24 hours.”

      

  

  

  > **Dev task:**

  

  - > Use whatever metrics / logs you already have to populate as many columns as possible.

  - > If a column is not currently measurable (e.g. R:R), identify what needs to be logged (per-trade PnL, entry/exit) and instrument accordingly.

  

  

  ------

  

  

  ## **6. Equity & PnL Section**

  

  

  

  ### **6.1 Total Equity Over Time**

  

  

  - **Panel type:** Time series

  - **Values:** Account equity over time, with separate lines per venue (Spot / Futures) or per account.

  - **Tooltip copy:**

    > Total equity over time, optionally broken down by Spot vs Futures.

  - **Dev notes:**

    

    - Reuse same equity metric as top-row; just show historical.

    

  

  

  ------

  

  

  ### **6.2 PnL Breakdown**

  

  

  - **Panel type:** Stacked area or multiple line chart

  - **Components:**

    

    - Realized PnL
    - Unrealized PnL
    - Fees (negative)
    - Funding (positive/negative)

    

  - **Tooltip copy:**

    > Decomposition of performance into realized PnL, unrealized PnL, trading fees, and funding.

  - **Dev notes:**

    

    - If only total PnL is available today, consider adding:

      

      - per-trade fee tracking
      - funding per funding interval.

      

    

  

  

  ------

  

  

  ### **6.3 Daily PnL Bars**

  

  

  - **Panel type:** Bar chart by day

  - **Values:** PnL per calendar day.

  - **Tooltip copy:**

    > End-of-day PnL for each day in the selected range.

  - **Dev notes:**

    

    - Either emit a “daily PnL” metric or let Grafana sum/aggregate intraday PnL per day.

    

  

  

  ------

  

  

  ## **7. Risk & Exposure Section**

  

  

  

  ### **7.1 Exposure by Symbol (Current Snapshot)**

  

  

  - **Panel type:** Horizontal bar chart

  - **Values:** Net exposure per symbol, in quote terms.

  - **Tooltip copy:**

    > Net exposure per symbol (Spot + Futures), converted to quote currency.

  - **Dev notes:**

    

    - Need: position sizes, prices, and a consistent quote currency.

    

  

  

  ------

  

  

  ### **7.2 Gross vs Net Exposure Over Time**

  

  

  - **Panel type:** Time series

  - **Lines:**

    

    - Gross exposure (sum of absolute notionals)
    - Net exposure (signed)

    

  - **Tooltip copy:**

    > Evolution of gross vs net exposure over time for the selected environment.

  - **Dev notes:**

    

    - Derive from position snapshots; could be pre-aggregated in your metrics pipeline.

    

  

  

  ------

  

  

  ### **7.3 Drawdown Curve**

  

  

  - **Panel type:** Time series

  - **Values:** Running drawdown (e.g. from rolling max equity).

  - **Tooltip copy:**

    > Drawdown from recent equity highs. Helps gauge risk realized.

  - **Dev notes:**

    

    - Either compute in prometheus (recording rule) or emit from your backtester/bot.

    

  

  

  ------

  

  

  ### **7.4 Liquidation Distance (Futures)**

  

  

  - **Panel type:** Histogram or table

  - **Values:** For each futures position, percentage distance from current price to liquidation price.

  - **Tooltip copy:**

    > How far each futures position is from its liquidation level.

  - **Dev notes:**

    

    - Requires access to liquidation prices from exchange or your own risk calc.

    

  

  

  ------

  

  

  ## **8. Execution & Microstructure Section**

  

  

  

  ### **8.1 Trade Activity Over Time**

  

  

  - **Panel type:** Time series

  - **Values:** Trades per interval (count) and traded notional.

  - **Tooltip copy:**

    > Number of trades and traded volume over time; filterable by bot and symbol.

  - **Dev notes:**

    

    - Requires per-trade logging or exchange fills feed metrics.

    

  

  

  ------

  

  

  ### **8.2 Slippage Distribution**

  

  

  - **Panel type:** Histogram

  - **Values:** Execution slippage in bps per trade.

  - **Tooltip copy:**

    > Distribution of execution slippage relative to mid/mark prices.

  - **Dev notes:**

    

    - Requires your execution layer to log:

      

      - reference price at decision time
      - fill prices / quantities.

      

    

  

  

  ------

  

  

  ### **8.3 Fees & Funding by Bot/Symbol**

  

  

  - **Panel type:** Table

  - **Values:** Cumulative fees and funding per bot/symbol.

  - **Tooltip copy:**

    > Total fees and funding impact per strategy and market.

  - **Dev notes:**

    

    - Use whatever fee & funding fields the exchange gives; accumulate them per bot & symbol.

    

  

  

  ------

  

  

  ## **9. Health & Infrastructure Section**

  

  

  

  ### **9.1 Bot Heartbeats**

  

  

  - **Panel type:** Table or status map

  - **Values:** Age of last heartbeat per bot.

  - **Tooltip copy:**

    > Time since the bot last reported a healthy heartbeat.

  - **Dev notes:**

    

    - Each bot should periodically emit a simple “heartbeat timestamp / counter” metric.

    

  

  

  ------

  

  

  ### **9.2 Errors & Warnings**

  

  

  - **Panel type:** Table or stat(s)

  - **Values:**

    

    - Error count per bot (e.g. last 24h).
    - Optionally, last error message (via logging integration).

    

  - **Tooltip copy:**

    > Recent error activity per bot. Spikes may indicate connectivity or logic issues.

  - **Dev notes:**

    

    - Wire log aggregation (Loki/ELK) or increment counters on errors.

    

  

  

  ------

  

  

  ### **9.3 Resource Usage**

  

  

  - **Panel type:** Time series

  - **Values:** CPU and memory usage per bot/container.

  - **Tooltip copy:**

    > System-level resource usage for each bot process or container.

  - **Dev notes:**

    

    - Reuse node exporter / cAdvisor metrics with appropriate labels.

    

  

  

  ------

  

  

  ## **10. Per-Bot Deep Dive Section**

  

  

  All panels here are **filtered by bot (and often symbol)**.

  

  Suggested panels:

  

  1. **Bot Equity & PnL Curve**

     

     - Bot-specific equity and cumulative PnL.

     

  2. **Position Timeline (Spot & Futures)**

     

     - Spot size, futures size, effective leverage over time.

     

  3. **Price Chart with Trade Markers**

     

     - Market price with buy/sell markers (size, side).

     

  4. **Kelly / Sizing Diagnostics (if you implement advanced sizing)**

     

     - Rolling win-rate, R:R, Kelly fraction estimate, actual step allocation used.

     

  5. **Regime / Strategy Mode (if meta strategy)**

     

     - Regime score / ADX and active mode (trend vs mean reversion).

     

  

  

  > **Dev task:**

  

  - > Start simple (equity, positions, trades) and expand with Kelly/regime once metrics exist.

  - > Use Grafana’s “link” feature to jump from fleet overview table → deep-dive for selected bot.

  

  
# Other important panels to include



- Current Mode of the bot (MR or Trend)
- shorting activated status
- PnL of the bot (last 7, 30, total days)
- Gate status
- Riskstatus
- Phoenix protocol status
- Current leverage if futures
- Regime Analysis
- Exposure
- Dist to Action (bps) as current value meters and time graph
- weight tracking (Target vs Current)
- Actions (skips or trades) by reason
- Latency
