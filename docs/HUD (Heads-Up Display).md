This guide explains how to read every panel of the dashboard we just built, how to interpret the signals, and how to spot problems before they become losses.

------



# 🟢 Row 1: The HUD (Heads-Up Display)



*Goal: Instant health check. "Is the house on fire?"*



### 1. Total Wealth (BTC)



- **What it shows:** Your total portfolio value converted to BTC (BTC Balance + ETH Balance * Price).
- **How to read it:**
  - **Sparkline (Background):** Shows the trend over the last 24h. You want to see this line go up, even if the market is flat.
  - **Big Number:** Your "Score."



### 2. Session PnL



- **What it shows:** Profit or Loss since the bot last started (or since you reset the `state.json`).
- **Color Code:**
  - 🟢 **Green:** You are making BTC.
  - 🔴 **Red:** You are currently in drawdown relative to the session start.



### 3. Risk Status (The Check Engine Light)



- **What it shows:** Whether a safety breaker has been tripped.
- **States:**
  - 🟩 **NORMAL:** The bot is operating freely.
  - 🟥 **MAX DD HIT:** Critical. The bot hit your Max Drawdown limit (e.g., 15% loss). It has likely sold everything into BTC and stopped trading to protect capital.
  - 🟧 **DAILY LIMIT:** The bot hit the daily loss limit and is paused until tomorrow (UTC midnight).



### 4. Regime Gate (The Traffic Light)



- **What it shows:** The Long-Term Trend Filter (60-day ROC).
- **States:**
  - 🟩 **OPEN:** The market trend is healthy (or safe). Trading is allowed.
  - 🟥 **CLOSED:** The market is in a dangerous trend (e.g., violent bear market). The bot will **refuse** to open new positions, even if the short-term signal says "Buy."



### 5. Current Exposure (Gauge)



- **What it shows:** Percentage of your portfolio held in ETH.
- **Reading:**
  - **0%:** You are fully in BTC (Safety / Waiting).
  - **100%:** You are fully in ETH (Betting on ETH reversal).

------



# 🔵 Row 2: Decision History (The "Why?" Engine)



*Goal: Understand why the bot is waiting or skipping trades.*



### 6. Blocker Timeline



- **What it shows:** A color-coded history of the bot's internal decisions for every 15-minute bar.
- **Common States:**
  - `skip_threshold` (Most Common): The signal changed slightly, but not enough to justify paying fees. The bot is "filtering noise."
  - `skip_delta_zero`: Nothing happened. Market is dead flat.
  - `skip_min_notional`: The calculated trade size was too small (e.g., < 0.0001 BTC).
  - `gate_closed`: The strategy wants to trade, but the "Regime Gate" (Row 1) said No.
  - `exec_buy` / `exec_sell`: **Action!** A trade occurred.

------



# 🟡 Row 3: Strategy X-Ray



*Goal: Visualizing the "Invisible Bands."*



### 7. Signal vs Static Bands (The Main Chart)



- **Yellow Line (Signal Ratio):** This is the deviation of Price from the Trend.
  - If it goes **UP**: ETH is becoming expensive relative to trend.
  - If it goes **DOWN**: ETH is becoming cheap (oversold).
- **Red Zone (+Entry):** When the Yellow line enters the Red Zone, the bot **SELLS ETH**.
- **Green Zone (-Entry):** When the Yellow line enters the Green Zone, the bot **BUYS ETH**.
- **The "Void":** The empty space in the middle is the "Neutral Zone." We hold cash here and wait.



### 8. Proximity to Trade (The Radar)



- **Unit:** **bps** (Basis Points). 100 bps = 1%.
- **What it shows:** The distance to the *nearest* trigger.
- **How to read it:**
  - **> 100 bps:** Relax. No trade is imminent.
  - **< 20 bps:** Get ready. A trade is very likely in the next few bars.
  - **0 bps:** You are inside the band. The bot is actively trying to trade.

------



# 🟣 Row 4: Execution & Diagnostics



*Goal: Technical debugging. "Is the bot broken?"*



### 9. Weight Tracking (Stuck Order Detector)



- **Blue Dashed Line:** The **Target** weight (What the Strategy wants).
- **Green Solid Line:** The **Current** weight (What the Wallet actually has).
- **The Golden Rule:** **These lines should overlap.**
  - If they diverge (Blue goes up, Green stays flat), it means the bot **failed to execute**. Check "Why Skip?" panel immediately.



### 10. Why Skip? (Histogram)



- **What it shows:** A count of rejected/skipped trades per hour.
- **Red Flags:**
  - `skip_balance`: **CRITICAL.** Means "Insufficient Funds." You need to apply the "0.999 fee buffer" fix.
  - `skip_order_error`: API disconnected or Keys are invalid.



### 11. Bar Latency



- **What it shows:** How long it takes (in seconds) to calculate logic and talk to Binance.
- **Benchmarks:**
  - **< 1s:** Excellent.
  - **1s - 3s:** Normal.
  - **> 5s:** Sluggish. Check your server's internet or CPU.

------



# Scenarios: "What does it mean when..."



**Scenario A: The bot isn't trading.**

- Check **Row 3 (Proximity)**. Is the distance > 0 bps? Then it's simply waiting for the price to hit the band. This is normal.
- Check **Row 1 (Gate)**. Is it CLOSED? Then it's protecting you from a bad trend.

**Scenario B: The "Session PnL" is huge and incorrect.**

- You likely restarted the container without deleting `run_state/state.json`, so it thinks you started with 0.16 BTC but now have 1.0 BTC.
- **Fix:** Run `rm run_state/state.json` and restart.

**Scenario C: Weight Tracking lines are splitting apart.**

- The Strategy (Blue) wants 50% ETH, but Wallet (Green) stays at 0%.
- Look at **Row 4 (Why Skip?)**. If you see `skip_min_notional`, your capital is too small for the calculated trade size. If you see `skip_balance`, your fee calculation is wrong.