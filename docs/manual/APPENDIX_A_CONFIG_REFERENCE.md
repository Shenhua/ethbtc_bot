# Appendix A: Configuration Reference

> **Purpose:** Complete reference of all configuration parameters with types, defaults, and valid ranges.

---

## A.1 Configuration Structure

```json
{
  "fees": { ... },
  "strategy": { ... },
  "execution": { ... },
  "risk": { ... }
}
```

---

## A.2 Fees Section

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `maker_fee` | float | `0.0001` | 0–0.01 | Maker fee rate (0.01%) |
| `taker_fee` | float | `0.0005` | 0–0.01 | Taker fee rate (0.05%) |
| `slippage_bps` | float | `1.0` | 0–50 | Expected slippage in basis points |
| `bnb_discount` | float | `0.25` | 0–1 | BNB fee discount (25% = 0.25) |
| `pay_fees_in_bnb` | bool | `true` | — | Pay fees using BNB balance |

**Example:**
```json
"fees": {
  "maker_fee": 0.0002,
  "taker_fee": 0.0004,
  "slippage_bps": 1.5,
  "bnb_discount": 0.25,
  "pay_fees_in_bnb": true
}
```

---

## A.3 Strategy Section

### Core Strategy Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `strategy_type` | str | `"mean_reversion"` | `mean_reversion`, `trend`, `meta` | Strategy type |
| `long_only` | bool | `true` | — | Long-only mode (no shorting) |

### Mean Reversion Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `trend_kind` | str | `"sma"` | `sma`, `roc` | Trend calculation method |
| `trend_lookback` | int | `200` | 50–500 | Trend MA lookback (bars) |
| `flip_band_entry` | float | `0.03` | 0.01–0.10 | Entry threshold (3%) |
| `flip_band_exit` | float | `0.015` | 0.005–0.05 | Exit threshold (1.5%) |
| `vol_window` | int | `60` | 20–120 | Volatility window (bars) |
| `vol_adapt_k` | float | `0.005` | 0–0.02 | Volatility adaptation factor |
| `target_vol` | float | `0.4` | 0.1–1.0 | Target annualized volatility |
| `min_mult` | float | `0.5` | 0.1–1.0 | Minimum volatility multiplier |
| `max_mult` | float | `1.5` | 1.0–3.0 | Maximum volatility multiplier |
| `cooldown_minutes` | int | `120` | 15–480 | Cooldown between trades |
| `step_allocation` | float | `0.5` | 0.1–1.0 | Position step size |
| `max_position` | float | `1.0` | 0.1–1.0 | Maximum position (100%) |

### Trend Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `fast_period` | int | `50` | 10–200 | Fast MA period |
| `slow_period` | int | `200` | 50–500 | Slow MA period |
| `ma_type` | str | `"ema"` | `ema`, `sma` | Moving average type |

### Meta Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `adx_threshold` | float | `25.0` | 10–50 | ADX regime threshold |
| `mean_reversion_override` | dict | `{}` | — | MR-specific overrides |
| `trend_override` | dict | `{}` | — | Trend-specific overrides |

### Gate Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `gate_window_days` | int | `60` | 30–180 | Gate lookback (days) |
| `gate_roc_threshold` | float | `0.01` | 0–0.05 | Minimum ROC to open gate |

### Enhanced Filters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rsi_filter` | bool | `false` | Enable RSI divergence filter |
| `rsi_period` | int | `14` | RSI calculation period |
| `rsi_oversold` | float | `30.0` | RSI oversold threshold |
| `rsi_overbought` | float | `70.0` | RSI overbought threshold |
| `bollinger_squeeze` | bool | `false` | Enable Bollinger squeeze filter |
| `bollinger_period` | int | `20` | Bollinger band period |
| `bollinger_squeeze_pct` | float | `0.5` | Squeeze compression threshold |
| `volume_confirm` | bool | `false` | Enable volume confirmation |
| `volume_mult` | float | `1.5` | Volume multiple required |
| `funding_counter` | bool | `false` | Enable funding counter-trend |
| `funding_counter_threshold` | float | `0.05` | Funding threshold |
| `htf_filter` | bool | `false` | Enable higher timeframe filter |
| `htf_period` | int | `4` | HTF multiplier (4x = 1h from 15m) |

### Position Sizing

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `position_sizing_mode` | str | `"static"` | `static`, `volatility`, `kelly` | Sizing mode |
| `position_sizing_target_vol` | float | `0.5` | 0.1–1.0 | Target vol for sizing |
| `position_sizing_min_step` | float | `0.1` | 0.05–0.5 | Minimum step size |
| `position_sizing_max_step` | float | `1.0` | 0.5–1.0 | Maximum step size |
| `kelly_win_rate` | float | `0.55` | 0.4–0.7 | Kelly win rate |
| `kelly_avg_win` | float | `0.02` | 0.01–0.10 | Kelly average win |
| `kelly_avg_loss` | float | `0.015` | 0.01–0.10 | Kelly average loss |
| `kelly_fraction` | float | `0.5` | 0.1–1.0 | Half-Kelly fraction |

### Funding Filters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `funding_limit_long` | float | `0.05` | 0.01–0.20 | Max funding for longs (%) |
| `funding_limit_short` | float | `-0.05` | -0.20–-0.01 | Max funding for shorts (%) |

---

## A.4 Execution Section

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `exchange_type` | str | `"spot"` | `spot`, `futures` | Exchange market type |
| `interval` | str | `"15m"` | `1m`–`1d` | Bar interval |
| `poll_sec` | int | `5` | 1–30 | Order poll interval (seconds) |
| `ttl_sec` | int | `60` | 30–300 | Order TTL before fallback |
| `taker_fallback` | bool | `true` | — | Enable taker fallback |
| `max_taker_btc` | float | `0.1` | 0–1.0 | Max taker order size |
| `max_spread_bps_for_taker` | float | `10.0` | 0–50 | Max spread for taker |
| `min_trade_btc` | float | `0.0001` | 0–0.01 | Minimum trade size |
| `min_trade_frac` | float | `0.01` | 0–0.1 | Min trade as fraction |
| `min_trade_floor_btc` | float | `0.0001` | 0–0.01 | Absolute minimum |
| `min_trade_cap_btc` | float | `0.01` | 0–0.1 | Maximum of min_trade |
| `leverage` | int | `1` | 1–10 | Futures leverage |

**Example:**
```json
"execution": {
  "exchange_type": "futures",
  "interval": "15m",
  "poll_sec": 5,
  "ttl_sec": 60,
  "taker_fallback": true,
  "max_taker_btc": 0.05,
  "leverage": 3
}
```

---

## A.5 Risk Section

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `risk_mode` | str | `"fixed_basis"` | `fixed_basis`, `dynamic` | Risk calculation mode |
| `basis_btc` | float | `1.0` | 0.01–100 | Initial capital (BTC) |
| `max_daily_loss_btc` | float | `0.0` | 0–10 | Max daily loss (absolute) |
| `max_daily_loss_frac` | float | `0.03` | 0–0.2 | Max daily loss (3%) |
| `max_dd_btc` | float | `0.0` | 0–10 | Max drawdown (absolute) |
| `max_dd_frac` | float | `0.15` | 0–0.5 | Max drawdown (15%) |
| `drawdown_reset_days` | float | `0.0` | 0–30 | Phoenix cooldown (days) |
| `drawdown_reset_score` | float | `30.0` | 0–100 | Phoenix score threshold |

**Example:**
```json
"risk": {
  "risk_mode": "dynamic",
  "basis_btc": 1.0,
  "max_daily_loss_frac": 0.03,
  "max_dd_frac": 0.15,
  "drawdown_reset_days": 3.0,
  "drawdown_reset_score": 25.0
}
```

---

## A.6 Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `MODE` | Operating mode | `dry`, `testnet`, `live` |
| `BINANCE_KEY` | API key (Spot) | `abc123...` |
| `BINANCE_SECRET` | API secret (Spot) | `xyz789...` |
| `BINANCE_FUTURES_KEY` | API key (Futures) | `abc123...` |
| `BINANCE_FUTURES_SECRET` | API secret (Futures) | `xyz789...` |
| `BINANCE_BASE_URL` | API base URL | `https://api.binance.com` |
| `STATE_FILE` | State file path | `/data/state.json` |
| `TRADE_LOG_FILE` | Trade log path | `/data/trades.jsonl` |
| `LOGLEVEL` | Logging level | `DEBUG`, `INFO`, `WARNING` |
| `LOG_JSON` | JSON log output | `true`, `false` |
| `METRICS_PORT` | Prometheus port | `9109` |
| `STATUS_PORT` | Status HTTP port | `9110` |
| `DISCORD_WEBHOOK_URL` | Discord alerts | `https://discord.com/...` |
| `TELEGRAM_TOKEN` | Telegram bot token | `123456:ABC...` |
| `TELEGRAM_CHAT_ID` | Telegram chat ID | `-1001234567890` |
| `SWEEPER_IGNORE` | Dust sweeper ignore | `SHIB,DOGE` |

---

## A.7 Complete Production Config Example

```json
{
  "fees": {
    "maker_fee": 0.0002,
    "taker_fee": 0.0004,
    "slippage_bps": 1.5,
    "bnb_discount": 0.25,
    "pay_fees_in_bnb": true
  },
  "strategy": {
    "strategy_type": "meta",
    "long_only": true,
    "trend_kind": "sma",
    "trend_lookback": 160,
    "flip_band_entry": 0.042,
    "flip_band_exit": 0.022,
    "vol_window": 60,
    "vol_adapt_k": 0.005,
    "target_vol": 0.5,
    "cooldown_minutes": 120,
    "step_allocation": 0.5,
    "max_position": 1.0,
    "fast_period": 50,
    "slow_period": 200,
    "ma_type": "ema",
    "adx_threshold": 25.0,
    "gate_window_days": 60,
    "gate_roc_threshold": 0.01,
    "position_sizing_mode": "volatility",
    "position_sizing_target_vol": 0.5,
    "position_sizing_min_step": 0.15,
    "position_sizing_max_step": 1.0,
    "funding_limit_long": 0.05,
    "funding_limit_short": -0.05,
    "rsi_filter": false,
    "bollinger_squeeze": false,
    "volume_confirm": false
  },
  "execution": {
    "exchange_type": "futures",
    "interval": "15m",
    "poll_sec": 5,
    "ttl_sec": 60,
    "taker_fallback": true,
    "max_taker_btc": 0.05,
    "max_spread_bps_for_taker": 10.0,
    "min_trade_btc": 0.0005,
    "leverage": 3
  },
  "risk": {
    "risk_mode": "dynamic",
    "basis_btc": 1.0,
    "max_daily_loss_frac": 0.03,
    "max_dd_frac": 0.15,
    "drawdown_reset_days": 3.0,
    "drawdown_reset_score": 25.0
  }
}
```

---

*Return to: [Table of Contents](./MASTER_TABLE_OF_CONTENTS.md)*
