#!/bin/bash
# Meta Strategy Complete Optimization Workflow
# This script runs the full optimization pipeline: MR → Trend → Meta

set -e  # Exit on error

echo "========================================"
echo "Meta Strategy Optimization Workflow"
echo "========================================"

# Configuration
DATA_15M="data/raw/ETHBTC_15m_2021-2025_vision.csv"
FUNDING_DATA="data/raw/ETHUSDT_funding_2021-2025.csv"
TRAIN_START="2021-01-01"
TRAIN_END="2024-06-30"
TEST_START="2024-07-01"
TEST_END="2025-06-01"

# Number of trials (adjust for speed vs accuracy)
MR_TRIALS=50   # Mean Reversion
TR_TRIALS=30   # Trend
META_TRIALS=8  # Meta (grid search over ADX thresholds)

# Output directories
mkdir -p results
mkdir -p configs

# =============================================
# Step 1: Optimize Mean Reversion
# =============================================
echo ""
echo "[1/5] Optimizing Mean Reversion Strategy..."
python3 tools/optimizer_cli.py \
  --data "$DATA_15M" \
  --funding-data "$FUNDING_DATA" \
  --config configs/debug_test.json \
  --train-start "$TRAIN_START" \
  --train-end "$TRAIN_END" \
  --test-start "$TEST_START" \
  --test-end "$TEST_END" \
  --n-trials $MR_TRIALS \
  --study-name ethbtc_mr_meta_workflow \
  --lambda-turns 0.5 \
  --lambda-fees 0.5 \
  --lambda-turnover 0.25 \
  --gap-penalty 0.1 \
  --turns-scale 1000.0 \
  --out results/opt_mr_results.csv

echo "✓ Mean Reversion optimization complete"

# =============================================
# Step 2: Extract Best MR Parameters (Using wf_pick.py for robustness)
# =============================================
echo ""
echo "[2/5] Selecting robust Mean Reversion config..."
python3 tools/wf_pick.py \
  --runs results/opt_mr_results.csv \
  --out-csv results/wf_mr_families.csv \
  --emit-config configs/best_mr_params_only.json \
  --family-index 0 \
  --penalty-fees 1.0 \
  --penalty-turnover 0.5 \
  --disp-weight 0.25

# Wrap in full config structure
python3 - <<'PYTHON'
import json

with open("configs/best_mr_params_only.json") as f:
    params = json.load(f)

# Remove metadata
for k in ["_generated_by", "_generated_at", "_family"]:
    params.pop(k, None)

config = {
    "fees": {
        "maker_fee": 0.0002,
        "taker_fee": 0.0004,
        "slippage_bps": 1.0,
        "bnb_discount": 0.25,
        "pay_fees_in_bnb": True
    },
    "strategy": params,
    "execution": {
        "interval": "15m",
        "poll_sec": 5
    },
    "risk": {
        "basis_btc": 1.0,
        "max_daily_loss_btc": 0.0,
        "max_dd_btc": 0.0,
        "risk_mode": "fixed_basis"
    }
}

# Ensure required fields
if "strategy_type" not in config["strategy"]:
    config["strategy"]["strategy_type"] = "mean_reversion"
if "bar_interval_minutes" not in config["strategy"]:
    config["strategy"]["bar_interval_minutes"] = 15

with open("configs/best_mr.json", "w") as f:
    json.dump(config, f, indent=2)

print("✓ Robust MR config saved to configs/best_mr.json")
PYTHON

# =============================================
# Step 3: Optimize Trend
# =============================================
echo ""
echo "[3/5] Optimizing Trend Following Strategy..."
python3 tools/optimize_trend.py \
  --data "$DATA_15M" \
  --funding-data "$FUNDING_DATA" \
  --train-start "$TRAIN_START" \
  --train-end "$TRAIN_END" \
  --test-start "$TEST_START" \
  --test-end "$TEST_END" \
  --n-trials $TR_TRIALS \
  --out results/opt_trend_results.csv

echo "✓ Trend optimization complete"

# =============================================
# Step 4: Extract Best Trend Parameters (Using wf_pick.py)
# =============================================
echo ""
echo "[4/5] Selecting robust Trend config..."
python3 tools/wf_pick.py \
  --runs results/opt_trend_results.csv \
  --out-csv results/wf_trend_families.csv \
  --emit-config configs/best_trend_params_only.json \
  --family-index 0 \
  --penalty-fees 0.5 \
  --disp-weight 0.3

# Wrap in full config structure
python3 - <<'PYTHON'
import json

with open("configs/best_trend_params_only.json") as f:
    params = json.load(f)

# Remove metadata
for k in ["_generated_by", "_generated_at", "_family"]:
    params.pop(k, None)

config = {
    "fees": {
        "maker_fee": 0.0002,
        "taker_fee": 0.0004,
        "slippage_bps": 1.0,
        "bnb_discount": 0.25,
        "pay_fees_in_bnb": True
    },
    "strategy": params,
    "execution": {
        "interval": "15m",
        "poll_sec": 5
    },
    "risk": {
        "basis_btc": 1.0,
        "max_daily_loss_btc": 0.0,
        "max_dd_btc": 0.0,
        "risk_mode": "fixed_basis"
    }
}

# Ensure required fields
if "strategy_type" not in config["strategy"]:
    config["strategy"]["strategy_type"] = "trend"
if "step_allocation" not in config["strategy"]:
    config["strategy"]["step_allocation"] = 1.0
if "max_position" not in config["strategy"]:
    config["strategy"]["max_position"] = 1.0
if "long_only" not in config["strategy"]:
    config["strategy"]["long_only"] = True

with open("configs/best_trend.json", "w") as f:
    json.dump(config, f, indent=2)

print("✓ Robust Trend config saved to configs/best_trend.json")
PYTHON

# =============================================
# Step 5: Optimize Meta (ADX Threshold)
# =============================================
echo ""
echo "[5/5] Optimizing Meta Strategy (ADX threshold)..."
python3 tools/optimize_meta.py \
  --data "$DATA_15M" \
  --funding-data "$FUNDING_DATA" \
  --mr-config configs/best_mr.json \
  --trend-config configs/best_trend.json \
  --out results/opt_meta_results.csv

echo "✓ Meta optimization complete"

# =============================================
# Step 6: Generate Final Unified Config
# =============================================
echo ""
echo "Generating final unified Meta config..."
python3 - <<'PYTHON'
import pandas as pd
import json

# Load best meta result
df = pd.read_csv("results/opt_meta_results.csv")
best_meta = df.sort_values("final_btc", ascending=False).iloc[0]
best_adx = int(best_meta["adx_threshold"])

# Load MR and Trend configs
with open("configs/best_mr.json") as f:
    mr_cfg = json.load(f)
with open("configs/best_trend.json") as f:
    tr_cfg = json.load(f)

# Create unified meta config
config = {
    "fees": mr_cfg["fees"],
    "strategy": {
        "strategy_type": "meta",
        "adx_threshold": float(best_adx),
        
        # Mean Reversion params
        "trend_kind": mr_cfg["strategy"]["trend_kind"],
        "trend_lookback": mr_cfg["strategy"]["trend_lookback"],
        "flip_band_entry": mr_cfg["strategy"]["flip_band_entry"],
        "flip_band_exit": mr_cfg["strategy"]["flip_band_exit"],
        "vol_window": mr_cfg["strategy"]["vol_window"],
        "vol_adapt_k": mr_cfg["strategy"]["vol_adapt_k"],
        "target_vol": mr_cfg["strategy"].get("target_vol", 0.5),
        "min_mult": 0.5,
        "max_mult": 1.5,
        "gate_window_days": mr_cfg["strategy"]["gate_window_days"],
        "gate_roc_threshold": mr_cfg["strategy"]["gate_roc_threshold"],
        
        # Trend params
        "fast_period": tr_cfg["strategy"]["fast_period"],
        "slow_period": tr_cfg["strategy"]["slow_period"],
        "ma_type": tr_cfg["strategy"]["ma_type"],
        
        # Shared params (use MR as baseline)
        "cooldown_minutes": mr_cfg["strategy"]["cooldown_minutes"],
        "step_allocation": mr_cfg["strategy"]["step_allocation"],
        "max_position": mr_cfg["strategy"]["max_position"],
        "long_only": mr_cfg["strategy"]["long_only"],
        "rebalance_threshold_w": mr_cfg["strategy"]["rebalance_threshold_w"],
        "funding_limit_long": mr_cfg["strategy"]["funding_limit_long"],
        "funding_limit_short": mr_cfg["strategy"]["funding_limit_short"],
        "bar_interval_minutes": 15
    },
    "execution": {
        "interval": "15m",
        "poll_sec": 5,
        "ttl_sec": 30,
        "min_trade_floor_btc": 0.0001,
        "min_trade_btc": 0.0001,
        "min_trade_frac": 0.0,
        "min_trade_cap_btc": 1.0,
        "taker_fallback": True
    },
    "risk": {
        "basis_btc": 1.0,
        "max_daily_loss_btc": 0.0,
        "max_dd_btc": 0.0,
        "risk_mode": "fixed_basis"
    }
}

with open("configs/meta_optimized.json", "w") as f:
    json.dump(config, f, indent=2)

print(f"✓ Final Meta config saved to configs/meta_optimized.json")
print(f"  ADX Threshold: {best_adx}")
print(f"  Final BTC: {best_meta['final_btc']:.4f}")
PYTHON

# =============================================
# Summary
# =============================================
echo ""
echo "========================================"
echo "✓ Optimization Workflow Complete!"
echo "========================================"
echo ""
echo "Generated files:"
echo "  - configs/best_mr.json         (Mean Reversion)"
echo "  - configs/best_trend.json      (Trend Following)"
echo "  - configs/meta_optimized.json  (Unified Meta)"
echo ""
echo "Results:"
echo "  - results/opt_mr_results.csv"
echo "  - results/opt_trend_results.csv"
echo "  - results/opt_meta_results.csv"
echo ""
echo "Next steps:"
echo "  1. Review configs/meta_optimized.json"
echo "  2. Backtest with: ./run_meta_test.sh (update config path)"
echo "  3. Deploy to live/testnet if satisfied"
echo ""
