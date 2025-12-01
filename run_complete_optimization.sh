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
# Step 6: Generate Final Unified Config (V2 Format)
# =============================================
echo ""
echo "[6/6] Generating Final V2 Config..."

python3 tools/assemble_v2_config.py \
  --mr configs/best_mr.json \
  --trend configs/best_trend.json \
  --meta-results results/opt_meta_results.csv \
  --out configs/meta_optimized_v2.json

echo ""
echo "========================================"
echo "✓ Optimization Workflow Complete!"
echo "========================================"
echo ""
echo "New Config: configs/meta_optimized_v2.json"
echo "Usage: python live_executor.py --params configs/meta_optimized_v2.json ..."
echo ""