#!/bin/bash
# Meta Strategy Complete Optimization Workflow (Generic + WFO Support)
# Usage: ./run_complete_optimization.sh [PRICE_CSV] [FUNDING_CSV] [TAG] [--wfo]

set -e  # Exit on error

# --- 1. CONFIGURATION & ARGUMENT PARSING ---

# Defaults
DEFAULT_PRICE="data/raw/ETHBTC_15m_2021-2025_vision.csv"
DEFAULT_FUND="data/raw/ETHUSDT_funding_2021-2025.csv"
DEFAULT_TAG="ETH"

# Variables
PRICE_DATA=$DEFAULT_PRICE
FUNDING_DATA=$DEFAULT_FUND
TAG=$DEFAULT_TAG
USE_WFO=false

# Parse Arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --wfo)
      USE_WFO=true
      shift # past argument
      ;;
    *.csv)
      if [[ -z "$FOUND_PRICE" ]]; then
        PRICE_DATA="$1"
        FOUND_PRICE=true
      else
        FUNDING_DATA="$1"
      fi
      shift
      ;;
    *)
      if [[ "$1" != -* ]]; then
        TAG="$1"
      fi
      shift
      ;;
  esac
done

# Dates (Static Mode)
TRAIN_START="2021-01-01"
TRAIN_END="2024-06-30"
TEST_START="2024-07-01"
TEST_END="2025-06-01"

# WFO Settings (Rolling Mode)
WINDOW_DAYS=180
STEP_DAYS=30

# Complexity
MR_TRIALS=50
TR_TRIALS=30
META_TRIALS=8

# Filenames
OUT_MR_CSV="results/opt_mr_${TAG}.csv"
OUT_MR_PARAMS="configs/best_mr_params_${TAG}.json"
OUT_MR_CONF="configs/best_mr_${TAG}.json"

OUT_TR_CSV="results/opt_trend_${TAG}.csv"
OUT_TR_WFO_CSV="results/wfo_trend_${TAG}.csv"
OUT_TR_FAMILIES="results/wf_trend_families_${TAG}.csv"
OUT_TR_PARAMS="configs/best_trend_params_${TAG}.json"
OUT_TR_CONF="configs/best_trend_${TAG}.json"

OUT_META_CSV="results/opt_meta_${TAG}.csv"
FINAL_CONFIG="configs/meta_optimized_v2_${TAG}.json"

echo "========================================"
echo "Optimization Workflow: ${TAG}"
echo "Mode: $( [ "$USE_WFO" = true ] && echo "Walk-Forward (Rolling)" || echo "Static" )"
echo "========================================"
echo "Price Data:   ${PRICE_DATA}"
echo "Funding Data: ${FUNDING_DATA}"
echo "Output Config: ${FINAL_CONFIG}"
echo ""

mkdir -p results configs

# =============================================
# Step 1: Optimize Mean Reversion (Static)
# =============================================
# Note: MR is usually best optimized statically on long timeframes to find universal "chop" params.
echo ""
echo "[1/6] Optimizing Mean Reversion..."
python3 tools/optimizer_cli.py \
  --data "$PRICE_DATA" \
  --funding-data "$FUNDING_DATA" \
  --train-start "$TRAIN_START" --train-end "$TRAIN_END" \
  --test-start "$TEST_START" --test-end "$TEST_END" \
  --n-trials $MR_TRIALS \
  --study-name "mr_${TAG}_study" \
  --storage "sqlite:///data/db/optuna.db" \
  --out "$OUT_MR_CSV"

# Step 2: Pick Best MR
echo "[2/6] Selecting Best MR..."
python3 tools/wf_pick.py \
  --runs "$OUT_MR_CSV" \
  --emit-config "$OUT_MR_PARAMS" \
  --family-index 0 --min-occurs 1

# Wrap MR Config
python3 - <<PYTHON
import json
with open("$OUT_MR_PARAMS") as f: params = json.load(f)
for k in ["_generated_by", "_generated_at", "_family"]: params.pop(k, None)
config = {
    "fees": { "maker_fee": 0.0002, "taker_fee": 0.0004, "slippage_bps": 1.0, "bnb_discount": 0.25, "pay_fees_in_bnb": True },
    "strategy": params,
    "execution": { "interval": "15m", "poll_sec": 5 },
    "risk": { "basis_btc": 1.0, "risk_mode": "fixed_basis" }
}
config["strategy"]["strategy_type"] = "mean_reversion"
config["strategy"]["bar_interval_minutes"] = 15
with open("$OUT_MR_CONF", "w") as f: json.dump(config, f, indent=2)
PYTHON

# =============================================
# Step 3: Optimize Trend (Conditional)
# =============================================
echo ""
echo "[3/6] Optimizing Trend..."

if [ "$USE_WFO" = true ]; then
    # --- WFO PATH ---
    echo "Running Walk-Forward Optimization (Window=${WINDOW_DAYS}d, Step=${STEP_DAYS}d)..."
    python3 tools/optimize_trend.py \
      --data "$PRICE_DATA" \
      --funding-data "$FUNDING_DATA" \
      --wfo \
      --allow-shorts \
      --window-days $WINDOW_DAYS \
      --step-days $STEP_DAYS \
      --n-trials $TR_TRIALS \
      --storage "sqlite:///data/db/optuna.db" \
      --study-name "trend_${TAG}_wfo" \
      --out "$OUT_TR_WFO_CSV"

echo "[4/6] Extracting Latest WFO Params..."
    # Extract the params from the LAST window in the CSV (The most recent market regime)
    python3 - <<PYTHON
import pandas as pd
import json
import sys

try:
    df = pd.read_csv("$OUT_TR_WFO_CSV")
    latest = df.iloc[-1] # Get the last row
    print(f"Selected Window ending: {latest['window_end']}")
    print(f"OOS Profit: {latest['oos_profit']}")

    # FIX: Use json.loads because the generator uses json.dumps
    # (ast.literal_eval fails on 'true'/'false' from JSON)
    params_str = latest["best_params"]
    try:
        params = json.loads(params_str)
    except json.JSONDecodeError:
        # Fallback if somehow it wasn't valid JSON (e.g. single quotes)
        import ast
        params = ast.literal_eval(params_str)

    print(f"Latest Params (LongOnly={params.get('long_only')}): Saved.")

    with open("$OUT_TR_PARAMS", "w") as f:
        json.dump(params, f, indent=2)

except Exception as e:
    print(f"Error extracting params: {e}")
    sys.exit(1)
PYTHON

else
    # --- STATIC PATH ---
    echo "Running Static Optimization..."
    python3 tools/optimize_trend.py \
      --data "$PRICE_DATA" \
      --funding-data "$FUNDING_DATA" \
      --train-start "$TRAIN_START" --train-end "$TRAIN_END" \
      --test-start "$TEST_START" --test-end "$TEST_END" \
      --n-trials $TR_TRIALS \
      --storage "sqlite:///data/db/optuna.db" \
      --study-name "trend_${TAG}_static" \
      --out "$OUT_TR_CSV"

    echo "[4/6] Picking Robust Static Params..."
    python3 tools/wf_pick.py \
      --runs "$OUT_TR_CSV" \
      --emit-config "$OUT_TR_PARAMS" \
      --family-index 0 --min-occurs 1
fi

# Wrap Trend Config (Common)
python3 - <<PYTHON
import json
with open("$OUT_TR_PARAMS") as f: params = json.load(f)
for k in ["_generated_by", "_generated_at", "_family"]: params.pop(k, None)

config = {
    "fees": { "maker_fee": 0.0002, "taker_fee": 0.0004, "slippage_bps": 1.0, "bnb_discount": 0.25, "pay_fees_in_bnb": True },
    "strategy": params,
    "execution": { "interval": "15m", "poll_sec": 5 },
    "risk": { "basis_btc": 1.0, "risk_mode": "fixed_basis" }
}
config["strategy"]["strategy_type"] = "trend"

# IMPORTANT: Do NOT force long_only=True here. 
# Allow the optimizer's choice (which is inside 'params') to prevail.
# Only set defaults if missing.
if "long_only" not in config["strategy"]: config["strategy"]["long_only"] = True
if "step_allocation" not in config["strategy"]: config["strategy"]["step_allocation"] = 1.0
if "max_position" not in config["strategy"]: config["strategy"]["max_position"] = 1.0

with open("$OUT_TR_CONF", "w") as f: json.dump(config, f, indent=2)
PYTHON

# =============================================
# Step 5 & 6: Meta Optimization & Assembly
# =============================================
echo ""
echo "[5/6] Optimizing Meta Threshold..."
python3 tools/optimize_meta.py \
  --data "$PRICE_DATA" \
  --funding-data "$FUNDING_DATA" \
  --mr-config "$OUT_MR_CONF" \
  --trend-config "$OUT_TR_CONF" \
  --out "$OUT_META_CSV"

echo ""
echo "[6/6] Assembling Final V2 Config..."
python3 tools/assemble_v2_config.py \
  --mr "$OUT_MR_CONF" \
  --trend "$OUT_TR_CONF" \
  --meta-results "$OUT_META_CSV" \
  --out "$FINAL_CONFIG"

echo ""
echo "========================================"
echo "✓ DONE: ${TAG} Optimization Complete"
echo "========================================"
echo "File: ${FINAL_CONFIG}"
echo ""