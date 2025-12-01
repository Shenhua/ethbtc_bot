#!/bin/bash
# Meta Strategy Complete Optimization Workflow (Generic)
# Usage: ./run_complete_optimization.sh <PRICE_CSV> <FUNDING_CSV> <SYMBOL_TAG>

set -e  # Exit on error

# --- 1. CONFIGURATION & ARGUMENT PARSING ---

# Defaults (ETHBTC)
DEFAULT_PRICE="data/raw/ETHBTC_15m_2021-2025_vision.csv"
DEFAULT_FUND="data/raw/ETHUSDT_funding_2021-2025.csv"
DEFAULT_TAG="ETH"

# Read Args (or use defaults)
PRICE_DATA="${1:-$DEFAULT_PRICE}"
FUNDING_DATA="${2:-$DEFAULT_FUND}"
TAG="${3:-$DEFAULT_TAG}"

# Date Ranges (Edit these if you want different periods for different pairs)
TRAIN_START="2021-01-01"
TRAIN_END="2024-06-30"
TEST_START="2024-07-01"
TEST_END="2025-06-01"

# Complexity Settings
MR_TRIALS=50    # Mean Reversion Trials
TR_TRIALS=50    # Trend Trials
META_TRIALS=8   # ADX Threshold Steps

# Dynamic Output Filenames (Namespaced by TAG)
OUT_MR_CSV="results/opt_mr_${TAG}.csv"
OUT_MR_FAMILIES="results/wf_mr_families_${TAG}.csv"
OUT_MR_CONF="configs/best_mr_${TAG}.json"
OUT_MR_PARAMS="configs/best_mr_params_${TAG}.json"

OUT_TR_CSV="results/opt_trend_${TAG}.csv"
OUT_TR_FAMILIES="results/wf_trend_families_${TAG}.csv"
OUT_TR_CONF="configs/best_trend_${TAG}.json"
OUT_TR_PARAMS="configs/best_trend_params_${TAG}.json"

OUT_META_CSV="results/opt_meta_${TAG}.csv"
FINAL_CONFIG="configs/meta_optimized_v2_${TAG}.json"

# --- 2. PRE-FLIGHT CHECKS ---

echo "========================================"
echo "Optimization Workflow: ${TAG}"
echo "========================================"
echo "Price Data:   ${PRICE_DATA}"
echo "Funding Data: ${FUNDING_DATA}"
echo "Output Config: ${FINAL_CONFIG}"
echo ""

mkdir -p results
mkdir -p configs

# =============================================
# Step 1: Optimize Mean Reversion
# =============================================
echo ""
echo "[1/6] Optimizing Mean Reversion (${TAG})..."
python3 tools/optimizer_cli.py \
  --data "$PRICE_DATA" \
  --funding-data "$FUNDING_DATA" \
  --train-start "$TRAIN_START" \
  --train-end "$TRAIN_END" \
  --test-start "$TEST_START" \
  --test-end "$TEST_END" \
  --n-trials $MR_TRIALS \
  --storage "sqlite:///data/db/optuna.db" \
  --study-name "mr_${TAG}_study" \
  --lambda-turns 0.5 --lambda-fees 0.5 --lambda-turnover 0.25 --gap-penalty 0.1 \
  --out "$OUT_MR_CSV"

# =============================================
# Step 2: Pick Best MR
# =============================================
echo ""
echo "[2/6] Selecting best MR config..."
python3 tools/wf_pick.py \
  --runs "$OUT_MR_CSV" \
  --out-csv "$OUT_MR_FAMILIES" \
  --emit-config "$OUT_MR_PARAMS" \
  --family-index 0 \
  --penalty-fees 1.0 --penalty-turnover 0.5 \
  --min-occurs 1

# Wrap params into full config structure
python3 - <<PYTHON
import json
with open("$OUT_MR_PARAMS") as f: params = json.load(f)
# Clean metadata
for k in ["_generated_by", "_generated_at", "_family"]: params.pop(k, None)

config = {
    "fees": { "maker_fee": 0.0002, "taker_fee": 0.0004, "slippage_bps": 1.0, "bnb_discount": 0.25, "pay_fees_in_bnb": True },
    "strategy": params,
    "execution": { "interval": "15m", "poll_sec": 5 },
    "risk": { "basis_btc": 1.0, "risk_mode": "fixed_basis" }
}
# Defaults
config["strategy"]["strategy_type"] = "mean_reversion"
if "bar_interval_minutes" not in config["strategy"]: config["strategy"]["bar_interval_minutes"] = 15

with open("$OUT_MR_CONF", "w") as f: json.dump(config, f, indent=2)
PYTHON

# =============================================
# Step 3: Optimize Trend
# =============================================
echo ""
echo "[3/6] Optimizing Trend (${TAG})..."
python3 tools/optimize_trend.py \
  --data "$PRICE_DATA" \
  --funding-data "$FUNDING_DATA" \
  --train-start "$TRAIN_START" \
  --train-end "$TRAIN_END" \
  --test-start "$TEST_START" \
  --test-end "$TEST_END" \
  --n-trials $TR_TRIALS \
  --storage "sqlite:///data/db/optuna.db" \
  --study-name "trend_${TAG}_study" \
  --out "$OUT_TR_CSV"

# =============================================
# Step 4: Pick Best Trend
# =============================================
echo ""
echo "[4/6] Selecting best Trend config..."
python3 tools/wf_pick.py \
  --runs "$OUT_TR_CSV" \
  --out-csv "$OUT_TR_FAMILIES" \
  --emit-config "$OUT_TR_PARAMS" \
  --family-index 0 \
  --penalty-fees 0.5 \
  --min-occurs 1

# Wrap params
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
if "step_allocation" not in config["strategy"]: config["strategy"]["step_allocation"] = 1.0
if "max_position" not in config["strategy"]: config["strategy"]["max_position"] = 1.0
if "long_only" not in config["strategy"]: config["strategy"]["long_only"] = True

with open("$OUT_TR_CONF", "w") as f: json.dump(config, f, indent=2)
PYTHON

# =============================================
# Step 5: Optimize Meta Threshold
# =============================================
echo ""
echo "[5/6] Optimizing Meta Threshold (${TAG})..."
python3 tools/optimize_meta.py \
  --data "$PRICE_DATA" \
  --funding-data "$FUNDING_DATA" \
  --mr-config "$OUT_MR_CONF" \
  --trend-config "$OUT_TR_CONF" \
  --out "$OUT_META_CSV"

# =============================================
# Step 6: Assemble Final V2 Config
# =============================================
echo ""
echo "[6/6] Generating Final V2 Config..."

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