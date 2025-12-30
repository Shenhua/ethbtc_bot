#!/bin/bash
# Meta Strategy Complete Optimization Workflow (Generic + WFO + Dates)
# Usage: ./run_complete_optimization.sh [PRICE_CSV] [FUNDING_CSV] [TAG] [--wfo] [--train-start YYYY-MM-DD] ...

set -e  # Exit on error

# --- 1. CONFIGURATION & ARGUMENT PARSING ---

# Defaults
DEFAULT_PRICE="data/raw/BTCUSDT_15m_2021-2025_vision.csv"
DEFAULT_FUND="data/raw/BTCUSDT_funding_2021-2025.csv"
DEFAULT_TAG="BTC"

# Default Dates (Full History)
TRAIN_START="2021-01-01"
TRAIN_END="2024-06-30"
TEST_START="2024-07-01"
TEST_END="2025-06-01"

# Variables
PRICE_DATA=$DEFAULT_PRICE
FUNDING_DATA=$DEFAULT_FUND
TAG=$DEFAULT_TAG
WFO_MODE=false
EXHAUSTIVE_MODE=false
LONG_ONLY_FUTURES_MODE=false
FUTURES_ONLY_MODE=false

# Parse Arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --wfo)
      WFO_MODE=true
      shift
      ;;
    --exhaustive)
      EXHAUSTIVE_MODE=true
      shift
      ;;
    --long-only-futures)
      LONG_ONLY_FUTURES_MODE=true
      EXHAUSTIVE_MODE=true
      shift
      ;;
    --futures-only)
      FUTURES_ONLY_MODE=true
      EXHAUSTIVE_MODE=true
      shift
      ;;
    --train-start)
      TRAIN_START="$2"
      shift 2
      ;;
    --train-end)
      TRAIN_END="$2"
      shift 2
      ;;
    --test-start)
      TEST_START="$2"
      shift 2
      ;;
    --test-end)
      TEST_END="$2"
      shift 2
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

# WFO Settings (Rolling Mode)
WINDOW_DAYS=180
STEP_DAYS=30

# Optimization Complexity
MR_TRIALS=50
TR_TRIALS=30
META_TRIALS=8

# === PARALLELIZATION SETTINGS ===
# Auto-detect CPU cores (use 75% for parallel work)
AVAIL_CORES=$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)
MAX_PARALLEL_COMBOS=$((AVAIL_CORES > 8 ? 8 : AVAIL_CORES))  # Max 8 (one per combo)
JOBS_PER_COMBO=$((AVAIL_CORES / MAX_PARALLEL_COMBOS))       # Distribute remaining

# Ensure at least 1 job per combo
[[ $JOBS_PER_COMBO -lt 1 ]] && JOBS_PER_COMBO=1

# === SIGNAL TRAP (Ctrl+C Support) ===
cleanup() {
    echo ""
    echo "🛑 Interrupt received! Killing background Optuna jobs..."
    
    # Kill all background jobs listed by the shell
    # 'jobs -p' gets the PIDs of background processes
    pids=$(jobs -p)
    if [ -n "$pids" ]; then
        kill $pids 2>/dev/null
    fi
    
    echo "💀 Cleanup complete. Exiting."
    exit 1
}

# Register the trap for SIGINT (Ctrl+C) and SIGTERM
trap cleanup SIGINT SIGTERM

# =============================================
# LOGGING & PROGRESS TRACKING
# =============================================
START_TIME=$(date +%s)
LOG_FILE="results/optimization_log_${TAG}_$(date +%Y%m%d_%H%M%S).log"
LOG_LEVEL="${LOG_LEVEL:-INFO}"  # INFO, DEBUG, or QUIET

# Logging function with timestamps
log() {
    local level="$1"
    local message="$2"
    local elapsed=$(($(date +%s) - START_TIME))
    local mins=$((elapsed / 60))
    local secs=$((elapsed % 60))
    local timestamp=$(date +"%H:%M:%S")
    local formatted="[$timestamp][+${mins}m${secs}s][$level] $message"
    
    # Always write to log file
    echo "$formatted" >> "$LOG_FILE"
    
    # Console output based on log level
    case "$LOG_LEVEL" in
        QUIET)
            [[ "$level" == "ERROR" || "$level" == "DONE" ]] && echo "$formatted"
            ;;
        INFO)
            [[ "$level" != "DEBUG" ]] && echo "$formatted"
            ;;
        DEBUG)
            echo "$formatted"
            ;;
    esac
}

# Progress bar function
progress_bar() {
    local current=$1
    local total=$2
    local pct=$((current * 100 / total))
    local filled=$((pct / 5))
    local empty=$((20 - filled))
    printf "\r  [%s%s] %d/%d (%d%%)" \
        "$(printf '█%.0s' $(seq 1 $filled 2>/dev/null))" \
        "$(printf '░%.0s' $(seq 1 $empty 2>/dev/null))" \
        "$current" "$total" "$pct"
}

# ETA calculator
calc_eta() {
    local current=$1
    local total=$2
    local start=$3
    local elapsed=$(($(date +%s) - start))
    
    if [[ $current -gt 0 ]]; then
        local rate=$((elapsed / current))
        local remaining=$(( (total - current) * rate ))
        local eta_mins=$((remaining / 60))
        local eta_secs=$((remaining % 60))
        echo "${eta_mins}m ${eta_secs}s"
    else
        echo "calculating..."
    fi
}

# Heartbeat function (run in background)
start_heartbeat() {
    (
        while true; do
            sleep 60
            local elapsed=$(($(date +%s) - START_TIME))
            local mins=$((elapsed / 60))
            log "HEARTBEAT" "Still running... (${mins} minutes elapsed)"
        done
    ) &
    HEARTBEAT_PID=$!
}

stop_heartbeat() {
    if [[ -n "$HEARTBEAT_PID" ]]; then
        kill $HEARTBEAT_PID 2>/dev/null
    fi
}

# Update cleanup to stop heartbeat
cleanup() {
    echo ""
    log "ERROR" "Interrupt received! Killing background jobs..."
    stop_heartbeat
    pids=$(jobs -p)
    if [ -n "$pids" ]; then
        kill $pids 2>/dev/null
    fi
    log "DONE" "Cleanup complete. Exiting."
    exit 1
}

# Re-register trap with updated cleanup
trap cleanup SIGINT SIGTERM

# Start heartbeat in background
start_heartbeat

log "INFO" "Detected $AVAIL_CORES CPU cores"
log "INFO" "Parallelization: $MAX_PARALLEL_COMBOS combinations × $JOBS_PER_COMBO Optuna jobs each"
log "INFO" "Log file: $LOG_FILE"

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

# Common arguments for train/test dates
TRAIN_TEST_ARGS="--train-start \"$TRAIN_START\" --train-end \"$TRAIN_END\" --test-start \"$TEST_START\" --test-end \"$TEST_END\""

echo ""
echo "========================================"
echo "Optimization Workflow: $TAG"
if [[ "$EXHAUSTIVE_MODE" == "true" ]]; then
  echo "Mode: Exhaustive (All Combinations)"
elif [[ "$WFO_MODE" == "true" ]]; then
  echo "Mode: Walk-Forward (Rolling)"
else
  echo "Mode: Static (Optuna Auto-Exploration)"
fi
echo "----------------------------------------"
echo "Train: $TRAIN_START -> $TRAIN_END"
echo "Test:  $TEST_START -> $TEST_END"
echo ""

mkdir -p results configs

# =============================================
# Step 1: Optimize Mean Reversion
# =============================================
OUT_MR_WFO_CSV="results/wfo_mr_${TAG}.csv"

if [[ "$WFO_MODE" == "true" && "$EXHAUSTIVE_MODE" == "true" ]]; then
  # --- WFO + EXHAUSTIVE MODE (Most Thorough) ---
  echo "[1/6] Optimizing Mean Reversion (WFO + Exhaustive)..."
  echo "  🚀 Rolling Windows: ${WINDOW_DAYS}d train + ${STEP_DAYS}d OOS"
  echo "  🔍 Testing ALL 8 combinations per window"
  echo "  ⚡ Parallel Limit: $MAX_PARALLEL_COMBOS concurrent jobs"
  
  # For each combo, run WFO and merge results
  COMBO_COUNT=0
  for trend in sma roc; do
    for sizing in static volatility; do
      for long_only in true false; do
        
        # Filter for Long-Only Futures Mode
        if [[ "$LONG_ONLY_FUTURES_MODE" == "true" ]]; then
           if [[ "$long_only" == "false" ]]; then continue; fi
        fi
        
        # Filter for Futures Only Mode (Shorts Allowed)
        if [[ "$FUTURES_ONLY_MODE" == "true" ]]; then
           if [[ "$long_only" == "true" ]]; then continue; fi
        fi
        
        # Throttle parallel jobs
        while [[ $(jobs -r | wc -l) -ge $MAX_PARALLEL_COMBOS ]]; do
           sleep 2
        done

        COMBO_COUNT=$((COMBO_COUNT + 1))
        long_str=$([ "$long_only" == "true" ] && echo "long" || echo "short")
        echo "  → [$COMBO_COUNT/8] Launching WFO: trend=$trend, sizing=$sizing, long_only=$long_only"
        
        # Run WFO for this combo in background
        (
          python3 tools/optimizer_cli.py \
            --data "$PRICE_DATA" \
            --funding-data "$FUNDING_DATA" \
            --wfo \
            --window-days $WINDOW_DAYS \
            --step-days $STEP_DAYS \
            --n-trials $MR_TRIALS \
            --jobs $JOBS_PER_COMBO \
            --force-trend-kind $trend \
            --force-sizing-mode $sizing \
            --force-long-only $long_only \
            --storage "sqlite:///data/db/optuna.db" \
            --study-name "mr_${TAG}_wfo_${trend}_${sizing}_${long_str}" \
            --out "results/wfo_mr_${TAG}_${trend}_${sizing}_${long_str}.csv" \
            2>&1 | sed "s/^/    [$trend-$sizing-$long_str] /"
        ) &
        
      done
    done
  done
  
  echo "  ⏳ Waiting for all WFO combinations to complete..."
  wait
  echo "  ✅ All WFO combinations finished!"
  
  # Merge all WFO results into single CSV
  echo "  → Merging WFO results..."
  {
    FIRST_FILE=$(find results -name "wfo_mr_${TAG}_*.csv" 2>/dev/null | head -n 1)
    if [[ -n "$FIRST_FILE" ]]; then
        head -1 "$FIRST_FILE"
        for trend in sma roc; do
          for sizing in static volatility; do
            for long_only in true false; do
              long_str=$([ "$long_only" == "true" ] && echo "long" || echo "short")
              tail -n +2 "results/wfo_mr_${TAG}_${trend}_${sizing}_${long_str}.csv" 2>/dev/null || true
            done
          done
        done
    fi
  } > "$OUT_MR_WFO_CSV"
  
  TOTAL_ROWS=$(($(wc -l < "$OUT_MR_WFO_CSV") - 1))
  echo "  ✅ WFO + Exhaustive complete ($TOTAL_ROWS window-combo results)"
  
  # Copy for downstream steps
  cp "$OUT_MR_WFO_CSV" "$OUT_MR_CSV"

elif [[ "$WFO_MODE" == "true" ]]; then
  # --- WFO MODE ONLY ---
  echo "[1/6] Optimizing Mean Reversion (Walk-Forward)..."
  echo "  🚀 Rolling Windows: ${WINDOW_DAYS}d train + ${STEP_DAYS}d OOS"
  python3 tools/optimizer_cli.py \
    --data "$PRICE_DATA" \
    --funding-data "$FUNDING_DATA" \
    --wfo \
    --window-days $WINDOW_DAYS \
    --step-days $STEP_DAYS \
    --n-trials $MR_TRIALS \
    --jobs $AVAIL_CORES \
    --storage "sqlite:///data/db/optuna.db" \
    --study-name "mr_${TAG}_wfo" \
    --out "$OUT_MR_WFO_CSV"
  
  # Copy WFO output as MR CSV for downstream steps
  cp "$OUT_MR_WFO_CSV" "$OUT_MR_CSV"

elif [[ "$EXHAUSTIVE_MODE" == "true" ]]; then
  echo "[1/6] Optimizing Mean Reversion (Exhaustive - All Combinations)..."
  echo "  ⚡ Parallel Limit: $MAX_PARALLEL_COMBOS concurrent jobs"
  
  # Test all 8 combinations
  COMBO_COUNT=0
  for trend in sma roc; do
    for sizing in static volatility; do
      for long_only in true false; do
        
        # Filter for Long-Only Futures Mode
        if [[ "$LONG_ONLY_FUTURES_MODE" == "true" ]]; then
           if [[ "$long_only" == "false" ]]; then continue; fi
        fi
        
        # Filter for Futures Only Mode (Shorts Allowed)
        if [[ "$FUTURES_ONLY_MODE" == "true" ]]; then
           if [[ "$long_only" == "true" ]]; then continue; fi
        fi
        
        # --- FIX: ROBUST THROTTLING ---
        # While number of running jobs >= max, sleep briefly
        while [[ $(jobs -r | wc -l) -ge $MAX_PARALLEL_COMBOS ]]; do
           sleep 2
        done
        # ------------------------------

        COMBO_COUNT=$((COMBO_COUNT + 1))
        long_str=$([ "$long_only" == "true" ] && echo "long" || echo "short")
        echo "  → [$COMBO_COUNT/8] Launching: trend=$trend, sizing=$sizing, long_only=$long_only"
        
        # Run in background
        (
          python3 tools/optimizer_cli.py \
            --data "$PRICE_DATA" \
            --funding-data "$FUNDING_DATA" \
            --train-start "$TRAIN_START" --train-end "$TRAIN_END" \
            --test-start "$TEST_START" --test-end "$TEST_END" \
            --n-trials $MR_TRIALS \
            --jobs $JOBS_PER_COMBO \
            --force-trend-kind $trend \
            --force-sizing-mode $sizing \
            --force-long-only $long_only \
            --storage "sqlite:///data/db/optuna.db" \
            --study-name "mr_${TAG}_${trend}_${sizing}_${long_str}" \
            --out "results/opt_mr_${TAG}_${trend}_${sizing}_${long_str}.csv" \
            2>&1 | sed "s/^/    [$trend-$sizing-$long_str] /"
        ) &
        
      done
    done
  done
  
  # Wait for all remaining background jobs to finish
  echo "  ⏳ Waiting for final jobs to complete..."
  wait
  echo "  ✅ All combinations finished!"
  
  # Merge all results into single CSV
  echo "  → Merging results..."
  {
    # Get header from a file that exists (using find to be safe)
    FIRST_FILE=$(find results -name "opt_mr_${TAG}_*.csv" | head -n 1)
    if [[ -n "$FIRST_FILE" ]]; then
        head -1 "$FIRST_FILE"
        # Append all data (skip headers)
        for trend in sma roc; do
          for sizing in static volatility; do
            for long_only in true false; do
              long_str=$([ "$long_only" == "true" ] && echo "long" || echo "short")
              tail -n +2 "results/opt_mr_${TAG}_${trend}_${sizing}_${long_str}.csv" 2>/dev/null || true
            done
          done
        done
    fi
  } > "$OUT_MR_CSV"
  
  TOTAL_ROWS=$(($(wc -l < "$OUT_MR_CSV") - 1))
  echo "  ✅ Exhaustive optimization complete ($TOTAL_ROWS configs tested)"
  
else
  # Normal mode: Let Optuna explore automatically with parallel jobs
  echo "[1/6] Optimizing Mean Reversion..."
  echo "  ⚡ Using $AVAIL_CORES parallel Optuna jobs"
  python3 tools/optimizer_cli.py \
    --data "$PRICE_DATA" \
    --funding-data "$FUNDING_DATA" \
    --train-start "$TRAIN_START" --train-end "$TRAIN_END" \
    --test-start "$TEST_START" --test-end "$TEST_END" \
    --n-trials $MR_TRIALS \
    --jobs $AVAIL_CORES \
    --study-name "mr_${TAG}_study" \
    --storage "sqlite:///data/db/optuna.db" \
    --out "$OUT_MR_CSV"
fi

# Step 2: Pick Best MR
echo "[2/6] Selecting Best MR..."
if [[ "$WFO_MODE" == "true" ]]; then
  # Use smart WFO selection for MR
  python3 tools/wfo_select_best.py \
    --wfo-csv "$OUT_MR_WFO_CSV" \
    --out "$OUT_MR_PARAMS" \
    --strategy weighted
else
  python3 tools/wf_pick.py \
    --runs "$OUT_MR_CSV" \
    --emit-config "$OUT_MR_PARAMS" \
    --family-index 0 --min-occurs 1
fi

# Wrap MR Config
python3 - <<PYTHON
import json
with open("$OUT_MR_PARAMS") as f: params = json.load(f)
for k in ["_generated_by", "_generated_at", "_family"]: params.pop(k, None)
config = {
    "fees": { "maker_fee": 0.0002, "taker_fee": 0.0004, "slippage_bps": 1.0, "bnb_discount": 0.25, "pay_fees_in_bnb": True },
    "strategy": params,
    "execution": { "interval": "15m", "poll_sec": 5 },
    "risk": { "basis_btc": 1.0, "risk_mode": "fixed_basis", "drawdown_reset_days": 7.0, "drawdown_reset_score": 30.0}
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

if [ "$WFO_MODE" = true ]; then
    # --- WFO PATH ---
    # Note: WFO ignores train/test dates because it slices the whole file.
    # We pass them anyway for consistency but the script handles slicing.
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

    echo "[4/6] Smart WFO Selection..."
    python3 tools/wfo_select_best.py \
      --wfo-csv "$OUT_TR_WFO_CSV" \
      --out "$OUT_TR_PARAMS" \
      --strategy weighted
    echo ""
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

# Wrap Trend Config
python3 - <<PYTHON
import json
with open("$OUT_TR_PARAMS") as f: params = json.load(f)
for k in ["_generated_by", "_generated_at", "_family"]: params.pop(k, None)

config = {
    "fees": { "maker_fee": 0.0002, "taker_fee": 0.0004, "slippage_bps": 1.0, "bnb_discount": 0.25, "pay_fees_in_bnb": True },
    "strategy": params,
    "execution": { "interval": "15m", "poll_sec": 5 },
    "risk": { "basis_btc": 1.0, "risk_mode": "fixed_basis", "drawdown_reset_days": 7.0, "drawdown_reset_score": 30.0 }
}
config["strategy"]["strategy_type"] = "trend"

if "long_only" not in config["strategy"]: config["strategy"]["long_only"] = True
if "step_allocation" not in config["strategy"]: config["strategy"]["step_allocation"] = 1.0
if "max_position" not in config["strategy"]: config["strategy"]["max_position"] = 1.0
if "rebalance_threshold_w" not in config["strategy"]: config["strategy"]["rebalance_threshold_w"] = 0.03

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

# =============================================
# Step 7: Walk-Forward Audit (Optional Validation)
# =============================================
# Runs WFO analyzer on both MR and Trend results when WFO mode is used.
if [[ "$WFO_MODE" == "true" ]]; then
  log "INFO" "Running Walk-Forward Validation..."
  
  # Analyze MR WFO
  if [[ -f "$OUT_MR_WFO_CSV" ]]; then
    log "INFO" "  → Analyzing MR strategy..."
    python3 tools/wfo_analyzer.py \
      --wfo-csv "$OUT_MR_WFO_CSV" \
      --out "results/wfo_analysis_mr_${TAG}.json"
  fi
  
  # Analyze Trend WFO
  if [[ -f "$OUT_TR_WFO_CSV" ]]; then
    log "INFO" "  → Analyzing Trend strategy..."
    python3 tools/wfo_analyzer.py \
      --wfo-csv "$OUT_TR_WFO_CSV" \
      --out "results/wfo_analysis_trend_${TAG}.json"
  fi
fi

# Stop heartbeat
stop_heartbeat

# Final Summary
ELAPSED=$(($(date +%s) - START_TIME))
ELAPSED_MINS=$((ELAPSED / 60))
ELAPSED_SECS=$((ELAPSED % 60))

echo ""
echo "════════════════════════════════════════════════════════════"
log "DONE" "Optimization Complete!"
echo "════════════════════════════════════════════════════════════"
echo "  Tag:           ${TAG}"
echo "  Total Time:    ${ELAPSED_MINS}m ${ELAPSED_SECS}s"
echo "  Config:        ${FINAL_CONFIG}"
echo "  Log File:      ${LOG_FILE}"
if [[ "$WFO_MODE" == "true" ]]; then
  echo ""
  echo "  WFO Analysis:"
  echo "    MR:    results/wfo_analysis_mr_${TAG}.json"
  echo "    Trend: results/wfo_analysis_trend_${TAG}.json"
fi
echo "════════════════════════════════════════════════════════════"
echo ""