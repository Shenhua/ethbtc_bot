#!/usr/bin/env python3
import sys
import os

# --- MAGIC PATH FIX ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ----------------------

import argparse, json, ast
from datetime import datetime, timezone
import pandas as pd
import numpy as np

# Keys to preserve in the final JSON config
MR_PARAMS = [
    "trend_kind", "trend_lookback",
    "flip_band_entry", "flip_band_exit",
    "vol_window", "vol_adapt_k", "target_vol",
    "min_mult", "max_mult", "gate_window_days", "gate_roc_threshold"
]

TREND_PARAMS = [
    "fast_period", "slow_period", "ma_type"
]

SHARED_PARAMS = [
    "cooldown_minutes", "step_allocation", "max_position",
    "long_only", "rebalance_threshold_w",
    "funding_limit_long", "funding_limit_short", "strategy_type",
    # Dynamic Position Sizing
    "position_sizing_mode", "position_sizing_target_vol",
    "position_sizing_min_step", "position_sizing_max_step",
    "kelly_win_rate", "kelly_avg_win", "kelly_avg_loss"
]

FEE_KEYS = ["maker_fee", "taker_fee", "slippage_bps", "bnb_discount", "pay_fees_in_bnb"]

def _to_native(v):
    if isinstance(v, (np.integer, int)): return int(v)
    if isinstance(v, (np.floating, float)): return float(v)
    if isinstance(v, (np.bool_, bool)): return bool(v)
    return v

def _parse_params_series(ser: pd.Series) -> pd.DataFrame:
    def parse_one(x):
        if isinstance(x, dict): return x
        if not isinstance(x, str): return {}
        x = x.strip()
        try: return json.loads(x)
        except Exception: pass
        try: return ast.literal_eval(x)
        except Exception: return {}
    obj = ser.apply(parse_one)
    try:
        return pd.json_normalize(obj)
    except Exception:
        return pd.DataFrame(list(obj))

def load_and_flatten(paths):
    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            df["__source"] = os.path.basename(p)
            
            # Handle nested "params" column if it exists (Optuna JSON style)
            if "params" in df.columns:
                pexp = _parse_params_series(df["params"])
                for c in pexp.columns:
                    if c not in df.columns:
                        df[c] = pexp[c]
            
            frames.append(df)
        except Exception as e:
            print(f"Error loading {p}: {e}")
            
    if not frames:
        return pd.DataFrame()
        
    out = pd.concat(frames, ignore_index=True)
    
    # --- FIX: Normalize Column Aliases ---
    # Different optimizers output different names. We standardize them here.
    aliases = {
        "test_profit": "test_final_btc",
        "train_profit": "train_final_btc",
        "turns": "turns_test",
        "final_btc": "test_final_btc"
    }
    out.rename(columns=aliases, inplace=True)
    
    # --- FIX: Deduplicate columns after rename ---
    # If multiple columns map to 'test_final_btc', we keep the first one.
    out = out.loc[:, ~out.columns.duplicated()]
    
    # Ensure required columns exist (fill with NaN/0 if missing)
    expected_cols = MR_PARAMS + TREND_PARAMS + SHARED_PARAMS + FEE_KEYS + [
        "test_final_btc", "train_final_btc", "fees_btc", "turnover_btc", "turns_test"
    ]
    for c in expected_cols:
        if c not in out.columns:
            out[c] = np.nan
            
    return out

def reasonable_filter(df, q_fee=0.75, q_turnover=0.75):
    df = df.copy()
    filt = pd.Series(True, index=df.index)
    
    # Only filter if we actually have fee data
    if df["fees_btc"].notna().sum() > 5:
        thr_fees = df["fees_btc"].quantile(q_fee)
        filt &= (df["fees_btc"] <= thr_fees) | df["fees_btc"].isna()
        
    return df[filt].copy()

def make_family_key(row, band_round=4, step_round=2, lb_bucket=40, cd_bucket=60):
    """
    Polymorphic Grouping:
    - If Trend Strategy (has fast/slow): Group by (Fast, Slow, MA, Cooldown, LongOnly)
    - If Mean Reversion: Group by (Entry, Exit, Lookback, Cooldown, LongOnly)
    """
    long_only = bool(row.get("long_only", True))

    # 1. Check for Trend Strategy
    fast = row.get("fast_period", np.nan)
    slow = row.get("slow_period", np.nan)

    if pd.notna(fast) and pd.notna(slow) and float(slow) > 0:
        ma = row.get("ma_type", "ema")
        cd = row.get("cooldown_minutes", np.nan)

        def _bucket(v, b):
            if pd.isna(v) or b <= 1:
                return v
            return int(round(float(v) / b) * b)

        cd_key = _bucket(cd, cd_bucket) if not pd.isna(cd) else 0

        return ("Trend", int(fast), int(slow), ma, cd_key, long_only)

    # 2. Mean Reversion family
    e = row.get("flip_band_entry", np.nan)
    x = row.get("flip_band_exit", np.nan)
    tlb = row.get("trend_lookback", np.nan)
    cd  = row.get("cooldown_minutes", np.nan)

    def _bucket(v, b):
        if pd.isna(v) or b <= 1:
            return v
        return int(round(float(v) / b) * b)

    e_key = round(float(e), band_round) if not pd.isna(e) else None
    x_key = round(float(x), band_round) if not pd.isna(x) else None
    tlb_key = _bucket(tlb, lb_bucket) if not pd.isna(tlb) else None
    cd_key  = _bucket(cd,  cd_bucket) if not pd.isna(cd)  else None

    return ("MR", e_key, x_key, tlb_key, cd_key, long_only)

def rank_families(df, penalty_turns=0.0, penalty_fees=1.0, penalty_turnover=0.5, disp_weight=0.25,
                  band_round=4, step_round=2, lb_bucket=40, cd_bucket=60,
                  consistency_weight=0.3):
    """
    Rank families by robustness score.
    """
    df = df.copy()
    
    # --- FIX START: Ensure numeric types before aggregation ---
    cols_to_numeric = ["test_final_btc", "train_final_btc", "fees_btc", "turnover_btc", "turns_test"]
    for c in cols_to_numeric:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = 0.0  # Create missing columns to prevent KeyError/AttributeError
    # --- FIX END ---

    df["__family"] = df.apply(
        lambda row: make_family_key(row, band_round, step_round, lb_bucket, cd_bucket), axis=1
    )
    
    # Groupby
    g = df.groupby("__family", dropna=False)
    
    # Aggregation - explicit dictionary to avoid ambiguity
    # Aggregation - using dict syntax to avoid pandas bug with named aggregation
    agg = g.agg({
        "test_final_btc": ["mean", "size", "std"],
        "train_final_btc": "mean",
        "fees_btc": "mean",
        "turnover_btc": "mean",
        "turns_test": "mean"
    }).reset_index()

    # Flatten MultiIndex columns
    new_cols = []
    for c in agg.columns:
        if isinstance(c, tuple):
            if c[1] == "": new_cols.append(c[0])
            else: new_cols.append(f"{c[0]}_{c[1]}")
        else:
            new_cols.append(c)
    agg.columns = new_cols

    # Rename to match expected output
    # Rename to match expected output
    agg.rename(columns={
        "test_final_btc_mean": "mean_test_btc",
        "test_final_btc_size": "n",
        "test_final_btc_std": "std_test",
        "train_final_btc_mean": "mean_train_btc",
        "fees_btc_mean": "fees",
        "turnover_btc_mean": "turnover",
        "turns_test_mean": "turns"
    }, inplace=True)

    # NEW: Calculate train/test consistency
    # Penalize configs where test >> train (overfitting) or test << train (doesn't generalize)
    agg["train_test_gap"] = abs(agg["mean_test_btc"] - agg["mean_train_btc"])
    agg["train_test_ratio"] = agg["mean_test_btc"] / agg["mean_train_btc"].replace(0, 1e-9)
    
    # Harmonic mean of train and test (only rewards if BOTH are good)
    agg["harmonic_mean"] = 2 * (agg["mean_test_btc"] * agg["mean_train_btc"]) / \
                           (agg["mean_test_btc"] + agg["mean_train_btc"] + 1e-9)
    
    # NEW SCORING: Base on harmonic mean + consistency penalties
    agg["score"] = (
        agg["harmonic_mean"]  # Requires BOTH train & test to be good
        - consistency_weight * agg["train_test_gap"]  # Penalize large gaps
        - penalty_turns * (agg["turns"].fillna(0) / 800.0)
        - penalty_fees * agg["fees"].fillna(0)
        - penalty_turnover * agg["turnover"].fillna(0)
        - disp_weight * agg["std_test"].fillna(0)
    )
    
    # Additional filter: Reject if test/train ratio is too extreme
    # Ratio > 1.5 means test is 50% better than train (overfitting red flag!)
    # Ratio < 0.7 means test is 30% worse than train (doesn't generalize)
    agg["suspicious"] = (agg["train_test_ratio"] > 1.5) | (agg["train_test_ratio"] < 0.7)
    
    agg = agg.sort_values(["score", "harmonic_mean", "n"], ascending=[False, False, False])
    return agg

def pick_representative(df_all, family_tuple, band_round=4, step_round=2, lb_bucket=40, cd_bucket=60):
    df = df_all.copy()
    df["__family"] = df.apply(
        lambda row: make_family_key(row, band_round, step_round, lb_bucket, cd_bucket), axis=1
    )
    pool = df[df["__family"] == family_tuple].copy()
    if pool.empty: return None
    
    for c in ["fees_btc", "turnover_btc"]:
        if c not in pool.columns: pool[c] = 0.0
        pool[c] = pool[c].fillna(0.0)

    # Pick best row based on raw profit
    pool["__row_score"] = pool["test_final_btc"].fillna(-1e9)
    best = pool.sort_values("__row_score", ascending=False).iloc[0]
    return best.to_dict()

def build_config_from_row(row: dict, include_fees=True):
    cfg = {}
    
    # 1. Detect strategy type
    fast = row.get("fast_period")
    has_trend_params = fast is not None and not np.isnan(fast) if isinstance(fast, float) else fast is not None
    
    flip = row.get("flip_band_entry")
    has_mr_params = flip is not None and not np.isnan(flip) if isinstance(flip, float) else flip is not None
    
    # 2. Determine which params to copy
    if has_trend_params and not has_mr_params:
        # Trend strategy
        relevant_params = TREND_PARAMS + SHARED_PARAMS
        cfg["strategy_type"] = "trend"
    elif has_mr_params and not has_trend_params:
        # Mean Reversion strategy
        relevant_params = MR_PARAMS + SHARED_PARAMS
        cfg["strategy_type"] = "mean_reversion"
    else:
        # Ambiguous or both - fall back to all params (shouldn't happen with clean data)
        relevant_params = MR_PARAMS + TREND_PARAMS + SHARED_PARAMS
        # Infer from presence
        if has_trend_params:
            cfg["strategy_type"] = "trend"
        elif has_mr_params:
            cfg["strategy_type"] = "mean_reversion"
    
    # 3. Copy only relevant params
    for k in relevant_params:
        v = row.get(k)
        if v is not None and v == v:  # not NaN
            cfg[k] = _to_native(v)
            
    # 4. Copy Fees
    if include_fees:
        for k in FEE_KEYS:
            v = row.get(k)
            if v is not None and v == v:
                cfg[k] = _to_native(v)
    return cfg

def parse_family_literal(s):
    if not isinstance(s, str): return s
    try:
        tpl = ast.literal_eval(s)
        if isinstance(tpl, tuple): return tpl
    except Exception:
        pass
    return s

def preflight(df_all):
    # Basic checks
    tfb_u = df_all["test_final_btc"].dropna().nunique() if "test_final_btc" in df_all.columns else 0
    return {"test_final_btc_uniq": tfb_u}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--out-csv", default="results/wf_ranked_families.csv")
    ap.add_argument("--emit-config", default="configs/selected_params.json")
    ap.add_argument("--min-occurs", type=int, default=1)
    ap.add_argument("--family-index", type=int, default=0)
    
    # Coarsening defaults
    ap.add_argument("--band-round", type=int, default=4)
    ap.add_argument("--step-round", type=int, default=2)
    ap.add_argument("--lb-bucket", type=int, default=40)
    ap.add_argument("--cd-bucket", type=int, default=60)
    
    # Scoring penalties (defaults)
    ap.add_argument("--q-fees", type=float, default=1.0)
    ap.add_argument("--q-turnover", type=float, default=1.0)
    ap.add_argument("--penalty-turns", type=float, default=0.0)
    ap.add_argument("--penalty-fees", type=float, default=0.0)
    ap.add_argument("--penalty-turnover", type=float, default=0.0)
    ap.add_argument("--disp-weight", type=float, default=0.0)
    ap.add_argument(
        "--force-long-only",
        action="store_true",
        help="Drop all trials with long_only = False before ranking families."
    )
    ap.add_argument(
        "--sanity-data",
        help="If provided, run a full backtest on this OHLC CSV using the emitted config.",
    )
    ap.add_argument(
        "--sanity-funding",
        help="Optional funding CSV with 'time' and 'rate' for sanity backtest.",
    )
    args = ap.parse_args()

    df_all = load_and_flatten(args.runs)
    print("Preflight:", preflight(df_all))

    if args.force_long_only and "long_only" in df_all.columns:
        df_all = df_all[df_all["long_only"].astype(bool)]
        print(f"Filtered to long_only=True: {len(df_all)} rows remaining.")

    df = reasonable_filter(df_all, q_fee=args.q_fees, q_turnover=args.q_turnover)
    
    fam = rank_families(
        df,
        args.penalty_turns, args.penalty_fees, args.penalty_turnover, args.disp_weight,
        args.band_round, args.step_round, args.lb_bucket, args.cd_bucket
    )
    
    # --- NEW: Kill Zombie Strategies ---
    # A strategy with 0 trades is mathematically "safe" (score 1.0) but useless.
    # We insist on at least 1 trade on average.
    fam = fam[fam["turns"] >= 1.0]
    
    fam = fam[fam["n"] >= args.min_occurs]
    
    if fam.empty:
        print("No families found.", file=sys.stderr)
        sys.exit(3)

    fam.to_csv(args.out_csv, index=False)
    print(f"Top Families saved to {args.out_csv}")
    print(fam.head(args.top_k).to_string(index=False))

    # Pick best config
    idx = max(0, min(int(args.family_index), len(fam)-1))
    fam_tuple = parse_family_literal(fam.iloc[idx]["__family"])
    
    print(f"\nSelecting Family: {fam_tuple}")

    best_row = pick_representative(df, fam_tuple,
        args.band_round, args.step_round, args.lb_bucket, args.cd_bucket
    )
    
    cfg = build_config_from_row(best_row)

    with open(args.emit_config, "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"Wrote config → {args.emit_config}")

    # Optional: run full sanity backtest
    if args.sanity_data:
        try:
            from sanity_check_config import run_sanity_check
        except ImportError:
            print("Warning: sanity_check_config not importable; skipping sanity backtest.")
        else:
            print("Running sanity backtest on full data...")
            summary = run_sanity_check(
                data_path=args.sanity_data,
                config_path=args.emit_config,
                funding_path=args.sanity_funding,
            )
            print("Sanity backtest summary:")
            print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()