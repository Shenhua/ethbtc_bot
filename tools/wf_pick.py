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
RICH_PARAM_KEYS = [
    # Mean Reversion
    "trend_kind", "trend_lookback",
    "flip_band_entry", "flip_band_exit",
    "vol_window", "vol_adapt_k", "target_vol",
    "min_mult", "max_mult", "gate_window_days", "gate_roc_threshold",
    
    # Trend
    "fast_period", "slow_period", "ma_type",
    
    # Shared
    "cooldown_minutes", "step_allocation", "max_position",
    "long_only", "rebalance_threshold_w",
    "funding_limit_long", "funding_limit_short", "strategy_type"
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
    
    # Ensure required columns exist (fill with NaN/0 if missing)
    expected_cols = RICH_PARAM_KEYS + FEE_KEYS + [
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
    - If Trend Strategy (has fast/slow): Group by (Fast, Slow, MA, Cooldown)
    - If Mean Reversion: Group by (Entry, Exit, Lookback, Cooldown)
    """
    # 1. Check for Trend Strategy
    fast = row.get("fast_period", np.nan)
    slow = row.get("slow_period", np.nan)
    
    if pd.notna(fast) and pd.notna(slow) and float(slow) > 0:
        # TREND FAMILY
        ma = row.get("ma_type", "ema")
        cd = row.get("cooldown_minutes", np.nan)
        
        def _bucket(v, b):
            if pd.isna(v) or b <= 1: return v
            return int(round(float(v) / b) * b)
        
        cd_key = _bucket(cd, cd_bucket) if not pd.isna(cd) else 0
        
        # Tuple: (Type="Trend", Fast, Slow, MA, Cooldown)
        return ("Trend", int(fast), int(slow), ma, cd_key)

    else:
        # MEAN REVERSION FAMILY
        e = row.get("flip_band_entry", np.nan)
        x = row.get("flip_band_exit", np.nan)
        tlb = row.get("trend_lookback", np.nan)
        cd  = row.get("cooldown_minutes", np.nan)

        def _bucket(v, b):
            if pd.isna(v) or b <= 1: return v
            return int(round(float(v) / b) * b)

        e_key = round(float(e), band_round) if not pd.isna(e) else None
        x_key = round(float(x), band_round) if not pd.isna(x) else None
        tlb_key = _bucket(tlb, lb_bucket) if not pd.isna(tlb) else None
        cd_key  = _bucket(cd,  cd_bucket) if not pd.isna(cd)  else None

        # Tuple: (Type="MR", Entry, Exit, Lookback, Cooldown)
        return ("MR", e_key, x_key, tlb_key, cd_key)

def rank_families(df, penalty_turns=0.0, penalty_fees=1.0, penalty_turnover=0.5, disp_weight=0.25,
                  band_round=4, step_round=2, lb_bucket=40, cd_bucket=60):
    df = df.copy()
    df["__family"] = df.apply(
        lambda row: make_family_key(row, band_round, step_round, lb_bucket, cd_bucket), axis=1
    )
    g = df.groupby("__family", dropna=False)
    
    # Aggregation
    agg = g.agg(
        mean_test_btc=("test_final_btc","mean"),
        mean_train_btc=("train_final_btc","mean"),
        fees=("fees_btc","mean"),
        turnover=("turnover_btc","mean"),
        turns=("turns_test","mean"),
        n=("test_final_btc","size"),
        std_test=("test_final_btc","std")
    ).reset_index()

    # Scoring
    agg["score"] = (
        agg["mean_test_btc"]
        - penalty_turns * (agg["turns"].fillna(0) / 800.0)
        - penalty_fees * agg["fees"].fillna(0)
        - penalty_turnover * agg["turnover"].fillna(0)
        - disp_weight * agg["std_test"].fillna(0)
    )
    
    agg = agg.sort_values(["score", "mean_test_btc", "n"], ascending=[False, False, False])
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
    
    # 1. Copy Strategy Params
    for k in RICH_PARAM_KEYS:
        v = row.get(k)
        if v is not None and v == v: # not NaN
            cfg[k] = _to_native(v)

    # 2. Infer Strategy Type if missing
    if "fast_period" in cfg:
        cfg["strategy_type"] = "trend"
    elif "trend_kind" in cfg:
        cfg["strategy_type"] = "mean_reversion"
            
    # 3. Copy Fees
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
    
    args = ap.parse_args()

    df_all = load_and_flatten(args.runs)
    print("Preflight:", preflight(df_all))

    df = reasonable_filter(df_all, q_fee=args.q_fees, q_turnover=args.q_turnover)
    
    fam = rank_families(
        df,
        args.penalty_turns, args.penalty_fees, args.penalty_turnover, args.disp_weight,
        args.band_round, args.step_round, args.lb_bucket, args.cd_bucket
    )
    
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
    cfg.update({
        "_generated_by": "wf_pick.py",
        "_generated_at": datetime.now(timezone.utc).isoformat(),
        "_family": str(fam_tuple),
    })
    
    with open(args.emit_config, "w") as f:
        json.dump(cfg, f, indent=2)
    
    print(f"Wrote config → {args.emit_config}")

if __name__ == "__main__":
    main()