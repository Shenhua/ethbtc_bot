
#!/usr/bin/env python3
"""
wf_pick.py — Walk-forward family picker (stable, cost-aware)
Version: 1.3.0 (2025-10-29)

Key features
- Robust "params" parsing (JSON first, then ast.literal_eval)
- Reasonable fee/turnover filter (quantiles)
- Family definition: (flip_band_entry, flip_band_exit, trend_lookback, cooldown_minutes, step_allocation)
  * with configurable rounding/bucketing via CLI:
      --band-round, --step-round, --lb-bucket, --cd-bucket
- Preflight checks: require variation in test_final_btc and costs if desired
- min-occurs filter: require families to repeat across runs to be considered
- Ranking uses mean test_final_btc, cost penalties, dispersion, and a tie-break on fewer turns
- Emits selected_params.json with rich fields + legacy flip_band mirror
"""

import argparse, json, ast, os, sys
from datetime import datetime, timezone
import pandas as pd
import numpy as np

RICH_PARAM_KEYS = [
    "trend_kind","trend_lookback",
    "flip_band_entry","flip_band_exit",
    "vol_window","vol_adapt_k","target_vol","min_mult","max_mult",
    "cooldown_minutes","step_allocation","max_position",
    "gate_window_days","gate_roc_threshold",
]
FEE_KEYS = ["maker_fee","taker_fee","slippage_bps","bnb_discount","pay_fees_in_bnb"]

def _to_native(v):
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating,)): return float(v)
    if isinstance(v, (np.bool_,)): return bool(v)
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
        df = pd.read_csv(p)
        df["__source"] = os.path.basename(p)
        if "params" in df.columns:
            pexp = _parse_params_series(df["params"])
            for c in pexp.columns:
                if c not in df.columns:
                    df[c] = pexp[c]
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)

    # Ensure columns exist
    for k in ["flip_band_entry","flip_band_exit","flip_band","trend_lookback",
              "cooldown_minutes","step_allocation","max_position",
              "vol_window","vol_adapt_k","target_vol",
              "gate_window_days","gate_roc_threshold","trend_kind"]:
        if k not in out.columns: out[k] = np.nan
    for k in ["fees_btc","turnover_btc","turns_test","test_final_btc","train_final_btc"]:
        if k not in out.columns: out[k] = np.nan
    return out

def reasonable_filter(df, q_fee=0.75, q_turnover=0.75):
    df = df.copy()
    filt = pd.Series(True, index=df.index)
    if "fees_btc" in df.columns and df["fees_btc"].notna().any():
        thr_fees = df["fees_btc"].quantile(q_fee)
        filt &= (df["fees_btc"] <= thr_fees) | df["fees_btc"].isna()
    if "turnover_btc" in df.columns and df["turnover_btc"].notna().any():
        thr_t = df["turnover_btc"].quantile(q_turnover)
        filt &= (df["turnover_btc"] <= thr_t) | df["turnover_btc"].isna()
    return df[filt].copy()

def make_family_key(row, band_round=4, step_round=2, lb_bucket=40, cd_bucket=60):
    e = row.get("flip_band_entry", np.nan)
    x = row.get("flip_band_exit", np.nan)
    fb = row.get("flip_band", np.nan)
    if (pd.isna(e) or pd.isna(x)) and not pd.isna(fb):
        if pd.isna(e): e = fb
        if pd.isna(x): x = fb
    tlb = row.get("trend_lookback", np.nan)
    cd  = row.get("cooldown_minutes", np.nan)
    sa  = row.get("step_allocation", np.nan)

    def _bucket(v, b):
        if pd.isna(v) or b <= 1: return v
        return int(round(float(v) / b) * b)

    e_key = round(float(e), band_round) if not pd.isna(e) else None
    x_key = round(float(x), band_round) if not pd.isna(x) else None
    tlb_key = _bucket(tlb, lb_bucket) if not pd.isna(tlb) else None
    cd_key  = _bucket(cd,  cd_bucket) if not pd.isna(cd)  else None
    sa_key  = round(float(sa), step_round) if not pd.isna(sa) else None

    return (e_key, x_key, tlb_key, cd_key, sa_key)

def rank_families(df, penalty_turns=0.0, penalty_fees=1.0, penalty_turnover=0.5, disp_weight=0.25,
                  band_round=4, step_round=2, lb_bucket=40, cd_bucket=60):
    df = df.copy()
    df["__family"] = df.apply(
        lambda row: make_family_key(row, band_round, step_round, lb_bucket, cd_bucket), axis=1
    )
    g = df.groupby("__family", dropna=False)
    agg = g.agg(mean_test_btc=("test_final_btc","mean"),
                mean_train_btc=("train_final_btc","mean"),
                fees=("fees_btc","mean"),
                turnover=("turnover_btc","mean"),
                turns=("turns_test","mean"),
                n=("test_final_btc","size"),
                std_test=("test_final_btc","std")).reset_index()
    agg["score"] = (agg["mean_test_btc"]
                    - penalty_turns * (agg["turns"].fillna(0) / 800.0)
                    - penalty_fees * agg["fees"].fillna(0)
                    - penalty_turnover * agg["turnover"].fillna(0)
                    - disp_weight * agg["std_test"].fillna(0))
    agg = agg.sort_values(["score","mean_test_btc","n","turns"], ascending=[False,False,False,True])
    return agg

def pick_representative(df_all, family_tuple, band_round=4, step_round=2, lb_bucket=40, cd_bucket=60):
    df = df_all.copy()
    df["__family"] = df.apply(
        lambda row: make_family_key(row, band_round, step_round, lb_bucket, cd_bucket), axis=1
    )
    pool = df[df["__family"] == family_tuple].copy()
    if pool.empty: return None
    for c in ["fees_btc","turnover_btc"]:
        if c not in pool.columns: pool[c] = 0.0
        pool[c] = pool[c].fillna(0.0)
    pool["__row_score"] = pool["test_final_btc"].fillna(-1e9) - 1.0*pool["fees_btc"] - 0.5*pool["turnover_btc"]
    best = pool.sort_values(["__row_score","test_final_btc","turns_test"], ascending=[False,False,True]).iloc[0]
    return best.to_dict()

def build_config_from_row(row: dict, include_fees=True, mirror_legacy=True):
    cfg = {}
    for k in RICH_PARAM_KEYS:
        v = row.get(k)
        if v is not None and v == v:
            cfg[k] = _to_native(v)
    fb = row.get("flip_band")
    if ("flip_band_entry" not in cfg or "flip_band_exit" not in cfg) and fb is not None and fb == fb:
        cfg.setdefault("flip_band_entry", _to_native(fb))
        cfg.setdefault("flip_band_exit",  _to_native(fb))
    if mirror_legacy and "flip_band_entry" in cfg and "flip_band" not in cfg:
        cfg["flip_band"] = _to_native(cfg["flip_band_entry"])
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

def preflight(df_all, require_costs=False, require_var=False):
    msgs = []
    tfb_u = df_all["test_final_btc"].dropna().nunique() if "test_final_btc" in df_all.columns else 0
    fees_u = df_all["fees_btc"].dropna().nunique() if "fees_btc" in df_all.columns else 0
    turn_u = df_all["turnover_btc"].dropna().nunique() if "turnover_btc" in df_all.columns else 0
    if tfb_u <= 1:
        msg = "WARNING: test_final_btc has ≤1 unique value across rows."
        if require_var: msg = "ERROR: " + msg
        msgs.append(msg)
    if (fees_u <= 1 or turn_u <= 1):
        msg = "WARNING: fees_btc/turnover_btc missing or constant."
        if require_costs: msg = "ERROR: " + msg
        msgs.append(msg)
    return msgs, {"test_final_btc_uniq": tfb_u, "fees_btc_uniq": fees_u, "turnover_btc_uniq": turn_u}

def main():
    ap = argparse.ArgumentParser(description="Pick stable optimizer configs across multiple runs and emit selected_params.json")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--q-fees", type=float, default=0.75)
    ap.add_argument("--q-turnover", type=float, default=0.75)
    ap.add_argument("--penalty-turns", type=float, default=0.0)
    ap.add_argument("--penalty-fees", type=float, default=1.0)
    ap.add_argument("--penalty-turnover", type=float, default=0.5)
    ap.add_argument("--disp-weight", type=float, default=0.25)
    ap.add_argument("--band-round", type=int, default=4)
    ap.add_argument("--step-round", type=int, default=2)
    ap.add_argument("--lb-bucket", type=int, default=40, help="Bucket size for trend_lookback")
    ap.add_argument("--cd-bucket", type=int, default=60, help="Bucket size (minutes) for cooldown_minutes")
    ap.add_argument("--min-occurs", type=int, default=2, help="Require families to appear in at least N rows")
    ap.add_argument("--require-costs", action="store_true", help="Fail if costs are missing/constant")
    ap.add_argument("--require-var", action="store_true", help="Fail if test_final_btc lacks variation")
    ap.add_argument("--top-k", type=int, default=12)
    ap.add_argument("--out-csv", default="wf_ranked_families.csv")
    ap.add_argument("--excel-out")
    ap.add_argument("--emit-config", default="selected_params.json")
    ap.add_argument("--family-index", type=int, default=0)
    args = ap.parse_args()

    df_all = load_and_flatten(args.runs)
    msgs, metrics = preflight(df_all, require_costs=args.require_costs, require_var=args.require_var)
    print("Preflight:", metrics)
    for m in msgs: print(m)
    if any(m.startswith("ERROR:") for m in msgs):
        sys.exit(2)

    base_n = len(df_all)
    df = reasonable_filter(df_all, q_fee=args.q_fees, q_turnover=args.q_turnover)
    print(f"Loaded rows: {base_n} | after reasonable filter: {len(df)}")
    print(f"Coarsening → band_round={args.band_round}, step_round={args.step_round}, lb_bucket={args.lb_bucket}, cd_bucket={args.cd_bucket}")

    fam = rank_families(
        df,
        args.penalty_turns, args.penalty_fees, args.penalty_turnover, args.disp_weight,
        args.band_round, args.step_round, args.lb_bucket, args.cd_bucket
    )
    before = len(fam)
    fam = fam[fam["n"] >= args.min_occurs]
    print(f"Families kept with n >= {args.min_occurs}: {len(fam)} (dropped {before - len(fam)})")

    if fam.empty:
        print("No families to choose from after n>=min-occurs filter.", file=sys.stderr)
        sys.exit(3)

    fam.to_csv(args.out_csv, index=False)
    print(f"Wrote family ranking → {args.out_csv}\n")
    print("Top families:")
    print(fam.head(args.top_k).to_string(index=False))

    idx = max(0, min(int(args.family_index), len(fam)-1))
    fam_tuple = parse_family_literal(fam.iloc[idx]["__family"])

    best_row = pick_representative(df, fam_tuple,
        args.band_round, args.step_round, args.lb_bucket, args.cd_bucket
    )
    if best_row is None:
        print("Could not pick a representative row; aborting.", file=sys.stderr)
        sys.exit(4)

    cfg = build_config_from_row(best_row, include_fees=True, mirror_legacy=True)
    cfg.update({
        "_generated_by": "wf_pick.py",
        "_generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "_source_runs": [os.path.basename(p) for p in args.runs],
        "_family": str(fam_tuple),
    })
    with open(args.emit_config, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"\nWrote recommended config → {args.emit_config}")

    if args.excel_out:
        with pd.ExcelWriter(args.excel_out) as w:
            fam.to_excel(w, sheet_name="family_rank", index=False)
            df.to_excel(w, sheet_name="filtered_rows", index=False)
        print(f"Wrote Excel → {args.excel_out}")

if __name__ == "__main__":
    main()
