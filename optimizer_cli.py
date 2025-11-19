
from __future__ import annotations
import os, json, math, argparse, time, random
import concurrent.futures as cf
import numpy as np
import pandas as pd
from ethbtc_accum_bot import (
    load_vision_csv, load_json_config, _write_excel,
    FeeParams, StratParams, EthBtcStrategy, Backtester
)

def sample_params():
    return StratParams(
        trend_kind=random.choice(["sma","roc"]),
        trend_lookback=int(random.choice([120,160,200,240,300])),
        flip_band_entry=float(random.uniform(0.01,0.05)),
        flip_band_exit=float(random.uniform(0.005,0.03)),
        vol_window=int(random.choice([45,60,90])),
        vol_adapt_k=float(random.choice([0.0, 0.25, 0.5, 0.75]))/100.0,
        target_vol=float(random.choice([0.3,0.4,0.5,0.6])),
        min_mult=0.5, max_mult=1.5,
        cooldown_minutes=int(random.choice([60,120,180,240])),
        step_allocation=float(random.choice([0.33,0.5,0.66,1.0])),
        max_position=float(random.choice([0.6,0.8,1.0])),
        gate_window_days=int(random.choice([30,60,90])),
        gate_roc_threshold=float(random.choice([0.0, 0.01, 0.02])),
    )

def score_rows(df, lam_turns=2.0, gap_penalty=0.35, turns_scale=800.0, lam_fees=2.0, lam_turnover=1.0):
    df = df.copy()
    if {"train_final_btc","test_final_btc"}.issubset(df.columns):
        df["gen_gap"] = df["train_final_btc"] - df["test_final_btc"]
    else:
        df["gen_gap"] = 0.0
    df["turns_test"] = df.get("turns_test", 0.0).astype(float)
    fees = df.get("fees_btc", 0.0).astype(float)
    tnov = df.get("turnover_btc", 0.0).astype(float)
    df["robust_score"] = (
        df["test_final_btc"].astype(float)
        - lam_turns * (df["turns_test"] / float(turns_scale))
        - gap_penalty * np.maximum(0.0, df["gen_gap"].astype(float))
        - lam_fees * fees
        - lam_turnover * tnov
    )
    return df

def select_from_top(df, top_quantile=0.95):
    if df.empty: raise ValueError("No rows to select from.")
    thr = df["test_final_btc"].quantile(top_quantile)
    pool = df[df["test_final_btc"] >= thr].copy()
    if pool.empty: pool = df.copy()
    pool = pool.sort_values(["robust_score","turns_test","test_final_btc"], ascending=[False,True,False])
    return pool.iloc[0]

def _eval_one(arg):
    (params_dict, fee, train_close, test_close, bnb_train, bnb_test) = arg
    p = StratParams(**params_dict)
    bt = Backtester(fee)
    res_tr = bt.simulate(train_close, EthBtcStrategy(p), bnb_price_series=bnb_train)
    res_te = bt.simulate(test_close,  EthBtcStrategy(p), bnb_price_series=bnb_test)
    return {
        "params": p.__dict__,
        "train_final_btc": res_tr["summary"]["final_btc"],
        "test_final_btc":  res_te["summary"]["final_btc"],
        "turns_test":      res_te["summary"]["turns"],
        "fees_btc":        res_te["summary"]["fees_btc"],
        "turnover_btc":    res_te["summary"]["turnover_btc"],
        "score":           res_te["summary"]["final_btc"],
    }

def main():
    ap = argparse.ArgumentParser(description="Fast Optimizer (parallel + early stop)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--bnb-data")
    ap.add_argument("--config")
    ap.add_argument("--train-start", required=True)
    ap.add_argument("--train-end", required=True)
    ap.add_argument("--test-start", required=True)
    ap.add_argument("--test-end", required=True)
    ap.add_argument("--n-random", type=int, default=400)
    ap.add_argument("--maker-fee", type=float, default=0.0002)
    ap.add_argument("--taker-fee", type=float, default=0.0004)
    ap.add_argument("--bnb-discount", type=float, default=0.25)
    ap.add_argument("--no-bnb", action="store_true")
    ap.add_argument("--slippage-bps", type=float, default=1.0)
    ap.add_argument("--out", default="opt_results_fast.csv")
    ap.add_argument("--excel-out")
    ap.add_argument("--lambda-turns", type=float, default=2.0)
    ap.add_argument("--gap-penalty", type=float, default=0.35)
    ap.add_argument("--turns-scale", type=float, default=800.0)
    ap.add_argument("--lambda-fees", type=float, default=2.0)
    ap.add_argument("--lambda-turnover", type=float, default=1.0)
    ap.add_argument("--top-quantile", type=float, default=0.95)
    ap.add_argument("--emit-config")
    # Fast controls
    ap.add_argument("--threads", type=int, default=0, help="0=auto (cpu count)")
    ap.add_argument("--chunk-size", type=int, default=32, help="tasks batched per worker fetch")
    ap.add_argument("--early-stop", type=int, default=120, help="stop after this many evals if not improving")
    ap.add_argument("--patience", type=int, default=3, help="checks without improvement before stop")
    ap.add_argument("--min-improve", type=float, default=0.005, help="relative improvement needed (0.5% default)")
    ap.add_argument("--no-excel", action="store_true")
    args = ap.parse_args()

    cfg = load_json_config(args.config)
    fee = FeeParams(
        maker_fee=float(cfg.get("maker_fee", args.maker_fee)),
        taker_fee=float(cfg.get("taker_fee", args.taker_fee)),
        bnb_discount=float(cfg.get("bnb_discount", args.bnb_discount)),
        slippage_bps=float(cfg.get("slippage_bps", args.slippage_bps)),
        pay_fees_in_bnb=bool(cfg.get("pay_fees_in_bnb", not args.no_bnb)),
    )

    df = load_vision_csv(args.data); close = df["close"]
    train_close = close.loc[args.train_start:args.train_end].dropna()
    test_close  = close.loc[args.test_start:args.test_end].dropna()

    bnb_train = bnb_test = None
    bnb_path = cfg.get("bnb_data", args.bnb_data)
    if bnb_path:
        df_bnb = load_vision_csv(bnb_path)["close"]
        bnb_train = df_bnb.reindex(train_close.index, method="ffill")
        bnb_test  = df_bnb.reindex(test_close.index,  method="ffill")

    tasks = []
    for _ in range(args.n_random):
        p = sample_params()
        tasks.append((p.__dict__, fee, train_close, test_close, bnb_train, bnb_test))

    threads = (os.cpu_count() or 2) if args.threads == 0 else args.threads
    results = []
    best_score = -1e9
    stale_checks = 0
    t0 = time.time()

    with cf.ProcessPoolExecutor(max_workers=threads) as ex:
        for i, res in enumerate(ex.map(_eval_one, tasks, chunksize=args.chunk_size), 1):
            results.append(res)
            if res["score"] > best_score * (1.0 + args.min_improve):
                best_score = res["score"]
                stale_checks = 0
            if i % args.early_stop == 0:
                stale_checks += 1
                if stale_checks >= args.patience:
                    break

    df_out = pd.DataFrame(results).sort_values("score", ascending=False)
    scored = score_rows(df_out, args.lambda_turns, args.gap_penalty, args.turns_scale,
                        args.lambda_fees, args.lambda_turnover)
    best = select_from_top(scored, args.top_quantile)

    scored.to_csv(args.out, index=False)
    print(f"Saved → {args.out}  | evaluated={len(df_out)}  | threads={threads}  | elapsed={time.time()-t0:.1f}s")

    if args.excel_out and not args.no_excel:
        _write_excel(args.excel_out, {"Optimization": scored})
        print(f"Wrote → {args.excel_out}")

    if args.emit_config:
        with open(args.emit_config, "w") as f: json.dump(best["params"], f, indent=2)
        print(f"Wrote selected params → {args.emit_config}")

if __name__ == "__main__":
    main()
