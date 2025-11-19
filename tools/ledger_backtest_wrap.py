#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, subprocess, sys, os

def main():
    ap = argparse.ArgumentParser("ledger backtest wrap")
    ap.add_argument("--data", required=True)
    ap.add_argument("--bnb-data", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--basis-btc", type=float, default=0.16)
    ap.add_argument("--ledger-out", default="out/ledger.csv")
    args = ap.parse_args()

    cmd = [sys.executable, "ethbtc_accum_bot.py", "backtest",
           "--data", args.data, "--bnb-data", args.bnb_data,
           "--basis-btc", str(args.basis_btc), "--config", args.config,
           "--start", args.start, "--end", args.end, "--out", "out/equity_backtest.csv"]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    os.makedirs(os.path.dirname(args.ledger_out), exist_ok=True)
    with open(args.ledger_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp","event","symbol","side","qty","price","notional_btc","fee_btc","maker_like"])
        w.writerow([args.end, "SUMMARY", "ETHBTC", "", "", "", "", "", ""])
    print("Ledger written to", args.ledger_out)

if __name__ == "__main__":
    main()
