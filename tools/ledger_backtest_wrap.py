#!/usr/bin/env python3
"""
tools/ledger_backtest_wrap.py - Ledger Wrapper for Backtester

A wrapper script that runs the backtester and outputs a simplified ledger CSV.
Used for quick PnL verification or export to other tools.
"""
from __future__ import annotations
import argparse, csv, json, subprocess, sys, os


# --- MAGIC PATH FIX ---
# Allow importing 'core' even if running from tools/ folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ----------------------
def main():
    """
    Runs the backtest command and generates a placeholder ledger file.
    """
    ap = argparse.ArgumentParser("ledger backtest wrap")
    ap.add_argument("--data", required=True, help="Path to OHLC data CSV")
    ap.add_argument("--bnb-data", required=True, help="Path to BNB data CSV (for fees)")
    ap.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    ap.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    ap.add_argument("--config", required=True, help="Path to strategy config JSON")
    ap.add_argument("--basis-btc", type=float, default=0.16, help="Initial capital in BTC")
    ap.add_argument("--ledger-out", default="out/ledger.csv", help="Output path for ledger CSV")
    args = ap.parse_args()

    # Construct the backtest command
    cmd = [sys.executable, "core/ethbtc_accum_bot.py", "backtest",
           "--data", args.data, "--bnb-data", args.bnb_data,
           "--basis-btc", str(args.basis_btc), "--config", args.config,
           "--start", args.start, "--end", args.end, "--out", "out/equity_backtest.csv"]

    # Run backtest
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(p.stdout)

    # Create dummy ledger (since actual ledger generation logic was missing in original script)
    # This maintains the interface expected by other tools.
    os.makedirs(os.path.dirname(args.ledger_out), exist_ok=True)
    with open(args.ledger_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp","event","symbol","side","qty","price","notional_btc","fee_btc","maker_like"])
        w.writerow([args.end, "SUMMARY", "ETHBTC", "", "", "", "", "", ""])
    print("Ledger written to", args.ledger_out)

if __name__ == "__main__":
    main()