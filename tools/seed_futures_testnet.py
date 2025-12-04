#!/usr/bin/env python3
"""
Seed a small Futures TESTNET position using binance-connector.

Usage:
  pip install binance-connector
  export FUTURES_TESTNET_KEY=YOUR_TESTNET_KEY
  export FUTURES_TESTNET_SECRET=YOUR_TESTNET_SECRET

  # Open a LONG position (buy 0.01 BTC worth of contracts)
  python seed_futures_testnet.py --symbol BTCUSDT --side BUY --qty 0.01

  # Open a SHORT position (sell 0.01 BTC worth of contracts)
  python seed_futures_testnet.py --symbol BTCUSDT --side SELL --qty 0.01

Notes:
- For BTCUSDT, quantity is in BTC units (base asset).
- Checks LOT_SIZE.stepSize and MIN_NOTIONAL, rounds qty down.
- Default base URL is Futures TESTNET: https://testnet.binancefuture.com
"""
import os
import sys
import argparse
from decimal import Decimal
from typing import Tuple

try:
    from binance.um_futures import UMFutures
except Exception as e:
    print("Error: binance-connector not installed. Try: pip install binance-connector", file=sys.stderr)
    raise

# --- MAGIC PATH FIX ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ----------------------

DEF_BASE_URL = "https://testnet.binancefuture.com"

def round_to_step(qty: float, step_size: float) -> float:
    """Floor quantity to the nearest multiple of step_size."""
    if not step_size:
        return float(qty)
    q = Decimal(str(qty))
    s = Decimal(str(step_size))
    return float((q // s) * s)

def fetch_filters(c: UMFutures, symbol: str) -> Tuple[float, float]:
    """Return (step_size, min_notional) using exchangeInfo."""
    ex = c.exchange_info()
    sym = None
    for s in ex["symbols"]:
        if s["symbol"] == symbol:
            sym = s
            break
    
    if not sym:
        raise ValueError(f"Symbol {symbol} not found in exchange info")
    
    step_size = 0.0
    min_notional = 0.0
    for f in sym.get("filters", []):
        t = f.get("filterType")
        if t == "LOT_SIZE":
            step_size = float(f.get("stepSize", 0.0))
        elif t in ("MIN_NOTIONAL", "NOTIONAL"):
            if "notional" in f:
                min_notional = float(f["notional"])
    return step_size, min_notional

def main():
    ap = argparse.ArgumentParser(description="Place a MARKET order on Binance Futures TESTNET.")
    ap.add_argument("--symbol", default="BTCUSDT", help="Trading pair symbol, e.g., BTCUSDT")
    ap.add_argument("--side", choices=["BUY","SELL"], default="BUY", help="Order side (BUY=long, SELL=short)")
    ap.add_argument("--qty", type=float, required=True, help="Quantity in BASE asset units (BTC for BTCUSDT)")
    ap.add_argument("--base-url", default=DEF_BASE_URL, help="Binance API base URL (default: Futures TESTNET)")
    args = ap.parse_args()

    api_key = os.getenv("FUTURES_TESTNET_KEY") or os.getenv("BINANCE_FUTURES_KEY")
    api_secret = os.getenv("FUTURES_TESTNET_SECRET") or os.getenv("BINANCE_FUTURES_SECRET")
    
    if not api_key or not api_secret:
        print("Set FUTURES_TESTNET_KEY and FUTURES_TESTNET_SECRET in your environment.", file=sys.stderr)
        sys.exit(2)

    c = UMFutures(key=api_key, secret=api_secret, base_url=args.base_url)

    # Set leverage to 1x (safest for testing)
    try:
        c.change_leverage(symbol=args.symbol, leverage=1)
        print(f"Set leverage to 1x for {args.symbol}")
    except Exception as e:
        print(f"Warning: Could not set leverage: {e}")

    # Filters and step rounding
    step_size, min_notional = fetch_filters(c, args.symbol)
    qty = round_to_step(args.qty, step_size)

    # Mid price from book ticker
    bt = c.book_ticker(symbol=args.symbol)
    bid = float(bt.get("bidPrice", 0.0))
    ask = float(bt.get("askPrice", 0.0))
    if bid <= 0.0 or ask <= 0.0:
        px = float(c.ticker_price(symbol=args.symbol)["price"])
        bid = ask = px
    mid = (bid + ask) / 2.0

    # Notional check
    notional = qty * mid
    if notional < min_notional:
        need = min_notional / max(mid, 1e-12)
        print(
            f"Quantity too small for min notional.\n"
            f"  step_size   = {step_size}\n"
            f"  mid         = {mid:.12f}\n"
            f"  qty_rounded = {qty:.8f}\n"
            f"  notional    = {notional:.12f}\n"
            f"  minNotional = {min_notional:.12f}\n"
            f"Try qty >= ~{need:.6f} (base units).",
            file=sys.stderr
        )
        sys.exit(1)

    print(f"Placing MARKET {args.side} {args.symbol} qty={qty:.8f} (mid≈{mid:.8f}) on {args.base_url}")
    resp = c.new_order(symbol=args.symbol, side=args.side, type="MARKET", quantity=qty)
    print("OK:", resp)

    # Show account balance
    acct = c.balance()
    print("\nAccount balances:")
    for bal in acct:
        if float(bal.get("balance", 0)) > 0:
            print(f"  {bal['asset']}: {bal['balance']}")
    
    # Show position
    positions = c.get_position_risk(symbol=args.symbol)
    for pos in positions:
        if float(pos.get("positionAmt", 0)) != 0:
            print(f"\nPosition {pos['symbol']}: {pos['positionAmt']} @ {pos['entryPrice']}")
            print(f"  Unrealized PnL: {pos['unRealizedProfit']}")

if __name__ == "__main__":
    main()
