#!/usr/bin/env python3
"""
Seed a small Spot TESTNET position using binance-connector (version-agnostic init).

Usage:
  pip install binance-connector
  export BINANCE_API_KEY=YOUR_TESTNET_KEY
  export BINANCE_API_SECRET=YOUR_TESTNET_SECRET

  # Buy 0.02 ETH on ETHBTC (spends BTC) on Spot TESTNET
  python seed_testnet_order.py --symbol ETHBTC --side BUY --qty 0.02

Notes:
- For ETHBTC, quantity is in ETH units (base asset).
- Checks LOT_SIZE.stepSize and MIN_NOTIONAL/NOTIONAL, rounds qty down, ensures notional >= min.
- Default base URL is Spot TESTNET: https://testnet.binance.vision
"""
import os
import sys
import argparse
import inspect
from decimal import Decimal
from typing import Tuple

try:
    from binance.spot import Spot
except Exception as e:
    print("Error: binance-connector not installed. Try: pip install binance-connector", file=sys.stderr)
    raise

DEF_BASE_URL = "https://testnet.binance.vision"

def make_client(api_key: str, api_secret: str, base_url: str) -> Spot:
    """Construct Spot client across connector versions (key/secret vs api_key/api_secret)."""
    params = inspect.signature(Spot.__init__).parameters
    if "key" in params and "secret" in params:
        return Spot(key=api_key, secret=api_secret, base_url=base_url)
    elif "api_key" in params and "api_secret" in params:
        return Spot(api_key=api_key, api_secret=api_secret, base_url=base_url)
    else:
        # Last resort: position-only args (api_key, api_secret)
        try:
            return Spot(api_key, api_secret, base_url=base_url)  # type: ignore[arg-type]
        except TypeError as te:
            raise SystemExit(f"Unsupported binance-connector Spot signature: {params}") from te

def round_to_step(qty: float, step_size: float) -> float:
    """Floor quantity to the nearest multiple of step_size."""
    if not step_size:
        return float(qty)
    q = Decimal(str(qty))
    s = Decimal(str(step_size))
    # floor to multiple
    return float((q // s) * s)

def fetch_filters(c: Spot, symbol: str) -> Tuple[float, float]:
    """Return (step_size, min_notional) using exchangeInfo."""
    ex = c.exchange_info(symbol=symbol)
    sym = ex["symbols"][0]
    step_size = 0.0
    min_notional = 0.0
    for f in sym.get("filters", []):
        t = f.get("filterType")
        if t == "LOT_SIZE":
            step_size = float(f.get("stepSize", 0.0))
        elif t in ("MIN_NOTIONAL", "NOTIONAL"):
            if "minNotional" in f:
                min_notional = float(f["minNotional"])
            elif "notional" in f:
                min_notional = float(f["notional"])
    return step_size, min_notional

def main():
    ap = argparse.ArgumentParser(description="Place a MARKET order on Binance Spot TESTNET (or change base-url for mainnet)." )
    ap.add_argument("--symbol", default="ETHBTC", help="Trading pair symbol, e.g., ETHBTC")
    ap.add_argument("--side", choices=["BUY","SELL"], default="BUY", help="Order side")
    ap.add_argument("--qty", type=float, required=True, help="Quantity in BASE asset units (ETH for ETHBTC)")
    ap.add_argument("--base-url", default=DEF_BASE_URL, help="Binance API base URL (default: Spot TESTNET)")
    args = ap.parse_args()

    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    if not api_key or not api_secret:
        print("Set BINANCE_API_KEY and BINANCE_API_SECRET in your environment.", file=sys.stderr)
        sys.exit(2)

    c = make_client(api_key, api_secret, base_url=args.base_url)

    # Filters and step rounding
    step_size, min_notional = fetch_filters(c, args.symbol)
    qty = round_to_step(args.qty, step_size)

    # Mid price from book ticker
    bt = c.book_ticker(symbol=args.symbol)
    bid = float(bt.get("bidPrice", 0.0)); ask = float(bt.get("askPrice", 0.0))
    if bid <= 0.0 or ask <= 0.0:
        px = float(c.ticker_price(symbol=args.symbol)["price"])  # fallback
        bid = ask = px
    mid = (bid + ask) / 2.0

    # Notional check (quote-asset value at mid)
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
            f"Try qty >= ~{need:.6f} (base units)." ,
            file=sys.stderr
        )
        sys.exit(1)

    print(f"Placing MARKET {args.side} {args.symbol} qty={qty:.8f} (mid≈{mid:.8f}) on {args.base_url}")
    resp = c.new_order(symbol=args.symbol, side=args.side, type="MARKET", quantity=qty)
    print("OK:", resp)

    acct = c.account()
    free = {b["asset"]: float(b["free"]) for b in acct["balances"] if float(b["free"]) > 0}
    print("Free balances:", free)

if __name__ == "__main__":
    main()
