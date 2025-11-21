#!/usr/bin/env python3
"""
dust_sweeper.py — Periodic Janitor for Dust Assets.
Syncs with global MODE:
 - MODE=dry     -> Preview only
 - MODE=testnet -> Execute Sweep (Wet)
 - MODE=live    -> Execute Sweep (Wet)
"""

import os, sys, argparse, logging, time
from binance.spot import Spot
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [SWEEPER] %(message)s")
log = logging.getLogger("dust_sweeper")

def get_deployment_mode():
    """Infers the operational mode from the global MODE environment variable."""
    env_mode = os.getenv("MODE", "dry").lower()
    
    if env_mode in ["testnet", "live"]:
        return "wet"  # Real actions allowed
    return "dry"      # Safety default

def run_sweep(client, threshold_usd, dry_run=True):
    mode_label = "DRY" if dry_run else "WET"
    log.info(f"--- Starting Dust Sweep ({mode_label}) ---")
    
    # 1. Fetch Balances
    try:
        acct = client.account()
        balances = [b for b in acct["balances"] if float(b["free"]) > 0]
    except Exception as e:
        log.error(f"Failed to fetch account: {e}")
        return

    dust_assets = []
    
    for b in balances:
        asset = b["asset"]
        free = float(b["free"])
        
        # SAFETY: Never sweep the fuel (BNB) or the accumulation target (BTC)
        if asset in ["BNB", "BTC"]: 
            continue 

        # Get approx USD value
        val_usd = 0.0
        try:
            # Try USDT pair first, then BTC pair
            ticker = f"{asset}USDT"
            price = float(client.ticker_price(ticker)["price"])
            val_usd = free * price
        except:
            # Fallback for assets with no USDT pair
            val_usd = 0.0

        if 0.0 < val_usd < threshold_usd:
            log.info(f"Found Dust: {free:.8f} {asset} (~${val_usd:.2f})")
            dust_assets.append(asset)

    if not dust_assets:
        log.info("Clean wallet. No dust found.")
        return

    # 2. Execute Sweep
    if dry_run:
        log.info(f"[DRY] Would convert: {dust_assets}")
    else:
        try:
            # POST /sapi/v1/asset/dust
            resp = client.user_asset_dust(asset=dust_assets)
            log.info(f"SUCCESS! Sweep response: {resp}")
        except Exception as e:
            log.error(f"Sweep failed: {e}")

def main():
    ap = argparse.ArgumentParser()
    # "auto" means: look at os.getenv("MODE")
    ap.add_argument("--mode", choices=["auto", "dry", "wet"], default="auto")
    ap.add_argument("--threshold-usd", type=float, default=5.0)
    ap.add_argument("--schedule", choices=["once", "weekly", "daily"], default="once")
    args = ap.parse_args()

    # 1. Determine Logic
    if args.mode == "auto":
        action_mode = get_deployment_mode()
    else:
        action_mode = args.mode
    
    is_dry = (action_mode == "dry")

    # 2. Connect
    key = os.getenv("BINANCE_KEY")
    secret = os.getenv("BINANCE_SECRET")
    base_url = os.getenv("BINANCE_BASE_URL", "https://api.binance.com")
    
    # Safety Check
    if not key:
        log.error("No API Key found. Exiting.")
        return

    client = Spot(api_key=key, api_secret=secret, base_url=base_url)
    log.info(f"Initialized. Environment MODE={os.getenv('MODE', 'unset')}. Action Mode={action_mode.upper()}. Schedule={args.schedule}")

    # 3. Execution Loop
    while True:
        run_sweep(client, args.threshold_usd, dry_run=is_dry)
        
        if args.schedule == "once":
            break
        
        if args.schedule == "daily":
            seconds = 24 * 60 * 60
        else: # weekly
            seconds = 7 * 24 * 60 * 60
            
        log.info(f"Sleeping for {args.schedule} ({seconds}s)...")
        time.sleep(seconds)

if __name__ == "__main__":
    main()