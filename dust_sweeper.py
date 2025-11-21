#!/usr/bin/env python3
"""
dust_sweeper.py — Periodic Janitor for Dust Assets.
Syncs with global MODE:
 - MODE=dry     -> Preview only
 - MODE=testnet -> Execute Sweep (Wet) -> Fails gracefully (Not supported on Testnet)
 - MODE=live    -> Execute Sweep (Wet)

Configuration:
 - Reads SWEEPER_IGNORE from .env (e.g. SWEEPER_IGNORE=SHIB,DOGE)
"""

import os, sys, argparse, logging, time
from binance.spot import Spot
from binance.error import ClientError
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [SWEEPER] %(message)s")
log = logging.getLogger("dust_sweeper")

def get_deployment_mode():
    """Infers the operational mode from the global MODE environment variable."""
    env_mode = os.getenv("MODE", "dry").lower()
    
    if env_mode in ["testnet", "live"]:
        return "wet"
    return "dry"

def run_sweep(client, threshold_usd, ignore_list, dry_run=True):
    mode_label = "DRY" if dry_run else "WET"
    log.info(f"--- Starting Dust Sweep ({mode_label}) ---")
    log.info(f"Ignoring protected assets: {ignore_list}")
    
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
        
        # SAFETY: Always ignore BNB, BTC, and user list
        if asset in ignore_list: 
            continue 

        # Get approx USD value
        val_usd = 0.0
        try:
            ticker = f"{asset}USDT"
            price = float(client.ticker_price(ticker)["price"])
            val_usd = free * price
        except:
            # Fallback logic for non-USDT pairs
            try:
                if asset != "BTC":
                    ticker = f"{asset}BTC"
                    # Just an existence check mostly
                    client.ticker_price(ticker)
                    # Treat as 0 USD (candidate for dust) if we can't price it easily
                    val_usd = 0.0 
            except:
                val_usd = 0.0

        if 0.0 < val_usd < threshold_usd:
            log.info(f"Found Dust: {free:.8f} {asset} (~${val_usd:.2f})")
            dust_assets.append(asset)

    if not dust_assets:
        log.info("Clean wallet. No dust found.")
        return

    if dry_run:
        log.info(f"[DRY] Would convert: {dust_assets}")
    else:
        try:
            # Correct method name for v3 connector
            resp = client.dust_transfer(asset=dust_assets)
            log.info(f"SUCCESS! Sweep response: {resp}")
        except ClientError as e:
            if e.error_code == -1002:
                log.error(f"Sweep failed (API Error): {e.error_message}. (Note: Not supported on Testnet).")
            else:
                log.error(f"Sweep failed (Binance Error): {e}")
        except AttributeError:
             log.error("Sweep failed: Method 'dust_transfer' not found. Update binance-connector.")
        except Exception as e:
            log.error(f"Sweep failed (Unknown): {e}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["auto", "dry", "wet"], default="auto")
    ap.add_argument("--threshold-usd", type=float, default=5.0)
    ap.add_argument("--schedule", choices=["once", "weekly", "daily"], default="once")
    ap.add_argument("--ignore", default="", help="Override .env ignore list")
    args = ap.parse_args()

    # 1. Determine Mode
    if args.mode == "auto":
        action_mode = get_deployment_mode()
    else:
        action_mode = args.mode
    is_dry = (action_mode == "dry")

    # 2. Build Ignore List from ENV + ARGS
    base_ignores = ["BNB", "BTC"]
    
    # Read from .env
    env_ignore_str = os.getenv("SWEEPER_IGNORE", "")
    env_ignores = [x.strip() for x in env_ignore_str.split(",") if x.strip()]
    
    # Read from CLI (overrides or adds to env? let's merge)
    cli_ignores = [x.strip() for x in args.ignore.split(",") if x.strip()]
    
    full_ignore_list = list(set(base_ignores + env_ignores + cli_ignores))

    # 3. Connect
    key = os.getenv("BINANCE_KEY")
    secret = os.getenv("BINANCE_SECRET")
    base_url = os.getenv("BINANCE_BASE_URL", "https://api.binance.com")
    
    if not key:
        log.error("No API Key found. Exiting.")
        return

    client = Spot(api_key=key, api_secret=secret, base_url=base_url)
    log.info(f"Initialized. MODE={os.getenv('MODE')}. Action={action_mode.upper()}. Schedule={args.schedule}")

    # 4. Loop
    while True:
        run_sweep(client, args.threshold_usd, full_ignore_list, dry_run=is_dry)
        
        if args.schedule == "once":
            break
        
        seconds = 24 * 60 * 60 if args.schedule == "daily" else 7 * 24 * 60 * 60
        log.info(f"Sleeping for {args.schedule} ({seconds}s)...")
        time.sleep(seconds)

if __name__ == "__main__":
    main()