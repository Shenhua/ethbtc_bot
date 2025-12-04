#!/usr/bin/env python3
import os
import sys
import json
from binance.spot import Spot
from dotenv import load_dotenv

# Path Fix to find 'core'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from core.alert_manager import AlertManager
except ImportError:
    print("Warning: AlertManager not found. Alerts will be skipped.")
    AlertManager = None

load_dotenv()

def main():
    # 1. Environment & Configuration
    symbol = os.getenv("SYMBOL", "ETHBTC")
    mode = os.getenv("MODE", "testnet")
    
    # Auto-detect state file based on mode if not explicitly provided
    default_state_file = f"/data/state_{mode}.json"
    state_path = os.getenv("STATE_FILE", default_state_file)
    
    # Binance API Setup
    api_key = os.getenv("BINANCE_KEY")
    api_secret = os.getenv("BINANCE_SECRET")
    base_url = os.getenv("BINANCE_BASE_URL", "https://api.binance.com")

    if not api_key:
        print("Error: BINANCE_KEY not found in environment.")
        return

    print(f"--- PnL Reconciler ({mode.upper()}) ---")
    print(f"Target: {symbol}")
    print(f"State File: {state_path}")
    print(f"API URL: {base_url}")

    # 2. Load Bot State (The Bot's Truth)
    if not os.path.exists(state_path):
        print(f"❌ Error: No state file found at {state_path}")
        return
        
    with open(state_path) as f:
        state = json.load(f)
        
    # Get Session Start Wealth
    start_w = float(state.get("session_start_W", 0.0))
    current_w_est = float(state.get("risk_equity_high", 0.0)) # Best approximation if current W isn't saved directly
    
    if start_w == 0:
        print("⚠️ Bot has no session history yet (start_W is 0).")
        return

    # 3. Load Real Balance (Binance's Truth)
    # FIX: Use api_key/api_secret and inject base_url
    client = Spot(api_key=api_key, api_secret=api_secret, base_url=base_url)
    
    try:
        # Get current price
        ticker = client.ticker_price(symbol=symbol)
        price = float(ticker["price"])
    except Exception as e:
        print(f"❌ API Error fetching price: {e}")
        return
    
    # Parse Assets (e.g. ETHBTC -> ETH, BTC)
    if "USDC" in symbol:
        base_asset = symbol.replace("USDC", "")
        quote_asset = "USDC"
    elif "USDT" in symbol:
        base_asset = symbol.replace("USDT", "")
        quote_asset = "USDT"
    else:
        base_asset = symbol.replace("BTC", "")
        quote_asset = "BTC"

    def get_bal(asset):
        try:
            acct = client.account()
            for b in acct["balances"]:
                if b["asset"] == asset:
                    return float(b["free"]) + float(b["locked"])
        except Exception as e:
            print(f"❌ API Error fetching balance for {asset}: {e}")
            return 0.0
        return 0.0

    real_base = get_bal(base_asset)
    real_quote = get_bal(quote_asset)
    
    # Calculate Real Wealth in Quote terms
    real_wealth = real_quote + (real_base * price)
    
    # 4. Compare & Alert
    diff = real_wealth - start_w
    diff_pct = (diff / start_w) * 100 if start_w > 0 else 0.0
    abs_diff_pct = abs(diff_pct)

    msg = (
        f"🔍 **AUDIT REPORT ({symbol})**\n"
        f"• Bot Start W: {start_w:.6f}\n"
        f"• Real Wallet: {real_wealth:.6f} {quote_asset} (Base={real_base:.4f}, Quote={real_quote:.4f})\n"
        f"• Actual PnL:  {diff_pct:+.2f}% (|Δ|={abs_diff_pct:.2f}%)"
    )
    print("\n" + msg)

    if AlertManager:
        alerter = AlertManager(prefix=f"AUDIT-{symbol}")

        # Alert when divergence is > 1% (up or down)
        if abs_diff_pct > 1.0:
            level = "CRITICAL" if abs_diff_pct > 5.0 else "WARNING"
            alerter.send(
                f"🚨 WALLET vs BOT DIVERGENCE {abs_diff_pct:.2f}% – check state & fills.\n\n{msg}",
                level=level,
            )
        else:
            # Optional: comment this out if you don’t want spam on healthy checks
            alerter.send(msg, level="INFO")

if __name__ == "__main__":
    main()