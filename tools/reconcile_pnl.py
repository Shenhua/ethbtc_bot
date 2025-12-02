import os
import sys
import json
import logging
from binance.spot import Spot
from dotenv import load_dotenv

# Path Fix
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.alert_manager import AlertManager

load_dotenv()

def main():
    symbol = os.getenv("SYMBOL", "ETHBTC")
    state_path = os.getenv("STATE_FILE", "run_state/eth/state.json")
    
    # 1. Load State (Bot's Truth)
    if not os.path.exists(state_path):
        print("No state file found.")
        return
        
    with open(state_path) as f:
        state = json.load(f)
        
    bot_wealth = state.get("risk_equity_high", 0.0) # Or current W if stored
    # Note: You might need to save 'current_wealth' explicitly in state.json in live_executor
    
    # 2. Load Real Balance (Binance's Truth)
    client = Spot(key=os.getenv("BINANCE_KEY"), secret=os.getenv("BINANCE_SECRET"))
    
    # Get Prices
    price = float(client.ticker_price(symbol=symbol)["price"])
    
    # Get Balances
    base = symbol.replace("BTC", "").replace("USDC", "") # Naive parsing
    quote = "BTC" if "BTC" in symbol else "USDC"
    
    def get_bal(asset):
        acct = client.account()
        for b in acct["balances"]:
            if b["asset"] == asset:
                return float(b["free"]) + float(b["locked"])
        return 0.0

    real_base = get_bal(base)
    real_quote = get_bal(quote)
    
    real_wealth_btc = real_quote + (real_base * price)
    
    # 3. Compare
    # Assuming 'session_start_W' tracks the initial deposit
    start_w = state.get("session_start_W", 0.0)
    
    # Calculate Drift
    # (This logic depends on how you track 'current wealth' in state. 
    #  If you rely on Prometheus for current wealth, this script needs to query Prometheus or just trust the Wallet.)
    
    print(f"Real Wallet Value: {real_wealth_btc:.6f} {quote}")
    print(f"Bot Session Start: {start_w:.6f} {quote}")
    
    # Simple Alert: If Wallet drops 50% below Start, something is catastrophic
    if real_wealth_btc < (start_w * 0.5):
        alerter = AlertManager(prefix="AUDITOR")
        alerter.send(f"🚨 CRITICAL: Wallet Balance ({real_wealth_btc:.4f}) is 50% below Session Start!", level="CRITICAL")

if __name__ == "__main__":
    main()