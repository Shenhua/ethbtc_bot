import pandas as pd
import sys

def analyze(csv_path):
    df = pd.read_csv(csv_path)
    df['close_time'] = pd.to_datetime(df['close_time'])
    df = df.set_index('close_time').sort_index()
    
    # Assuming standard backtest output where we have 'wealth_btc' (Strategy Equity)
    # We need Price data. The CSV usually doesn't have Price column unless we added it?
    # Wait, audit_result csv usually has diagnostics.
    # Let's check columns.
    # The head output showed: close_time,wealth_btc,target_w,regime_score,regime_state,sig_mr,sig_trend
    # It does NOT have price.
    # But we can infer price return if we assume fully invested periods match price return? No.
    # We need the price data.
    # I will load the price data from the source CSV used in backtest.
    
    return df

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python analyze_exposure.py <audit_csv> <price_csv>")
        sys.exit(1)
        
    audit_path = sys.argv[1]
    price_path = sys.argv[2]
    
    print(f"Loading audit: {audit_path}")
    res = pd.read_csv(audit_path)
    # Fix: Use format='mixed' to handle varying timestamp formats
    res['close_time'] = pd.to_datetime(res['close_time'], utc=True, format='mixed')
    res = res.set_index('close_time').sort_index()
    
    print(f"Loading price: {price_path}")
    px = pd.read_csv(price_path)
    # Vision CSV format
    px.columns = [c.strip().lower() for c in px.columns]
    if 'date' in px.columns: px = px.rename(columns={'date':'close_time'})
    if 'closetime' in px.columns: px = px.rename(columns={'closetime':'close_time'})
    
    # Parse dates
    # Try numeric first
    try:
        px['close_time'] = pd.to_datetime(pd.to_numeric(px['close_time']), unit='ms', utc=True)
    except:
        px['close_time'] = pd.to_datetime(px['close_time'], utc=True)
        
    px = px.set_index('close_time').sort_index()
    
    # Align
    common_idx = res.index.intersection(px.index)
    res = res.loc[common_idx]
    px = px.loc[common_idx]
    
    # Calculate Stats
    start_price = px['close'].iloc[0]
    end_price = px['close'].iloc[-1]
    bh_return = (end_price / start_price) - 1.0
    
    start_wealth = res['wealth_btc'].iloc[0]
    end_wealth = res['wealth_btc'].iloc[-1]
    bot_return = (end_wealth / start_wealth) - 1.0
    
    avg_exposure = res['target_w'].mean()
    time_in_market = (res['target_w'] > 0.01).mean()
    
    print("-" * 40)
    print(f"Period: {res.index[0]} to {res.index[-1]}")
    print(f"Start Price: {start_price:.2f}")
    print(f"End Price:   {end_price:.2f}")
    print(f"Buy & Hold Return: {bh_return*100:.2f}% ({end_price/start_price:.2f}x)")
    print(f"Bot Return:        {bot_return*100:.2f}% ({end_wealth/start_wealth:.2f}x)")
    print("-" * 40)
    print(f"Average Exposure:  {avg_exposure*100:.2f}%")
    print(f"Time in Market:    {time_in_market*100:.2f}% (> 1% exposure)")
    print("-" * 40)
    
    # Check correlation
    # Daily returns
    res_daily = res['wealth_btc'].resample('1D').last().pct_change()
    px_daily = px['close'].resample('1D').last().pct_change()
    corr = res_daily.corr(px_daily)
    print(f"Daily Correlation: {corr:.4f}")
