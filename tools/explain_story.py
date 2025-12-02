#!/usr/bin/env python3
import pandas as pd
import argparse
import sys

def analyze_story(csv_path, adx_threshold=10.0, output_file=None):
    output_lines = []

    # Helper to print to screen AND buffer for file
    def log(msg):
        print(msg)
        output_lines.append(msg)

    log(f"📖 Reading Story from: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
        
        # Handle various date formats
        time_col = None
        if "close_time" in df.columns: time_col = "close_time"
        elif "time" in df.columns: time_col = "time"
        
        if time_col:
            df["time"] = pd.to_datetime(df[time_col], format='mixed', utc=True)
        else:
            df["time"] = pd.to_datetime(df.iloc[:, 0], format='mixed', utc=True)
            
        df = df.set_index("time").sort_index()

    except Exception as e:
        log(f"Error loading CSV: {e}")
        return

    # Columns check
    required = ["wealth_btc", "regime_score", "target_w"]
    if not all(col in df.columns for col in required):
        log(f"❌ Missing columns! Need: {required}")
        return

    log(f"🔍 Analyzing {len(df)} bars with Threshold {adx_threshold}...\n")
    log("="*70)
    log(f"{'TIMESTAMP':<20} | {'EVENT':<45} | {'DETAILS'}")
    log("="*70)

    # State Tracking
    current_regime = "UNKNOWN"
    last_target = df["target_w"].iloc[0]
    peak_wealth = df["wealth_btc"].iloc[0]
    last_wealth = df["wealth_btc"].iloc[0]
    
    is_halted = False
    
    # Monthly PnL
    current_month = df.index[0].month
    month_start_wealth = df["wealth_btc"].iloc[0]

    for t, row in df.iterrows():
        wealth = row["wealth_btc"]
        score = row["regime_score"]
        target = row["target_w"]
        price = row["close"] if "close" in df.columns else 0.0

        # 1. NEW ATH DETECTION
        if wealth > peak_wealth:
            peak_wealth = wealth
            if (wealth / last_wealth) > 1.02: 
                log(f"{str(t):<20} | 🚀 NEW ATH REACHED                          | {wealth:.4f} BTC")

        dd_pct = (peak_wealth - wealth) / peak_wealth if peak_wealth > 0 else 0.0

        # 2. REGIME SWITCH
        new_regime = "TREND" if score > adx_threshold else "MEAN_REV"
        if new_regime != current_regime:
            if current_regime != "UNKNOWN":
                log(f"{str(t):<20} | 🔄 REGIME SWITCH: {new_regime:<15}         | Score: {score:.1f}")
            current_regime = new_regime

        # 3. TRADE DETECTION
        delta = target - last_target
        if abs(delta) > 0.01:
            if delta > 0:
                action = "🟢 BUY"
                px_str = f" @ {price:.2f}" if price > 0 else ""
                log(f"{str(t):<20} | {action:<45} | Incr Exp {last_target:.2f}->{target:.2f}{px_str}")
            else:
                action = "🔴 SELL"
                px_str = f" @ {price:.2f}" if price > 0 else ""
                log(f"{str(t):<20} | {action:<45} | Decr Exp {last_target:.2f}->{target:.2f}{px_str}")

        # 4. SAFETY HALT DETECTION
        if not is_halted and target == 0.0 and dd_pct > 0.19: # Assuming 20% is limit
             is_halted = True
             log(f"{str(t):<20} | 🚨 SAFETY BREAKER TRIPPED                    | DD: -{dd_pct:.1%}")

        # 5. PHOENIX REBIRTH
        if is_halted and target > 0.0:
            is_halted = False
            log(f"{str(t):<20} | 🔥 PHOENIX PROTOCOL ACTIVATED                | Resuming Trades")

        # 6. MONTHLY REPORT
        if t.month != current_month:
            m_ret = (wealth / month_start_wealth) - 1.0
            icon = "📈" if m_ret > 0 else "🔻"
            log(f"{str(t.date()):<20} | {icon} MONTHLY PnL: {m_ret:+.2%}                    | Bal: {wealth:.4f}")
            
            current_month = t.month
            month_start_wealth = wealth
        
        last_target = target
        last_wealth = wealth

    log("="*70)
    log(f"FINAL RESULT: {df['wealth_btc'].iloc[-1]:.4f} BTC")
    log("="*70)

    # Write to file if requested
    if output_file:
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("\n".join(output_lines))
            print(f"\n✅ Report saved to: {output_file}")
        except Exception as e:
            print(f"\n❌ Failed to save report: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="Path to backtest CSV")
    parser.add_argument("--threshold", type=float, default=10.0, help="ADX Threshold")
    parser.add_argument("--out", help="Optional output file path for the report")
    args = parser.parse_args()
    
    analyze_story(args.csv, args.threshold, args.out)