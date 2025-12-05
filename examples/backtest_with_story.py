#!/usr/bin/env python3
"""
Demo: Backtest with Real-Time Story Generation

Shows how to run a backtest that generates a story file in real-time,
just like live trading does.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ethbtc_accum_bot import load_vision_csv, FeeParams, Backtester
from core.meta_strategy import MetaStrategy, MRParams, TrendParams
from core.story_writer import StoryWriter
from datetime import datetime

def main():
    # 1. Load data
    df = load_vision_csv("data/raw/BTCUSDT_15m_2021-2025_vision.csv")
    print(f"Loaded {len(df)} bars")
    
    # 2. Create strategy (example: Meta Strategy)
    mr_params = MRParams(
        trend_kind="sma",
        trend_lookback=240,
        flip_band_entry=0.015,
        flip_band_exit=0.010,
        cooldown_minutes=180,
        step_allocation=0.5,
        long_only=True
    )
    
    trend_params = TrendParams(
        fast_period=50,
        slow_period=200,
        ma_type="ema",
        cooldown_minutes=240,
        long_only=True
    )
    
    adx_threshold = 20.0
    strategy = MetaStrategy(mr_params, trend_params, adx_threshold=adx_threshold)
    
    # 3. Create StoryWriter with DYNAMIC filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    strategy_name = type(strategy).__name__
    story_file = f"results/backtest_{strategy_name}_adx{int(adx_threshold)}_{timestamp}.txt"
    story_writer = StoryWriter(story_file, symbol="BTCUSDT")
    
    print(f"Story will be written to: {story_file}")
    
    # 4. Run backtest with story logging!
    fee = FeeParams(maker_fee=0.0002, taker_fee=0.0004)
    bt = Backtester(fee)
    
    print(f"Running backtest with story logging...")
    result = bt.simulate(
        df["close"],
        strategy,
        full_df=df,
        story_writer=story_writer  # ← The magic happens here!
    )
    
    # 5. Print summary
    print(f"\n✅ Backtest complete!")
    print(f"Final BTC: {result['summary']['final_btc']:.4f}")
    print(f"Trades: {result['summary']['n_trades']}")
    print(f"\n📖 Story file generated: {story_file}")
    print(f"   View it with: cat {story_file}")

if __name__ == "__main__":
    main()
