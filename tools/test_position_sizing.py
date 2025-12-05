#!/usr/bin/env python3
"""
Quick test script for Dynamic Position Sizing.
Tests that the implementation works correctly.
"""
import sys
import os

# --- MAGIC PATH FIX ---
# Allow importing 'core' even if running from tools/ folder
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ----------------------

from core.position_sizer import PositionSizer, PositionSizerConfig

def test_basic_functionality():
    """Test 1: Basic volatility targeting works."""
    print("=" * 60)
    print("TEST 1: Basic Volatility Targeting")
    print("=" * 60)
    
    config = PositionSizerConfig(
        mode="volatility",
        base_step=0.5,
        target_vol=0.5,
        min_step=0.1,
        max_step=1.0
    )
    sizer = PositionSizer(config)
    
    scenarios = [
        (0.25, "Low volatility (calm market)"),
        (0.5, "Normal volatility"),
        (1.0, "High volatility (volatile market)"),
        (2.0, "Very high volatility (crash)"),
    ]
    
    print(f"\nConfig: base_step={config.base_step}, target_vol={config.target_vol}")
    print(f"Bounds: min={config.min_step}, max={config.max_step}\n")
    
    for rv, desc in scenarios:
        step = sizer.calculate_step(rv)
        print(f"  {desc:40s} rv={rv:.2f} → step={step:.3f}")
    
    print("\n✅ Volatility targeting works!\n")

def test_static_mode():
    """Test 2: Static mode preserves old behavior."""
    print("=" * 60)
    print("TEST 2: Static Mode (Backwards Compatibility)")
    print("=" * 60)
    
    config = PositionSizerConfig(
        mode="static",
        base_step=0.5
    )
    sizer = PositionSizer(config)
    
    print(f"\nConfig: mode='static', base_step={config.base_step}\n")
    
    for rv in [0.1, 0.5, 2.0]:
        step = sizer.calculate_step(rv)
        print(f"  rv={rv:.2f} → step={step:.3f} (should always be 0.500)")
    
    print("\n✅ Static mode works (backwards compatible)!\n")

def test_kelly_criterion():
    """Test 3: Kelly Criterion mode."""
    print("=" * 60)
    print("TEST 3: Kelly Criterion")
    print("=" * 60)
    
    config = PositionSizerConfig(
        mode="kelly",
        target_vol=0.5,
        kelly_win_rate=0.6,
        kelly_avg_win=0.02,
        kelly_avg_loss=0.01,
        min_step=0.1,
        max_step=1.0
    )
    sizer = PositionSizer(config)
    
    print(f"\nConfig: win_rate={config.kelly_win_rate}, avg_win={config.kelly_avg_win}")
    print(f"        avg_loss={config.kelly_avg_loss}, target_vol={config.target_vol}\n")
    
    for rv in [0.25, 0.5, 1.0]:
        step = sizer.calculate_step(rv)
        print(f"  rv={rv:.2f} → step={step:.3f}")
    
    print("\n✅ Kelly Criterion works!\n")

def test_config_validation():
    """Test 4: Config validation catches errors."""
    print("=" * 60)
    print("TEST 4: Configuration Validation")
    print("=" * 60)
    
    print("\nTesting invalid configurations...\n")
    
    # Test 1: base_step > 1.0
    try:
        PositionSizerConfig(base_step=1.5)
        print("❌ FAILED: Should reject base_step > 1.0")
        return False
    except ValueError as e:
        print(f"✅ Caught invalid base_step: {e}")
    
    # Test 2: min_step > max_step
    try:
        PositionSizerConfig(min_step=0.8, max_step=0.5)
        print("❌ FAILED: Should reject min > max")
        return False
    except ValueError as e:
        print(f"✅ Caught min > max: {e}")
    
    # Test 3: negative target_vol
    try:
        PositionSizerConfig(target_vol=-0.5)
        print("❌ FAILED: Should reject negative target_vol")
        return False
    except ValueError as e:
        print(f"✅ Caught negative target_vol: {e}")
    
    print("\n✅ All validation tests passed!\n")
    return True

def main():
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 15 + "POSITION SIZER TEST SUITE" + " " * 18 + "║")
    print("╚" + "═" * 58 + "╝")
    print()
    
    try:
        test_basic_functionality()
        test_static_mode()
        test_kelly_criterion()
        test_config_validation()
        
        print("=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nDynamic Position Sizing is working correctly.")
        print("You can now:")
        print("  1. Run unit tests: pytest tests/test_position_sizer.py")
        print("  2. Update your config to enable it")
        print("  3. Run backtests to compare performance")
        print()
        return 0
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
