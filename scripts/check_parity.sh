#!/bin/bash
# scripts/check_parity.sh
# Checks for parity issues between backtest and live executor

echo "🔍 Checking for direct cfg.strategy access in live_executor.py..."
# Exclude model_dump which is valid for extracting base params
VIOLATIONS=$(grep -n "cfg\.strategy\." live_executor.py | grep -v "model_dump" | grep -v "strategy_type" | grep -v "^\s*#" | grep -v "getattr(cfg.strategy")

if [ -n "$VIOLATIONS" ]; then
    echo "❌ PARITY VIOLATION: Direct cfg.strategy access found:"
    echo "$VIOLATIONS"
    echo ""
    echo "Please use 'mr_params' or 'tr_params' from strategy_factory instead."
    exit 1
fi

echo "✅ No direct cfg.strategy access found"

echo ""
echo "🧪 Running parity tests..."
# Run pytest on the parity suite
# Using python -m pytest to ensure path is correct
python3 -m pytest tests/test_backtest_live_parity.py -v

if [ $? -ne 0 ]; then
    echo "❌ Parity tests failed!"
    exit 1
fi

echo ""
echo "🧪 Running ML regime parity tests..."
python3 -m pytest tests/test_ml_regime_parity.py -v

if [ $? -ne 0 ]; then
    echo "❌ ML regime parity tests failed!"
    exit 1
fi

echo ""
echo "🧪 Running feature engineering tests..."
python3 -m pytest tests/test_regime_features.py -v

if [ $? -ne 0 ]; then
    echo "❌ Feature engineering tests failed!"
    exit 1
fi

echo "✅ All parity checks passed!"
exit 0
