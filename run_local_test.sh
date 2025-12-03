#!/bin/bash
set -e  # Exit on error

echo "🚀 Setting up Local Test Environment (with venv)..."

# 1. Create venv if it doesn't exist
if [ ! -d "venv" ]; then
    echo "🔨 Creating virtual environment..."
    python3 -m venv venv
fi

# 2. Activate venv
source venv/bin/activate

# 3. Install Dependencies
echo "📦 Installing dependencies into venv..."
pip install -r requirements.txt

# 4. Run Spot Test (ETHBTC)
echo "---------------------------------------------------"
echo "🏃 Running Test 1: SPOT (ETHBTC)..."
echo "---------------------------------------------------"
python live_executor.py \
  --params configs/prod_meta_live.json \
  --symbol ETHBTC \
  --mode dry \
  --once

# 5. Run Futures Test (BTCUSDT)
# We use the futures config we saw earlier
echo "---------------------------------------------------"
echo "🏃 Running Test 2: FUTURES (BTCUSDT)..."
echo "---------------------------------------------------"
python live_executor.py \
  --params configs/prod_btc_meta_live.json \
  --symbol BTCUSDT \
  --mode dry \
  --once

echo "---------------------------------------------------"
echo "✅ All Tests Complete! (Spot & Futures)"
