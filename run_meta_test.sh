#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)

python3 core/ethbtc_accum_bot.py backtest \
  --data data/raw/ETHBTC_15m_2021-2025_vision.csv \
  --funding-data data/raw/ETHUSDT_funding_2021-2025.csv \
  --config configs/meta_optimized.json \
  --out results/meta_test_results.csv
