#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)

python3 tools/optimizer_cli.py \
  --data data/raw/ETHBTC_15m_2021-2025_vision.csv \
  --funding-data data/raw/ETHUSDT_funding_2021-2025.csv \
  --config configs/debug_test.json \
  --train-start 2023-01-01 \
  --train-end 2023-12-31 \
  --test-start 2024-01-01 \
  --test-end 2024-06-01 \
  --n-trials 20 \
  --jobs 1 \
  --study-name ethbtc_opt_demo \
  --out results/opt_results_demo.csv
