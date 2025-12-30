
#!/bin/bash
python3 core/ethbtc_accum_bot.py backtest --data data/raw/BTCUSDT_15m_2021-2025_vision.csv --funding-data data/raw/BTCUSDT_funding_2021-2025.csv --config configs/WFO_TEST_WEIGHTED.json --out results/bt_test_weighted.csv --report
python3 core/ethbtc_accum_bot.py backtest --data data/raw/BTCUSDT_15m_2021-2025_vision.csv --funding-data data/raw/BTCUSDT_funding_2021-2025.csv --config configs/WFO_TEST_CONSISTENT.json --out results/bt_test_consistent.csv --report
python3 core/ethbtc_accum_bot.py backtest --data data/raw/BTCUSDT_15m_2021-2025_vision.csv --funding-data data/raw/BTCUSDT_funding_2021-2025.csv --config configs/WFO_TEST_RECENT.json --out results/bt_test_recent.csv --report
python3 core/ethbtc_accum_bot.py backtest --data data/raw/BTCUSDT_15m_2021-2025_vision.csv --funding-data data/raw/BTCUSDT_funding_2021-2025.csv --config configs/WFO_TEST_ENSEMBLE.json --out results/bt_test_ensemble.csv --report
python3 core/ethbtc_accum_bot.py backtest --data data/raw/BTCUSDT_15m_2021-2025_vision.csv --funding-data data/raw/BTCUSDT_funding_2021-2025.csv --config configs/WFO_TEST_STABLE_ENSEMBLE.json --out results/bt_test_stable_ensemble.csv --report
