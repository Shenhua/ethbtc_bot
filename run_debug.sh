#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
python3 live_executor.py --params configs/debug_test.json --mode dry --symbol ETHBTC --state run_state/debug_state.json
