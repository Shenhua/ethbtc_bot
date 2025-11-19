#!/bin/sh
set -e

# Allow overriding API keys etc via environment
export BINANCE_KEY="${BINANCE_KEY:-}"
export BINANCE_SECRET="${BINANCE_SECRET:-}"
export BINANCE_BASE_URL="${BINANCE_BASE_URL:-}"

# Optional: metrics/status ports (9109, 9110 default in live_executor)
export METRICS_PORT="${METRICS_PORT:-9109}"
export STATUS_PORT="${STATUS_PORT:-9110}"

# Ensure /data exists for state/trades if mounted
mkdir -p /data

exec "$@"