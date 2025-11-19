#!/usr/bin/env bash
set -euo pipefail

# If first arg is an option, run the bot directly
if [[ "${1:-}" = -* ]]; then
  exec python live_executor.py "$@"
fi

# Convenience: `run` behaves like "python live_executor.py ..."
if [[ "${1:-}" = "run" ]]; then
  shift
  exec python live_executor.py "$@"
fi

# Otherwise, execute whatever the user passed (e.g., bash)
exec "$@"