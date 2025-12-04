#!/usr/bin/env python3
import time
import os
import sys

# Ensure tools/ and core/ imports work if run from container root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import reconcile_pnl  # type: ignore


def main():
    interval_sec = float(os.getenv("PNL_RECONCILE_INTERVAL_SEC", 4 * 60 * 60))

    while True:
        try:
            reconcile_pnl.main()
        except Exception as e:
            # Last-resort logging to stdout so Docker logs show it
            print(f"[pnl_reconciler] ERROR: {e}", file=sys.stderr)

        time.sleep(interval_sec)


if __name__ == "__main__":
    main()