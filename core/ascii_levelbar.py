#!/usr/bin/env python3
"""
ASCII signal meters for live trading logs.

Exports
-------
- ascii_level_bar(ratio, entry, exitb, width=48, ascii_only=False) -> str
    Visual meter with landmarks: [-entry ... -exit ... 0 ... +exit ... +entry]
    Markers: B=buy entry, b=buy exit, 0=zero, s=sell exit, S=sell entry, |=current ratio

- dist_to_buy_sell_bps(ratio, entry, exitb) -> (dist_to_buy_bps, dist_to_sell_bps)
    Positive distances in bps to the next BUY/SELL entry thresholds. 0 if already inside that zone.

- live_block(current, target, min_val, max_val, width=40, history=None, label="", units="", ascii_only=False) -> str
    Backwards compatible simple bar + optional sparkline (used elsewhere in your logs).

Notes
-----
- "entry" and "exitb" are positive numbers representing +entry / +exit magnitudes.
  Negative side thresholds are symmetric: -entry, -exitb.
- The meter range is chosen to be a bit wider than ±entry so you can see when you're close.
- Colors are used only if stdout is a TTY; otherwise ASCII-safe output is returned.
"""

from __future__ import annotations
import sys
from typing import List, Optional, Tuple

# ANSI colors (used only if stdout isatty())
RESET = "\x1b[0m"
GREEN = "\x1b[32m"
YELLOW = "\x1b[33m"
CYAN = "\x1b[36m"

def _use_color() -> bool:
    try:
        return sys.stdout.isatty()
    except Exception:
        return False

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

# -------------------- Core meter --------------------

def ascii_level_bar(ratio: float, entry: float, exitb: float, width: int = 44) -> str:
    """
    Visual meter showing buy/sell entries & exits with the current ratio.
    Layout: [-entry.....-exit.....0.....+exit.....+entry]
            markers: B=buy entry, b=buy exit, s=sell exit, S=sell entry, | = current
    """
    # Key thresholds (symmetric bands)
    neg_entry = -entry
    neg_exit  = -exitb
    pos_exit  = exitb
    pos_entry = entry

    # Choose a symmetric display range so the ends are a bit beyond entries
    lo = -1.2 * entry
    hi = +1.2 * entry
    if lo >= hi:
        lo, hi = -1.0, 1.0  # safety

    def to_pos(x: float) -> int:
        # clamp to [0,width]
        t = (x - lo) / max(hi - lo, 1e-12)
        return max(0, min(width, int(round(t * width))))

    pos_neg_entry = to_pos(neg_entry)
    pos_neg_exit  = to_pos(neg_exit)
    pos_zero      = to_pos(0.0)
    pos_pos_exit  = to_pos(pos_exit)
    pos_pos_entry = to_pos(pos_entry)
    pos_ratio     = to_pos(ratio)

    # Build the bar
    bar = ["-"] * (width + 1)

    # Landmarks
    for idx, ch in [
        (pos_neg_entry, "B"),  # BUY entry (left)
        (pos_neg_exit,  "b"),  # BUY exit
        (pos_zero,      "0"),
        (pos_pos_exit,  "s"),  # SELL exit
        (pos_pos_entry, "S"),  # SELL entry (right)
    ]:
        if 0 <= idx <= width:
            bar[idx] = ch

    # Current ratio marker ('|'), wins over landmarks if overlapping
    if 0 <= pos_ratio <= width:
        bar[pos_ratio] = "|"

    return "[" + "".join(bar) + "]"

def dist_to_buy_sell_bps(ratio: float, entry: float, exitb: float) -> Tuple[float, float]:
    """Return (dist_to_buy_bps, dist_to_sell_bps).
    
    - Distance to BUY is how many bps you need to move *down* to reach -entry.
      If already in BUY zone (ratio <= -entry) → 0.0.
    - Distance to SELL is how many bps you need to move *up* to reach +entry.
      If already in SELL zone (ratio >= +entry) → 0.0.
    """
    entry = abs(entry)
    neg_entry = -entry
    # Distance in *ratio* units
    d_buy  = max(0.0, ratio - neg_entry)     # positive if above -entry
    d_sell = max(0.0, entry - ratio)         # positive if below +entry
    # Convert to basis points
    return (d_buy * 1e4, d_sell * 1e4)
