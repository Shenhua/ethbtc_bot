#!/usr/bin/env python3
"""
ASCII signal meters for live trading logs.

This module provides tools to visualize the current signal ratio relative to
trading bands (entry/exit) using ASCII art. It helps in quickly understanding
the bot's state from the console logs.

Exports
-------
- ascii_level_bar: Generates a visual meter string.
- dist_to_buy_sell_bps: Calculates distance to trading thresholds in basis points.
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
    Generates a visual ASCII meter showing buy/sell entries & exits with the current ratio.

    Layout: [-entry.....-exit.....0.....+exit.....+entry]
            markers: B=buy entry, b=buy exit, s=sell exit, S=sell entry, | = current

    Args:
        ratio: Current trend ratio/deviation.
        entry: Entry threshold (absolute value).
        exitb: Exit threshold (absolute value).
        width: Width of the ASCII bar in characters.

    Returns:
        str: The ASCII meter string.
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
    """
    Calculates the distance to the nearest buy and sell entry thresholds in basis points.

    - Distance to BUY is how many bps you need to move *down* to reach -entry.
      If already in BUY zone (ratio <= -entry) → 0.0.
    - Distance to SELL is how many bps you need to move *up* to reach +entry.
      If already in SELL zone (ratio >= +entry) → 0.0.

    Args:
        ratio: Current trend ratio/deviation.
        entry: Entry threshold (absolute value).
        exitb: Exit threshold (unused in this calc but kept for interface consistency).

    Returns:
        Tuple[float, float]: (distance_to_buy_bps, distance_to_sell_bps).
    """
    entry = abs(entry)
    neg_entry = -entry
    # Distance in *ratio* units
    d_buy  = max(0.0, ratio - neg_entry)     # positive if above -entry
    d_sell = max(0.0, entry - ratio)         # positive if below +entry
    # Convert to basis points
    return (d_buy * 1e4, d_sell * 1e4)