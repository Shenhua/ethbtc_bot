#!/usr/bin/env python3
"""
Walk-Forward Analyzer (wfo_analyzer.py)

This tool validates the Walk-Forward Optimization (WFO) *process*.
Instead of just selecting the "best" historical window, it:
1.  Concatenates all Out-of-Sample (OOS) periods to create a realistic
    "What-If" equity curve that simulates continuous re-optimization.
2.  Calculates Walk-Forward Efficiency (WFE) = (OOS Performance) / (IS Performance).
3.  Analyzes parameter stability across windows to detect "Drift" (overfitting).

Usage:
    python3 tools/wfo_analyzer.py --wfo-csv results/wfo_trend_BTC.csv --out results/wfo_analysis.json

Author: AI Audit System
Date: 2024-12-30
"""

import argparse
import json
import logging
import sys
from typing import Any, Dict, List

import pandas as pd
import numpy as np

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [WFO-ANALYZER] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("wfo_analyzer")


def load_wfo_results(csv_path: str) -> pd.DataFrame:
    """
    Load and parse the WFO results CSV file.
    
    The expected columns are:
    - window_end: Timestamp of the training window end.
    - oos_profit: Final BTC value at end of Out-of-Sample period.
    - train_profit: Final BTC value at end of In-Sample (training) period.
    - best_params: JSON string of the optimal parameters for this window.
    
    Returns:
        A DataFrame with parsed parameters.
    """
    log.info(f"Loading WFO results from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    if df.empty:
        log.error("WFO results CSV is empty!")
        sys.exit(1)
    
    # Parse the best_params JSON column
    df["params_dict"] = df["best_params"].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
    
    log.info(f"Loaded {len(df)} WFO windows.")
    return df


def calculate_wfo_efficiency(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate the Walk-Forward Efficiency (WFE).
    
    WFE is a measure of how well In-Sample (training) performance
    translates to Out-of-Sample (live) performance.
    
    Formula:
        Total OOS Return = Average(oos_profit) across all windows
        Total IS Return = Average(train_profit) across all windows
        WFE = Total OOS Return / Total IS Return
    
    A WFE of 0.5 means OOS performance is 50% of IS performance.
    A WFE > 0.7 is generally considered robust.
    """
    total_oos_return = df["oos_profit"].mean()
    total_is_return = df["train_profit"].mean()
    
    # Guard against division by zero
    if total_is_return == 0:
        wfe = 0.0
        log.warning("Walk-Forward Efficiency is 0.0 due to zero IS return.")
    else:
        wfe = total_oos_return / total_is_return
    
    log.info(f"Walk-Forward Efficiency (WFE): {wfe:.2%}")
    log.info(f"  - Avg. OOS Profit: {total_oos_return:.4f}")
    log.info(f"  - Avg. IS (Train) Profit: {total_is_return:.4f}")
    
    return {
        "wfe": wfe,
        "avg_oos_profit": total_oos_return,
        "avg_is_profit": total_is_return,
    }


def stitch_oos_equity_curve(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Conceptually stitch together OOS returns into a single curve.
    
    This simulates a "live" equity trajectory where you re-optimize
    every 30 days and trade the OOS period with the new parameters.
    The "cumulative return" compounds all OOS periods.
    
    Returns:
        A list of dicts representing the equity curve points.
    """
    log.info("Stitching concatenated OOS equity curve...")
    
    cumulative_wealth = 1.0  # Start with 1 BTC
    equity_curve = []
    
    for idx, row in df.iterrows():
        # The OOS profit *is* the final_btc value after starting at 1.0 BTC
        # So, the period return is (oos_profit / 1.0 - 1) = oos_profit - 1
        # If oos_profit = 1.02, the window made +2%.
        period_return = row["oos_profit"] - 1.0
        
        # Compound: new_wealth = old_wealth * (1 + return)
        cumulative_wealth = cumulative_wealth * (1.0 + period_return)
        
        equity_curve.append({
            "window_end": str(row["window_end"]),
            "period_oos_return_pct": period_return * 100,
            "cumulative_wealth_btc": cumulative_wealth
        })
        
    log.info(f"Final Concatenated OOS Wealth: {cumulative_wealth:.4f} BTC")
    return equity_curve


def analyze_parameter_stability(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze how much the "optimal" parameters vary across windows.
    High variance = Overfitting to noise.
    
    We extract key numeric parameters and compute their standard deviation
    relative to their mean (Coefficient of Variation).
    """
    log.info("Analyzing parameter stability across windows...")
    
    # Expand the params_dict into columns
    params_df = pd.json_normalize(df["params_dict"])
    
    # Find numeric columns
    numeric_cols = params_df.select_dtypes(include=np.number).columns.tolist()
    
    stability_report = {}
    for col in numeric_cols:
        mean_val = params_df[col].mean()
        std_val = params_df[col].std()
        
        # Coefficient of Variation (CV) - lower is more stable
        if mean_val != 0:
            cv = std_val / abs(mean_val)
        else:
            cv = np.inf if std_val > 0 else 0.0
        
        # Determine stability (heuristic: CV < 50% is stable)
        is_stable = bool(cv < 0.5)  # Explicit bool conversion for JSON
        
        stability_report[col] = {
            "mean": float(round(mean_val, 4)),  # Ensure native Python float
            "std": float(round(std_val, 4)),
            "cv": float(round(cv, 4)),  # Coefficient of Variation
            "is_stable": is_stable
        }
        
        if cv > 0.5:
            log.warning(f"  ⚠️ UNSTABLE PARAM: '{col}' (CV={cv:.2%})")
        else:
            log.debug(f"  ✅ Stable param: '{col}' (CV={cv:.2%})")
            
    return stability_report


def main():
    parser = argparse.ArgumentParser(description="Walk-Forward Analyzer")
    parser.add_argument("--wfo-csv", required=True, help="Path to the WFO results CSV file.")
    parser.add_argument("--out", default="results/wfo_analysis.json", help="Output path for the analysis JSON.")
    args = parser.parse_args()
    
    # 1. Load Data
    df = load_wfo_results(args.wfo_csv)
    
    # 2. Calculate WFE
    wfe_metrics = calculate_wfo_efficiency(df)
    
    # 3. Stitch OOS Curve
    equity_curve = stitch_oos_equity_curve(df)
    
    # 4. Analyze Stability
    stability = analyze_parameter_stability(df)
    
    # 5. Assemble Report
    report = {
        "wfo_efficiency": wfe_metrics,
        "concatenated_oos_curve": equity_curve,
        "parameter_stability": stability,
        "summary": {
            "final_concatenated_wealth": equity_curve[-1]["cumulative_wealth_btc"] if equity_curve else 0.0,
            "wfe_score": wfe_metrics["wfe"],
            "num_windows": len(df),
        }
    }
    
    # 6. Save Report
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    
    log.info(f"✅ Analysis complete. Report saved to: {args.out}")
    
    # 7. Print Summary
    print("\n" + "=" * 60)
    print("WALK-FORWARD ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"  WFE Score:                  {wfe_metrics['wfe']:.2%}")
    print(f"  Final Concatenated Wealth:  {report['summary']['final_concatenated_wealth']:.4f} BTC")
    print(f"  Number of Windows:          {len(df)}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
