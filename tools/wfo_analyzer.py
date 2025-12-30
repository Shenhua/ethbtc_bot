#!/usr/bin/env python3
"""
Walk-Forward Analyzer (wfo_analyzer.py)

This tool validates the Walk-Forward Optimization (WFO) *process*.
It detects if multiple combinations (e.g. roc-static, sma-volatility) exist in the results
and analyzes each one separately to avoid "Frankenstein" equity curves.

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
    - combo: (Optional) Strategy combination identifier.
    
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
    
    # Ensure combo column exists
    if "combo" not in df.columns:
        df["combo"] = "default"
        
    log.info(f"Loaded {len(df)} WFO windows across {df['combo'].nunique()} combinations.")
    return df


def calculate_wfo_efficiency(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate the Walk-Forward Efficiency (WFE).
    """
    total_oos_return = df["oos_profit"].mean()
    total_is_return = df["train_profit"].mean()
    
    # Guard against division by zero
    if total_is_return == 0:
        wfe = 0.0
    else:
        wfe = total_oos_return / total_is_return
    
    return {
        "wfe": wfe,
        "avg_oos_profit": total_oos_return,
        "avg_is_profit": total_is_return,
    }


def stitch_oos_equity_curve(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Conceptually stitch together OOS returns into a single curve.
    Should be called on a filtered DataFrame (single combo).
    """
    # Sort by window_end to ensure chronological order
    df = df.sort_values("window_end")
    
    cumulative_wealth = 1.0  # Start with 1 BTC
    equity_curve = []
    
    for idx, row in df.iterrows():
        # The OOS profit *is* the final_btc value after starting at 1.0 BTC
        # So, the period return is (oos_profit / 1.0 - 1) = oos_profit - 1
        period_return = row["oos_profit"] - 1.0
        
        # Compound: new_wealth = old_wealth * (1 + return)
        cumulative_wealth = cumulative_wealth * (1.0 + period_return)
        
        equity_curve.append({
            "window_end": str(row["window_end"]),
            "period_oos_return_pct": period_return * 100,
            "cumulative_wealth_btc": cumulative_wealth
        })
        
    return equity_curve


def analyze_parameter_stability(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze how much the "optimal" parameters vary across windows.
    """
    # Expand the params_dict into columns
    params_df = pd.json_normalize(df["params_dict"])
    
    # Find numeric columns
    numeric_cols = params_df.select_dtypes(include=np.number).columns.tolist()
    
    stability_report = {}
    for col in numeric_cols:
        mean_val = params_df[col].mean()
        std_val = params_df[col].std()
        
        if mean_val != 0:
            cv = std_val / abs(mean_val)
        else:
            cv = np.inf if std_val > 0 else 0.0
        
        is_stable = bool(cv < 0.5)
        
        stability_report[col] = {
            "mean": float(round(mean_val, 4)),
            "std": float(round(std_val, 4)),
            "cv": float(round(cv, 4)),
            "is_stable": is_stable
        }
            
    return stability_report


def main():
    parser = argparse.ArgumentParser(description="Walk-Forward Analyzer")
    parser.add_argument("--wfo-csv", required=True, help="Path to the WFO results CSV file.")
    parser.add_argument("--out", default="results/wfo_analysis.json", help="Output path for the analysis JSON.")
    args = parser.parse_args()
    
    # 1. Load Data
    full_df = load_wfo_results(args.wfo_csv)
    
    final_report = {}
    combos = full_df["combo"].unique()
    
    print("\n" + "=" * 80)
    print(f"WALK-FORWARD ANALYSIS ({len(combos)} Combinations)")
    print("=" * 80)
    print(f"{'Combo':<25} | {'WFE':<8} | {'Final Wealth':<12} | {'Windows':<8}")
    print("-" * 80)

    best_wealth = -1.0
    best_combo_report = None

    for combo in combos:
        # Filter for this combo
        df = full_df[full_df["combo"] == combo].copy()
        
        # Calculate Metrics
        wfe_metrics = calculate_wfo_efficiency(df)
        equity_curve = stitch_oos_equity_curve(df)
        stability = analyze_parameter_stability(df)
        
        final_wealth = equity_curve[-1]["cumulative_wealth_btc"] if equity_curve else 0.0
        
        # Build Report Block
        combo_report = {
            "wfo_efficiency": wfe_metrics,
            "concatenated_oos_curve": equity_curve,
            "parameter_stability": stability,
            "summary": {
                "final_concatenated_wealth": final_wealth,
                "wfe_score": wfe_metrics["wfe"],
                "num_windows": len(df),
            }
        }
        
        final_report[combo] = combo_report
        
        # Print Row
        print(f"{combo:<25} | {wfe_metrics['wfe']:.2%}   | {final_wealth:.4f} BTC   | {len(df):<8}")

        # Track "Best" for summary JSON structure compatibility if needed
        # (For now we dump the full dict, but simple tools might want the 'best' at top level)
        if final_wealth > best_wealth:
            best_wealth = final_wealth
            best_combo_report = combo_report

    print("=" * 80 + "\n")
    
    # 2. Save Report
    # We save the FULL breakdown keyed by combo.
    # If a tool expects the flat structure, it might break, but this is accurate.
    # To be safe, we can add a "best_overall" key.
    
    output_payload = {
        "combos": final_report,
        "best_strategy": best_combo_report,  # Backwards compatibility / ease of use
        "summary": {
            "best_wealth": best_wealth,
            "analyzed_combinations": list(combos)
        }
    }

    with open(args.out, "w") as f:
        json.dump(output_payload, f, indent=2)
    
    log.info(f"✅ Analysis complete. Report saved to: {args.out}")


if __name__ == "__main__":
    main()
