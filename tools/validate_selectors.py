
import sys
import os
import json
import pandas as pd
import numpy as np

# Add tools to path to import wfo_select_best
sys.path.append(os.path.join(os.getcwd(), 'tools'))
try:
    import wfo_select_best
except ImportError:
    # Fallback if running from tools dir
    sys.path.append(os.getcwd())
    import wfo_select_best

# Load WFO Results
WFO_CSV = "results/wfo_trend_WFO_NEW_LONG.csv"
print(f"Loading {WFO_CSV}...")
df_raw = pd.read_csv(WFO_CSV)

# Pre-calc metrics expected by wfo_select_best
# We need to replicate the dataframe prep from wfo_select_best.main()
# But wfo_select_best.main() does a lot of prep.
# Ideally we'd modify wfo_select_best to have a `process_df` function, but we can just duplicate the prep here.

df = df_raw.copy()
# Parse best_params JSON
df["params"] = df["best_params"].apply(lambda x: json.loads(x) if isinstance(x, str) else x)

# Calculate metrics
df["train_test_ratio"] = df["oos_profit"] / df["train_profit"].replace(0, 1e-9)
df["train_test_gap"] = abs(df["oos_profit"] - df["train_profit"])
df["suspicious"] = (df["train_test_ratio"] > 1.5) | (df["train_test_ratio"] < 0.7)

# Recency weight
n = len(df)
df["recency_weight"] = np.exp(np.linspace(0, 1, n))

# Stability Penalty (Logic copied from wfo_select_best)
all_params_list = df["params"].tolist()
numeric_keys = set()
for p in all_params_list:
    if isinstance(p, dict):
        for k, v in p.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                numeric_keys.add(k)

param_stats = {}
for key in numeric_keys:
    values = [p.get(key) for p in all_params_list if isinstance(p, dict) and key in p and isinstance(p[key], (int, float))]
    if len(values) > 1:
        param_stats[key] = {"mean": np.mean(values), "std": np.std(values) + 1e-9}

stability_penalties = []
for p in all_params_list:
    if not isinstance(p, dict):
        stability_penalties.append(0)
        continue
    z_scores = []
    for key, stats in param_stats.items():
        if key in p and isinstance(p[key], (int, float)):
            z = abs(p[key] - stats["mean"]) / stats["std"]
            z_scores.append(z)
    stability_penalties.append(np.mean(z_scores) if z_scores else 0)

df["stability_penalty"] = stability_penalties

# --- Define Helper to Run Selection ---
def run_selection(strategy, ensemble_n=5, stability_lambda=0.1):
    # Calculate SCORE based on strategy
    temp_df = df.copy()
    
    # Base Components
    temp_df["score_weighted"] = (
        temp_df["oos_profit"] * 0.6
        + (temp_df["oos_profit"] + temp_df["train_profit"]) / 2 * 0.3
        + temp_df["recency_weight"] / temp_df["recency_weight"].max() * 0.1
        - temp_df["train_test_gap"] * 0.2
    )

    if strategy == "best_oos":
        temp_df["score"] = temp_df["oos_profit"]
        
    elif strategy == "consistent":
        temp_df["harmonic_mean"] = 2 * temp_df["oos_profit"] * temp_df["train_profit"] / (temp_df["oos_profit"] + temp_df["train_profit"] + 1e-9)
        temp_df["score"] = temp_df["harmonic_mean"] - temp_df["train_test_gap"] * 0.5
        
    elif strategy == "recent":
        temp_df["score"] = temp_df["oos_profit"] * temp_df["recency_weight"] / temp_df["recency_weight"].max()
        
    elif strategy == "weighted":
        temp_df["score"] = temp_df["score_weighted"]

    elif strategy in ["stable", "stable_ensemble"]:
        # Weighted metric - stability penalty
        temp_df["score"] = temp_df["score_weighted"] - temp_df["stability_penalty"] * stability_lambda
        
    elif strategy == "ensemble":
         # Ensemble uses weighted score to find top N
         temp_df["score"] = temp_df["score_weighted"]

    # Sort
    temp_df = temp_df.sort_values("score", ascending=False)
    
    # Select
    if strategy in ["ensemble", "stable_ensemble"]:
        candidates = temp_df.head(ensemble_n)
        return wfo_select_best.ensemble_average_params(candidates, ensemble_n)
    else:
        # Single best
        return temp_df.iloc[0]["params"]


# --- Generate ALL Configs ---
base_config = json.load(open("configs/WFO_NEW_LONG_20251230_173947.json"))

strategies = [
    "best_oos", 
    "weighted", 
    "consistent", 
    "recent", 
    "ensemble", 
    "stable_ensemble"
]

for strat in strategies:
    print(f"Generating {strat} config...")
    # Use N=5 for ensembles, default lambda
    params = run_selection(strat, ensemble_n=5)
    
    cfg = base_config.copy()
    if "trend_overrides" not in cfg["strategy"]:
        cfg["strategy"]["trend_overrides"] = {}

    # Update params
    for k, v in params.items():
         cfg["strategy"]["trend_overrides"][k] = v
             
    out_name = f"configs/WFO_TEST_{strat.upper()}.json"
    with open(out_name, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Saved {out_name}")

print("Done generating configs.")
