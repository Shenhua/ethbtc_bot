#!/usr/bin/env python3
from __future__ import annotations
import argparse, subprocess, json, tempfile, os, sys


# --- MAGIC PATH FIX ---
# Allow importing 'core' even if running from tools/ folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ----------------------
try:
    import optuna
except Exception:
    print("This tool requires 'optuna'. Install with: pip install optuna", file=sys.stderr); sys.exit(1)

def run_backtest(data, bnb_data, params_json, start, end, basis_btc=0.16):
    cmd = [
        sys.executable, "ethbtc_accum_bot.py", "backtest",
        "--data", data, "--bnb-data", bnb_data, "--basis-btc", str(basis_btc),
        "--config", params_json, "--start", start, "--end", end
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = p.stdout
    # naive parsing fallback — adapt to your output
    score = 0.0
    for line in out.splitlines():
        if "final_btc" in line or "wealth" in line.lower():
            for tok in line.replace(","," ").split():
                try:
                    score = float(tok)
                except Exception:
                    pass
    return score

def suggest_params(trial):
    return {
        "trend_kind": trial.suggest_categorical("trend_kind", ["roc","sma"]),
        "trend_lookback": trial.suggest_int("trend_lookback", 20, 360),
        "flip_band_entry": trial.suggest_float("flip_band_entry", 0.005, 0.08),
        "flip_band_exit": trial.suggest_float("flip_band_exit", 0.002, 0.06),
        "vol_window": trial.suggest_int("vol_window", 10, 120),
        "vol_adapt_k": trial.suggest_float("vol_adapt_k", 0.0, 0.01),
        "target_vol": trial.suggest_float("target_vol", 0.0, 1.0),
        "cooldown_minutes": trial.suggest_int("cooldown_minutes", 0, 24*60),
        "step_allocation": trial.suggest_float("step_allocation", 0.1, 1.0),
        "max_position": trial.suggest_float("max_position", 0.2, 1.0),
        "gate_window_days": trial.suggest_int("gate_window_days", 0, 90),
        "gate_roc_threshold": trial.suggest_float("gate_roc_threshold", 0.0, 0.1),
        "rebalance_threshold_w": trial.suggest_float("rebalance_threshold_w", 0.0, 0.1),
    }

def main():
    ap = argparse.ArgumentParser("optuna optimizer")
    ap.add_argument("--data", required=True)
    ap.add_argument("--bnb-data", required=True)
    ap.add_argument("--train-start", required=True)
    ap.add_argument("--train-end", required=True)
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--basis-btc", type=float, default=0.16)
    ap.add_argument("--emit-config", default="out/optuna_selected_params.json")
    args = ap.parse_args()

    os.makedirs("out", exist_ok=True)

    def objective(trial):
        params = suggest_params(trial)
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
            json.dump(params, tf); tf.flush()
            train_score = run_backtest(args.data, args.bnb_data, tf.name, args.train_start, args.train_end, args.basis_btc)
        return train_score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.trials, n_jobs=1)

    best = suggest_params(study.best_trial)
    os.makedirs(os.path.dirname(args.emit_config), exist_ok=True)
    with open(args.emit_config, "w") as f:
        json.dump(best, f, indent=2)
    print("Best params saved to", args.emit_config)

if __name__ == "__main__":
    main()
