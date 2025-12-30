# Optimization Orchestrator Manual

The new Python-based orchestrator (`tools/run_optimization.py`) replaces the legacy shell script with a more robust, feature-rich optimization engine.

## 🚀 Key Features

- **Unified CLI**: Single entry point for all optimization steps.
- **Real-time Progress**: Rich progress bars with ETA for all steps.
- **Parallel Execution**: Uses `ProcessPoolExecutor` for parallel optimization.
- **Unified Logging**: All logs (orchestrator + subprocesses) captured in a single file.
- **Smart Selection**: Advanced strategies for picking best parameters (Ensemble, Stability, etc.).
- **Error Handling**: Graceful failure recovery and detailed error reporting.

---

## 🛠 Usage

### Basic Usage

```bash
# Run standard optimization (WFO + Exhaustive)
python3 tools/run_optimization.py --wfo --exhaustive --long-only-futures

# Run fast test (fewer trials)
python3 tools/run_optimization.py --wfo --mr-trials 10 --trend-trials 10
```

### Modes

| Flag | Description |
|------|-------------|
| `--wfo` | Enables Walk-Forward Optimization (recommended). |
| `--exhaustive` | Runs all combinations (grid search) for MR strategy. |
| `--long-only-futures` | Constrains to long-only strategies (e.g., for futures). |
| `--futures-only` | Constrains to short-only strategies (legacy name). |

---

## 🧠 Selection Strategies

When running WFO, the system produces optimal parameters for **each window**. The selection strategy determines how to pick the single "best" parameter set for production.

**Usage:**
```bash
python3 tools/run_optimization.py --wfo ... --selection-strategy [STRATEGY]
```

### Available Strategies

| Strategy | Description | Best For | CLI Args |
|----------|-------------|----------|----------|
| **`weighted`** | **(Default)** Balanced score of OOS profit (60%), Consistency (30%), Recency (10%). | General purpose balance. | N/A |
| **`stable`** | Penalizes windows with "outlier" parameters that deviate from the global average. | Long-term robustness. | `--stability-lambda 0.1` |
| **`ensemble`** | **Averages** parameters from the top N best windows. | Smoothing noise, avoiding overfitting. | `--ensemble-n 5` |
| **`stable_ensemble`** | **Combined**: Penalizes stability outliers, then averages the top N stable ones. | **Production standard.** | Both |
| **`best_oos`** | Picks window with highest raw OOS profit. | Maximum theoretical performance. | N/A |
| **`consistent`** | Favors windows where Train and Test performance are similar. | Conservative / Predictable. | N/A |
| **`recent`** | Heavily weights recent windows. | Adapting to current market regime. | N/A |

### Strategy Details

#### 1. Stable Strategy (`--selection-strategy stable`)
Calculates a "stability penalty" for each window by comparing its parameters to the global mean/std across all windows.
- **Formula**: `Score = WeightedScore - λ * StabilityPenalty`
- **Tuning**: Use `--stability-lambda` (default 0.1) to control how much to penalize outliers. Higher = more conservative.

#### 2. Ensemble Strategy (`--selection-strategy ensemble`)
Instead of picking one window, it takes the **top N** windows (by weighted score) and averages their parameters.
- **Numeric Params**: Mean average (e.g., `fast_period=25`).
- **Categorical Params**: Majority vote (Mode).
- **Tuning**: Use `--ensemble-n` (default 5) to control how many windows to average.

#### 3. Stable Ensemble (`--selection-strategy stable_ensemble`)
The most robust option. It first filters for windows with stable parameters (using the penalty), then averages the top N surviving windows. This addresses both "parameter drift" and "noise" simultaneously.
- **Default Lambda**: `0.1` (Configurable via `--stability-lambda`)
- **Default Ensemble N**: `5` (Configurable via `--ensemble-n`)
- **Combo logic**: It uses the same score as `stable` to rank windows, then performs the same averaging as `ensemble` on the winners.

---

## 📊 Logging & Analysis

### Log Files
Logs are saved to `logs/<run_id>.log`.
- **Unified**: Contains output from the main orchestrator AND all parallel subprocesses.
- **Clean**: Parallel outputs are prefixed with combo names (e.g., `[sma-static-long]`).
- **Verbose**: Optuna trial details are captured for deep debugging.

---

## ❓ Troubleshooting

**Q: "Value: -1000000000.0" in logs?**
A: Normal. This is a penalty value for invalid trials (e.g., 0 trades or crashing errors). The optimizer learns to avoid these regions.

**Q: "Final Concatenated OOS Wealth: 277 BTC"?**
A: This is a **theoretical maximum**. It assumes you perfectly switched to the optimal parameter set for every single window in the past. Real performance will be much closer to the single-run backtest (~2.25x).

**Q: Logs show warnings about "UNSTABLE PARAM"?**
A: Use the `stable_ensemble` selection strategy to mitigate this. High instability implies the parameter might be chasing market noise rather than signal.
