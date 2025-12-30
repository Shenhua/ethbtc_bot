#!/usr/bin/env python3
"""
Optimization Orchestrator (run_optimization.py)

Python replacement for run_complete_optimization.sh with:
- Rich progress bars and ETA
- Parallel execution with concurrent.futures
- Proper logging with structlog
- Error handling and resumability
- Clean CLI with argparse

Usage:
    python3 tools/run_optimization.py --wfo --exhaustive
    python3 tools/run_optimization.py --wfo --long-only-futures
    python3 tools/run_optimization.py --help

Author: AI Audit System
Date: 2024-12-30
"""

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Optional: rich for better progress bars (fallback to tqdm or simple)
try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
    from rich.table import Table
    from rich.panel import Panel
    from rich import print as rprint
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    # Only warn once (not in subprocess calls)
    if os.environ.get("_OPTIMIZER_SUBPROCESS") != "1":
        print("[WARN] 'rich' not installed. Using simple progress. Install with: pip install rich")

# Fallback to tqdm if rich not available
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


# ============================================================
# CONFIGURATION
# ============================================================
@dataclass
class OptimizationConfig:
    """Configuration for the optimization run."""
    # Identifiers
    run_id: str = ""
    
    # Data paths
    price_data: str = "data/raw/BTCUSDT_15m_2021-2025_vision.csv"
    funding_data: str = "data/raw/BTCUSDT_funding_2021-2025.csv"
    tag: str = "BTC"
    
    # Date ranges (for static mode)
    train_start: str = "2021-01-01"
    train_end: str = "2024-06-30"
    test_start: str = "2024-07-01"
    test_end: str = "2025-06-01"
    
    # WFO settings
    wfo_mode: bool = False
    window_days: int = 180
    step_days: int = 30
    
    # Exhaustive mode
    exhaustive_mode: bool = False
    long_only_futures: bool = False
    futures_only: bool = False
    
    # WFO selection strategy
    selection_strategy: str = "weighted"  # 'weighted', 'ensemble', 'stable', 'stable_ensemble', 'best_oos', 'consistent', 'recent'
    ensemble_n: int = 5  # Number of windows to average for ensemble
    stability_lambda: float = 0.1  # Stability penalty weight for 'stable' strategy
    
    # Trial counts
    mr_trials: int = 50
    trend_trials: int = 30
    meta_trials: int = 8
    
    # Other
    max_workers: int = 8
    verbose: bool = False
    
    # Paths
    storage: str = "sqlite:///data/db/optuna.db"
    results_dir: str = "results"
    logs_dir: str = "logs"
    configs_dir: str = "configs"


# ============================================================
# LOGGING SETUP
# ============================================================
def setup_logging(run_id: str, verbose: bool = False) -> logging.Logger:
    """Setup logging with file and console handlers using run_id."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"{run_id}.log"
    
    level = logging.DEBUG if verbose else logging.INFO
    
    # Create logger
    logger = logging.getLogger("optimizer")
    if logger.handlers:
        logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    
    # File handler (always DEBUG)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    ))
    
    # Console handler (only prints if level matches)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter(
        "[%(asctime)s] %(message)s",
        datefmt="%H:%M:%S"
    ))
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    logger.debug(f"Log file initialized: {log_file}") # Use debug to avoid console noise on start if not verbose?
    # Actually logger.info(f"Log file: {log_file}") is fine, user likes knowing where logs are.
    logger.info(f"Log file: {log_file}")
    return logger


# ============================================================
# SUBPROCESS RUNNER
# ============================================================
def run_optimizer(
    cmd: List[str],
    description: str,
    logger: logging.Logger,
    progress: Optional[Any] = None,
    overall_task: Optional[Any] = None,
    approx_windows: int = 1,
    verbose: bool = False
) -> subprocess.CompletedProcess:
    """
    Run an optimizer script as a subprocess and stream output in real-time.
    If it emits OPTIMIZER_PROGRESS signals, visualize them with Rich.
    """
    # cmd is already a list, e.g. ["python3", "script.py", "--arg"]
        
    # cmd is already a list, e.g. ["python3", "script.py", "--arg"]
    # LOGGING: We do this later after checking for progress object to avoid breaking Rich display
    
    start = time.time()
    import json
    
    # Mode detection
    cmd_str = " ".join(cmd).lower()
    is_wfo = "--mode wfo" in cmd_str or "--wfo" in cmd_str
    
    # Try to find total trials in cmd for progress bar
    n_trials = 200 # Default
    for i, part in enumerate(cmd):
        if part == "--trials" and i + 1 < len(cmd):
            try: n_trials = int(cmd[i+1])
            except: pass
            
    total_steps = approx_windows if is_wfo else n_trials
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env={**os.environ, "_OPTIMIZER_SUBPROCESS": "1", "PYTHONUNBUFFERED": "1"}
    )
    
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn, TimeElapsedColumn, MofNCompleteColumn
    
    internal_progress = None
    if progress is None:
        internal_progress = Progress(
            SpinnerColumn(),
            TextColumn(f"[bold blue]{description}..."),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            transient=True
        )
        internal_progress.start()
    p_obj = progress if progress else internal_progress
    
    # Safe printing that doesn't break the bar
    msg = f"🚀 Running: {' '.join(cmd)}"
    if p_obj and hasattr(p_obj, "console"):
        # p_obj.console.log(msg) # User finds this noisy too? Let's hide it from console based on request "optional"
        # Actually user complained about "Trends Selection completed..." and "Running python3..." duplication bars.
        # But "Running..." is useful context. I'll print it to log file always, and console via Rich if desired.
        # The user output showed "[14:53:33] 🚀 Running: ..." breaking the bar. 
        # So we MUST NOT use logger.info here if Rich is active.
        logger.debug(msg) # File only
    else:
        logger.info(msg)

    # p_obj.console.log(f"[bold dim]{description} started...")
    task = p_obj.add_task(description, total=total_steps, transient=False)
    
    # Global stage size (1/6th of 100%)
    stage_size = 100.0 / 6.0
    
    try:
        windows_total = approx_windows
        last_overall_val = 0.0
        
        current_window = 1 # Start at window 1 context
        current_trial = 0
        
        for line in process.stdout:
            line = line.strip()
            if not line: continue
            
            # Catch progress signals
            if "OPTIMIZER_PROGRESS" in line:
                # 1. Log the signal to the file ALWAYS for debugging
                logger.debug(line) 
                
                try:
                    start_idx = line.find('{"signal": "OPTIMIZER_PROGRESS"')
                    if start_idx != -1:
                        msg = json.loads(line[start_idx:])
                        etype = msg.get("type")
                        
                        if etype == "total_windows":
                            windows_total = msg["data"]
                            total_steps = windows_total if is_wfo else n_trials
                            p_obj.update(task, total=total_steps)
                            
                        elif etype == "window":
                            current_window = msg["data"]
                            # Just snap to the start of the window
                            p_obj.update(task, completed=current_window - 1)
                            
                        elif etype == "trial":
                            trial_data = msg["data"]
                            # Use trial_count for relative progress to handle resumed studies
                            if isinstance(trial_data, dict):
                                t_rel = trial_data.get("trial_count")
                                t_abs = trial_data.get("trial_idx", 0)
                                # Prefer relative count if available, otherwise fallback to absolute
                                t_idx = t_rel if t_rel is not None else t_abs
                            else:
                                t_idx = trial_data
                            
                            if isinstance(t_idx, (int, float)):
                                current_trial = max(current_trial, t_idx)
                                
                                # fractional completion
                                if is_wfo:
                                    trial_progress = min(1.0, current_trial / max(1, n_trials))
                                    abs_completed = (current_window - 1) + trial_progress
                                else:
                                    abs_completed = current_trial
                                
                                # Cap to total to prevent overflows
                                abs_completed = min(abs_completed, total_steps)
                                    
                                # Only advance, never go backwards
                                p_obj.update(task, completed=max(p_obj._tasks[task].completed, abs_completed))
                                
                                if overall_task is not None:
                                    # Progress through this specific stage (0.0 to 1.0)
                                    latest_completed = p_obj._tasks[task].completed
                                    stage_progress = max(0.0, min(1.0, latest_completed / max(1, total_steps)))
                                    # Advance global bar proportionally
                                    delta = (stage_progress * stage_size) - last_overall_val
                                    if delta > 0:
                                        p_obj.update(overall_task, advance=delta)
                                        last_overall_val += delta
                except:
                    pass
                
                # Signal parsed and logged to file, now SKIP console print
                continue
            
            # 1. Log non-signal lines to file (ensure flush)
            logger.debug(line)
            for h in logger.handlers:
                h.flush()
                
            # 2. If verbose (-v), print to console safely via Rich
            if verbose and p_obj and hasattr(p_obj, "console"):
                p_obj.console.print(f"[dim]{line}[/dim]")
    finally:
        if internal_progress:
            internal_progress.stop()
        else:
            # Catch up overall task to ensure it finishes its 1/6th stage completely
            if overall_task is not None:
                final_delta = stage_size - last_overall_val
                if final_delta > 0:
                    p_obj.update(overall_task, advance=final_delta)
            # Finish the sub-task at 100%
            p_obj.update(task, completed=total_steps)
            # DO NOT remove_task(task) - user wants to see history
        
    process.wait()
    elapsed = time.time() - start
    
    if process.returncode != 0:
        logger.error(f"❌ {description} FAILED (exit {process.returncode})")
        raise RuntimeError(f"{description} failed")
    
    # User requested to hide "✅ ... completed" from console
    logger.debug(f"✅ {description} completed in {elapsed:.1f}s")
    return subprocess.CompletedProcess(cmd, process.returncode)


# ============================================================
# COMBO GENERATOR
# ============================================================
def get_combos(config: OptimizationConfig) -> List[Dict[str, Any]]:
    """
    Generate list of parameter combinations to test.
    
    Returns:
        List of dicts with trend_kind, sizing_mode, long_only keys.
    """
    combos = []
    
    for trend in ["sma", "roc"]:
        for sizing in ["static", "volatility"]:
            for long_only in [True, False]:
                # Apply filters
                if config.long_only_futures and not long_only:
                    continue
                if config.futures_only and long_only:
                    continue
                
                combos.append({
                    "trend_kind": trend,
                    "sizing_mode": sizing,
                    "long_only": long_only,
                    "long_str": "long" if long_only else "short"
                })
    
    return combos


# ============================================================
# PROGRESS HELPERS
# ============================================================
def mark_stage_done(progress: Optional[Any], task_id: Optional[Any], stage_idx: int):
    """Jumps the overall progress bar to the end of a specific stage (1-6)."""
    if progress and task_id is not None:
        progress.update(task_id, completed=stage_idx * (100.0 / 6.0))


# ============================================================
# STEP RUNNERS
# ============================================================
def run_mr_wfo_combo(args_tuple) -> Dict[str, Any]:
    """Worker function for exhaustive combo search."""
    config, combo, combo_idx, total_combos, total_windows, progress_queue = args_tuple
    
    combo_name = f"{combo['trend_kind']}-{combo['sizing_mode']}-{combo['long_str']}"
    
    # Build arguments
    out_file = f"results/{config.run_id}_wfo_mr_{combo['trend_kind']}_{combo['sizing_mode']}_{combo['long_str']}.csv"
    
    args = [
        "-m", "tools.optimizer.cli",
        "--strategy", "mr",
        "--mode", "wfo",
        "--tag", f"{config.tag}_{combo['trend_kind']}_{combo['sizing_mode']}_{combo['long_str']}",
        "--data", config.price_data,
        "--funding-data", config.funding_data,
        "--window-days", str(config.window_days),
        "--step-days", str(config.step_days),
        "--trials", str(config.mr_trials),
        "--jobs", "1",
        "--force-trend-kind", combo["trend_kind"],
        "--force-sizing-mode", combo["sizing_mode"],
        "--force-long-only", str(combo["long_only"]).lower(),
        "--storage", config.storage,
        "--study-name", f"mr_{config.tag}_wfo_{combo['trend_kind']}_{combo['sizing_mode']}_{combo['long_str']}",
        "--out", out_file,
    ]
    
    cmd = ["python3"] + args
    
    start = time.time()
    
    # Use Popen for real-time output streaming
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env={**os.environ, "_OPTIMIZER_SUBPROCESS": "1"}  # Suppress rich warning
    )
    
    # Stream output, parse JSON signals, and send to main log via queue
    import json
    window_count = 0
    trial_count = 0 # Track trials for this combo
    
    for line in process.stdout:
        line = line.strip()
        if not line: continue
        
        # Send all output to main log via queue (like run_optimizer does with logger.debug)
        if progress_queue:
            progress_queue.put(("log", combo_name, line))
        
        # Standardized JSON signals
        if "OPTIMIZER_PROGRESS" in line:
            try:
                start_idx = line.find('{"signal": "OPTIMIZER_PROGRESS"')
                if start_idx != -1:
                    msg = json.loads(line[start_idx:])
                    etype = msg.get("type")
                    if etype == "window":
                        window_count = msg.get("data", window_count)
                        trial_count = 0 # Reset trial count for new window
                        if progress_queue:
                            progress_queue.put(("window", combo_name, window_count))
                    elif etype == "total_windows":
                        if progress_queue:
                            progress_queue.put(("total_windows", combo_name, msg.get("data")))
                    elif etype == "trial":
                        trial_data = msg.get("data", {})
                        if progress_queue:
                            progress_queue.put(("trial", combo_name, trial_data))
            except:
                pass
                
        # Legacy fallback
        elif "[WFO]" in line and "Window" in line:
            window_count += 1
            trial_count = 0 # Reset trial count for new window
            if progress_queue:
                progress_queue.put(("window", combo_name, window_count))
    
    process.wait()
    elapsed = time.time() - start
    
    # Signal completion
    if progress_queue:
        progress_queue.put(("combo", combo_name, window_count, elapsed, process.returncode == 0))
    
    return {
        "combo": combo,
        "success": process.returncode == 0,
        "elapsed": elapsed,
        "out_file": out_file,
        "windows": window_count,
        "error": None if process.returncode == 0 else "Failed"
    }


def step1_mr_optimization(config: OptimizationConfig, logger: logging.Logger, progress: Optional[Any] = None, overall_task: Optional[Any] = None) -> str:
    """
    Step 1: Mean Reversion Optimization
    
    Returns:
        Path to the merged MR results CSV.
    """
    logger.debug("=" * 60)
    logger.debug("STEP 1: Mean Reversion Optimization")
    logger.debug("=" * 60)
    
    out_mr_wfo_csv = f"results/{config.run_id}_wfo_mr.csv"
    
    if config.wfo_mode and config.exhaustive_mode:
        # WFO + Exhaustive: Run all combos with WFO
        combos = get_combos(config)
        
        # Estimate total windows (approx: 4 years / step_days)
        approx_windows_per_combo = max(1, (1460 - config.window_days) // config.step_days)
        total_windows = len(combos) * approx_windows_per_combo
        
        logger.debug(f"Mode: WFO + Exhaustive ({len(combos)} combos × ~{approx_windows_per_combo} windows = ~{total_windows} total)")
        
        # Prepare arguments with progress queue
        from multiprocessing import Manager
        import threading
        
        manager = Manager()
        try:
            progress_queue = manager.Queue()
            mr_wfo_task = progress.add_task(f"[Step 1] MR WFO ({len(combos)} combos)", total=total_windows, transient=False)
            
            # Stage 1 is 1/6th of 100%
            stage_size = 100.0 / 6.0
            trials_per_window = config.mr_trials
            total_trials = total_windows * trials_per_window
            
            def progress_listener():
                nonlocal total_windows, total_trials, mr_wfo_task
                
                last_overall_val = 0.0
                windows_completed = 0
                combos_completed = 0
                
                # We need to track current window per combo to calculate fractional progress
                # However, for the AGGREGATED bar, we can just treat it as one giant WFO run of 
                # (total_windows_per_combo * len(combos)) windows.
                
                # Track window status per combo
                combo_windows = {} # combo_name -> current_window
                combo_trials = {}  # combo_name -> current_trial
                
                while True:
                    try:
                        msg = progress_queue.get(timeout=1.0)
                    except: # Timeout or queue closed
                        if getattr(threading.current_thread(), "stop_requested", False): break
                        continue
                        
                    if msg == "STOP": break
                    
                    etype = msg[0]
                    combo_name = msg[1]
                    
                    if etype == "total_windows":
                        # Actual windows for ONE combo
                        actual_win_per_combo = msg[2]
                        total_windows = actual_win_per_combo * len(combos)
                        total_trials = total_windows * trials_per_window
                        progress.update(mr_wfo_task, total=total_windows)
                        
                    elif etype == "window":
                        win_idx = msg[2]
                        combo_windows[combo_name] = win_idx
                        combo_trials[combo_name] = 0 # Reset trial on new window
                        
                    elif etype == "trial":
                        # msg[2] is trial_data dict
                        trial_data = msg[2]
                        if isinstance(trial_data, dict):
                            t_rel = trial_data.get("trial_count")
                            t_abs = trial_data.get("trial_idx", 0)
                            trial_val = t_rel if t_rel is not None else t_abs
                        else:
                            trial_val = trial_data
                            
                        combo_trials[combo_name] = max(combo_trials.get(combo_name, 0), trial_val)
                        
                        # Calculate AGGREGATED fractional progress
                        total_abs_completed = 0.0
                        for c in range(len(combos)):
                            cname = f"{combos[c]['trend_kind']}-{combos[c]['sizing_mode']}-{combos[c]['long_str']}"
                            win = combo_windows.get(cname, 0)
                            trial = combo_trials.get(cname, 0)
                            if win > 0:
                                trial_prog = min(1.0, trial / max(1, trials_per_window))
                                total_abs_completed += (win - 1) + trial_prog
                        
                        # Ensure we never go backwards
                        current_val = progress._tasks[mr_wfo_task].completed
                        new_val = min(total_windows, max(current_val, total_abs_completed))
                        progress.update(mr_wfo_task, completed=new_val)
                        
                        if overall_task is not None:
                            stage_progress = max(0.0, min(1.0, new_val / max(1, total_windows)))
                            delta = (stage_progress * stage_size) - last_overall_val
                            if delta > 0:
                                progress.update(overall_task, advance=delta)
                                last_overall_val += delta
                                
                    elif etype == "combo":
                        combos_completed += 1
                        c_name, win_count, elapsed, success = msg[1], msg[2], msg[3], msg[4]
                        status = "✅" if success else "❌"
                        progress.console.print(f"  {status} Finished: {c_name} ({elapsed:.0f}s, {win_count} windows)")
                    
                    elif etype == "log":
                        # Log line from MR subprocess -> write to main logger
                        log_line = msg[2]
                        logger.debug(f"[{combo_name}] {log_line}")

            listener_thread = threading.Thread(target=progress_listener, daemon=True)
            listener_thread.start()

            combo_args = [(config, combo, i, len(combos), total_windows, progress_queue) for i, combo in enumerate(combos)]
            
            results = []
            
            # Run parallel jobs with Queue-based progress tracking
            logger.info(f"Starting {len(combos)} parallel optimization jobs...")
            
            from concurrent.futures import ProcessPoolExecutor, as_completed
            
            # Submit all jobs to executor
            with ProcessPoolExecutor(max_workers=config.max_workers) as executor:
                futures = [executor.submit(run_mr_wfo_combo, arg) for arg in combo_args]
                
                # Wait for all to complete
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Combo execution error: {e}")
            
            progress_queue.put("STOP")
            listener_thread.join(timeout=5.0)
        finally:
            # Shutdown manager safely
            manager.shutdown()
            # Finish the task at 100%
            progress.update(mr_wfo_task, completed=total_windows)
            # DO NOT remove_task(mr_wfo_task) - user wants history
        
        # Ensure overall task is fully caught up for this stage
        if overall_task is not None:
            mark_stage_done(progress, overall_task, 1) # Step 1 is done
        
        # Check for failures
        failed = [r for r in results if not r["success"]]
        if failed:
            logger.warning(f"{len(failed)} combos failed:")
            for f in failed:
                logger.warning(f"  - {f['combo']}: {f['error']}")
        
        # Merge results
        logger.info("Merging MR WFO results...")
        merge_wfo_results(config, "mr", out_mr_wfo_csv)
        
    elif config.wfo_mode:
        # WFO only (no exhaustive)
        logger.info("Mode: WFO (Optuna exploration)")
        approx_windows = max(1, (1460 - config.window_days) // config.step_days)
        args = [
            "-m", "tools.optimizer.cli",
            "--strategy", "mr",
            "--mode", "wfo",
            "--tag", config.tag,
            "--data", config.price_data,
            "--funding-data", config.funding_data,
            "--window-days", str(config.window_days),
            "--step-days", str(config.step_days),
            "--trials", str(config.mr_trials),
            "--jobs", str(config.max_workers),
            "--storage", config.storage,
            "--study-name", f"mr_{config.tag}_wfo",
            "--out", out_mr_wfo_csv,
        ]
        run_optimizer(["python3"] + args, "[Step 1] MR WFO", logger, progress, overall_task, approx_windows=approx_windows, verbose=config.verbose)
    
    else:
        # Static mode
        logger.info("Mode: Static (train/test split)")
        args = [
            "-m", "tools.optimizer.cli",
            "--strategy", "mr",
            "--mode", "static",
            "--tag", config.tag,
            "--data", config.price_data,
            "--funding-data", config.funding_data,
            "--train-start", config.train_start,
            "--train-end", config.train_end,
            "--test-start", config.test_start,
            "--test-end", config.test_end,
            "--trials", str(config.mr_trials),
            "--jobs", str(config.max_workers),
            "--storage", config.storage,
            "--study-name", f"mr_{config.tag}_static",
            "--out", f"results/{config.run_id}_opt_mr.csv",
        ]
        run_optimizer(["python3"] + args, "[Step 1] MR Static", logger, progress, overall_task, verbose=config.verbose)
        return f"results/{config.run_id}_opt_mr.csv"
    
    return out_mr_wfo_csv


def merge_wfo_results(config: OptimizationConfig, strategy: str, out_file: str):
    """Merge multiple WFO result files into one."""
    import pandas as pd
    
    combos = get_combos(config)
    dfs = []
    
    for combo in combos:
        file = f"results/{config.run_id}_wfo_mr_{combo['trend_kind']}_{combo['sizing_mode']}_{combo['long_str']}.csv"
        if os.path.exists(file):
            df = pd.read_csv(file)
            df["combo"] = f"{combo['trend_kind']}_{combo['sizing_mode']}_{combo['long_str']}"
            dfs.append(df)
    
    if dfs:
        merged = pd.concat(dfs, ignore_index=True)
        merged.to_csv(out_file, index=False)


def step2_mr_selection(config: OptimizationConfig, mr_csv: str, logger: logging.Logger, progress: Optional[Any] = None, overall_task: Optional[Any] = None) -> str:
    """
    Step 2: Select best MR parameters.
    
    Returns:
        Path to the best MR params JSON.
    """
    logger.debug("=" * 60)
    logger.debug("STEP 2: MR Parameter Selection")
    logger.debug("=" * 60)
    
    out_params = f"configs/{config.run_id}_best_mr.json"
    
    if config.wfo_mode:
        args = [
            "--wfo-csv", mr_csv,
            "--out", out_params,
            "--strategy", config.selection_strategy,
        ]
        if config.selection_strategy in ["ensemble", "stable_ensemble"]:
            args.extend(["--ensemble-n", str(config.ensemble_n)])
        if config.selection_strategy in ["stable", "stable_ensemble"]:
            args.extend(["--stability-lambda", str(config.stability_lambda)])
        run_optimizer(["python3", "tools/wfo_select_best.py"] + args, "[Step 2] MR Selection (WFO)", logger, progress, overall_task, verbose=config.verbose)
    else:
        args = [
            "--runs", mr_csv,
            "--emit-config", out_params,
            "--family-index", "0",
            "--min-occurs", "1",
        ]
        run_optimizer(["python3", "tools/wf_pick.py"] + args, "[Step 2] MR Selection (Static)", logger, progress, overall_task, verbose=config.verbose)
    
    # Wrap into full config
    wrap_mr_config(config, out_params)
    
    return out_params


def wrap_mr_config(config: OptimizationConfig, params_file: str):
    """Wrap MR params into a full config file."""
    with open(params_file) as f:
        params = json.load(f)
    
    # Remove metadata
    for k in ["_generated_by", "_generated_at", "_family"]:
        params.pop(k, None)
    
    full_config = {
        "fees": {
            "maker_fee": 0.0002,
            "taker_fee": 0.0004,
            "slippage_bps": 1.0,
            "bnb_discount": 0.25,
            "pay_fees_in_bnb": True
        },
        "strategy": params,
        "execution": {"interval": "15m", "poll_sec": 5},
        "risk": {
            "basis_btc": 1.0,
            "risk_mode": "fixed_basis",
            "drawdown_reset_days": 7.0,
            "drawdown_reset_score": 30.0
        }
    }
    full_config["strategy"]["strategy_type"] = "mean_reversion"
    full_config["strategy"]["bar_interval_minutes"] = 15
    
    out_file = f"configs/{config.run_id}_final_mr.json"
    with open(out_file, "w") as f:
        json.dump(full_config, f, indent=2)


def step3_trend_optimization(config: OptimizationConfig, logger: logging.Logger, progress: Optional[Any] = None, overall_task: Optional[Any] = None) -> str:
    """
    Step 3: Trend Strategy Optimization.
    
    Returns:
        Path to trend results CSV.
    """
    logger.debug("=" * 60)
    logger.debug("STEP 3: Trend Optimization")
    logger.debug("=" * 60)
    
    if config.wfo_mode:
        out_file = f"results/{config.run_id}_wfo_trend.csv"
        args = [
            "-m", "tools.optimizer.cli",
            "--strategy", "trend",
            "--mode", "wfo",
            "--tag", config.tag,
            "--data", config.price_data,
            "--funding-data", config.funding_data,
            "--window-days", str(config.window_days),
            "--step-days", str(config.step_days),
            "--trials", str(config.trend_trials),
            "--jobs", str(config.max_workers),
            *([] if config.long_only_futures else ["--allow-shorts"]),
            "--storage", config.storage,
            "--study-name", f"trend_{config.tag}_wfo",
            "--out", out_file,
        ]
        
        # Estimate total windows (approx: 4 years / step_days)
        approx_windows = max(1, (1460 - config.window_days) // config.step_days)
        run_optimizer(["python3"] + args, "[Step 3] Trend WFO", logger, progress, overall_task, approx_windows=approx_windows, verbose=config.verbose)
    else:
        out_file = f"results/{config.run_id}_opt_trend.csv"
        args = [
            "-m", "tools.optimizer.cli",
            "--strategy", "trend",
            "--mode", "static",
            "--tag", config.tag,
            "--data", config.price_data,
            "--funding-data", config.funding_data,
            "--train-start", config.train_start,
            "--train-end", config.train_end,
            "--test-start", config.test_start,
            "--test-end", config.test_end,
            "--trials", str(config.trend_trials),
            "--jobs", str(config.max_workers),
            *([] if config.long_only_futures else ["--allow-shorts"]),
            "--storage", config.storage,
            "--study-name", f"trend_{config.tag}_static",
            "--out", out_file,
        ]
        run_optimizer(["python3"] + args, "[Step 3] Trend Static", logger, progress, overall_task, verbose=config.verbose)
    
    return out_file


def step4_trend_selection(config: OptimizationConfig, trend_csv: str, logger: logging.Logger, progress: Optional[Any] = None, overall_task: Optional[Any] = None) -> str:
    """Step 4: Select best Trend parameters."""
    logger.debug("=" * 60)
    logger.debug("STEP 4: Trend Parameter Selection")
    logger.debug("=" * 60)
    
    out_params = f"configs/{config.run_id}_best_trend_params.json"
    
    if config.wfo_mode:
        args = [
            "--wfo-csv", trend_csv,
            "--out", out_params,
            "--strategy", config.selection_strategy,
        ]
        if config.selection_strategy in ["ensemble", "stable_ensemble"]:
            args.extend(["--ensemble-n", str(config.ensemble_n)])
        if config.selection_strategy in ["stable", "stable_ensemble"]:
            args.extend(["--stability-lambda", str(config.stability_lambda)])
        run_optimizer(["python3", "tools/wfo_select_best.py"] + args, "[Step 4] Trend Selection (WFO)", logger, progress, overall_task, verbose=config.verbose)
    else:
        args = [
            "--runs", trend_csv,
            "--emit-config", out_params,
            "--family-index", "0",
            "--min-occurs", "1",
        ]
        run_optimizer(["python3", "tools/wf_pick.py"] + args, "[Step 4] Trend Selection (Static)", logger, progress, overall_task, verbose=config.verbose)
    
    # Wrap into full config
    wrap_trend_config(config, out_params)
    
    return out_params


def wrap_trend_config(config: OptimizationConfig, params_file: str):
    """Wrap Trend params into a full config file."""
    with open(params_file) as f:
        params = json.load(f)
    
    for k in ["_generated_by", "_generated_at", "_family"]:
        params.pop(k, None)
    
    full_config = {
        "fees": {
            "maker_fee": 0.0002,
            "taker_fee": 0.0004,
            "slippage_bps": 1.0,
            "bnb_discount": 0.25,
            "pay_fees_in_bnb": True
        },
        "strategy": params,
        "execution": {"interval": "15m", "poll_sec": 5},
        "risk": {
            "basis_btc": 1.0,
            "risk_mode": "fixed_basis",
            "drawdown_reset_days": 7.0,
            "drawdown_reset_score": 30.0
        }
    }
    full_config["strategy"]["strategy_type"] = "trend"
    full_config["strategy"].setdefault("long_only", True)
    full_config["strategy"].setdefault("step_allocation", 1.0)
    full_config["strategy"].setdefault("max_position", 1.0)
    full_config["strategy"].setdefault("rebalance_threshold_w", 0.03)
    
    out_file = f"configs/{config.run_id}_final_trend.json"
    with open(out_file, "w") as f:
        json.dump(full_config, f, indent=2)


def step5_meta_optimization(config: OptimizationConfig, logger: logging.Logger, progress: Optional[Any] = None, overall_task: Optional[Any] = None) -> str:
    """Step 5: Meta threshold optimization."""
    logger.debug("=" * 60)
    logger.debug("STEP 5: Meta Optimization")
    logger.debug("=" * 60)
    
    out_file = f"results/{config.run_id}_opt_meta.csv"
    args = [
        "-m", "tools.optimizer.cli",
        "--strategy", "meta",
        "--mode", "static",
        "--tag", config.tag,
        "--data", config.price_data,
        "--funding-data", config.funding_data,
        "--mr-config", f"configs/{config.run_id}_final_mr.json",
        "--trend-config", f"configs/{config.run_id}_final_trend.json",
        "--trials", str(config.meta_trials),
        "--jobs", "1",  # Sequential for better log visibility in Step 5
        "--storage", config.storage,
        "--study-name", f"meta_{config.tag}_static",
        "--out", out_file,
    ]
    run_optimizer(["python3"] + args, "[Step 5] Meta Optimization", logger, progress, overall_task, verbose=config.verbose)
    return out_file


def step6_assemble_config(config: OptimizationConfig, meta_csv: str, logger: logging.Logger, progress: Optional[Any] = None, overall_task: Optional[Any] = None) -> str:
    """Step 6: Assemble final config."""
    logger.debug("=" * 60)
    logger.debug("STEP 6: Assemble Final Config")
    logger.debug("=" * 60)
    
    out_file = f"configs/{config.run_id}.json"
    args = [
        "--mr", f"configs/{config.run_id}_final_mr.json",
        "--trend", f"configs/{config.run_id}_final_trend.json",
        "--meta-results", meta_csv,
        "--out", out_file,
    ]
    run_optimizer(["python3", "tools/assemble_v2_config.py"] + args, "[Step 6] Config Assembly", logger, progress, overall_task, verbose=config.verbose)
    return out_file


def step7_wfo_audit(config: OptimizationConfig, logger: logging.Logger, progress: Optional[Any] = None, overall_task: Optional[Any] = None):
    """Step 7: WFO Audit (if applicable)."""
    if not config.wfo_mode:
        return
    
    logger.info("=" * 60)
    logger.info("STEP 7: Walk-Forward Audit")
    logger.info("=" * 60)
    
    # Analyze MR
    mr_wfo = f"results/{config.run_id}_wfo_mr.csv"
    if os.path.exists(mr_wfo):
        args = ["--wfo-csv", mr_wfo, "--out", f"results/{config.run_id}_analysis_mr.json"]
        run_optimizer(["python3", "tools/wfo_analyzer.py"] + args, "MR WFO Analysis", logger, progress, overall_task, verbose=config.verbose)
    
    # Analyze Trend
    trend_wfo = f"results/{config.run_id}_wfo_trend.csv"
    if os.path.exists(trend_wfo):
        args = ["--wfo-csv", trend_wfo, "--out", f"results/{config.run_id}_analysis_trend.json"]
        run_optimizer(["python3", "tools/wfo_analyzer.py"] + args, "Trend WFO Analysis", logger, progress, overall_task, verbose=config.verbose)


# ============================================================
# UTILS
# ============================================================
def get_start_info(config: OptimizationConfig) -> str:
    """Get detailed status info for the startup panel."""
    info = []
    
    # 1. Mode Info
    mode_str = "WFO" if config.wfo_mode else "Static"
    if config.exhaustive_mode: mode_str += " + Exhaustive"
    info.append(f"[bold]Mode:[/bold] {mode_str}")
    
    # 2. Optuna DB Info
    try:
        import optuna
        # Check actual file exists first to avoid creating empty DB
        db_path = config.storage.replace("sqlite:///", "")
        if os.path.exists(db_path):
            loaded_studies = optuna.get_all_study_summaries(storage=config.storage)
            relevant = [s for s in loaded_studies if config.tag in s.study_name]
            
            if relevant:
                total_trials = sum(s.n_trials for s in relevant)
                info.append(f"[bold]Optuna DB:[/bold] Found {len(relevant)} studies for tag '{config.tag}'")
                info.append(f"[bold]Total History:[/bold] {total_trials} trials recorded")
            else:
                info.append(f"[bold]Optuna DB:[/bold] No existing studies found for tag '{config.tag}'")
        else:
             info.append(f"[bold]Optuna DB:[/bold] New database will be created")
    except Exception as e:
        info.append(f"[dim]Optuna info unavailable: {e}[/dim]")

    # 3. Data Info
    info.append(f"[bold]Data:[/bold] {os.path.basename(config.price_data)}")
    
    return "\n".join(info)


# ============================================================
# MAIN ENTRY POINT
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Optimization Orchestrator - Python replacement for run_complete_optimization.sh",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 tools/run_optimization.py --wfo --exhaustive
  python3 tools/run_optimization.py --wfo --long-only-futures
  python3 tools/run_optimization.py --tag ETHBTC --price-data data/raw/ETHBTC_15m.csv
        """
    )
    
    # Data options
    parser.add_argument("--price-data", default="data/raw/BTCUSDT_15m_2021-2025_vision.csv")
    parser.add_argument("--funding-data", default="data/raw/BTCUSDT_funding_2021-2025.csv")
    parser.add_argument("--tag", default="BTC")
    
    # Mode options
    parser.add_argument("--wfo", action="store_true", help="Enable Walk-Forward Optimization")
    parser.add_argument("--exhaustive", action="store_true", help="Test all parameter combinations")
    parser.add_argument("--long-only-futures", action="store_true", help="Only test long-only strategies")
    parser.add_argument("--futures-only", action="store_true", help="Only test shorting strategies")
    
    # WFO settings
    parser.add_argument("--window-days", type=int, default=180)
    parser.add_argument("--step-days", type=int, default=30)
    parser.add_argument("--selection-strategy", default="weighted",
                        choices=["weighted", "ensemble", "stable", "stable_ensemble", "best_oos", "consistent", "recent"],
                        help="WFO parameter selection strategy (default: weighted)")
    parser.add_argument("--ensemble-n", type=int, default=5,
                        help="Number of windows to average for ensemble strategy (default: 5)")
    parser.add_argument("--stability-lambda", type=float, default=0.1,
                        help="Stability penalty weight for 'stable' strategy (default: 0.1)")
    
    # Date settings (for static mode)
    parser.add_argument("--train-start", default="2021-01-01")
    parser.add_argument("--train-end", default="2024-06-30")
    parser.add_argument("--test-start", default="2024-07-01")
    parser.add_argument("--test-end", default="2025-06-01")
    
    # Trial counts
    parser.add_argument("--mr-trials", type=int, default=50)
    parser.add_argument("--trend-trials", type=int, default=30)
    parser.add_argument("--meta-trials", type=int, default=8)
    
    # Other
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--verbose", "-v", action="store_true")
    
    args = parser.parse_args()
    
    # Auto-enable exhaustive for futures modes (like shell script)
    if args.long_only_futures or args.futures_only:
        args.exhaustive = True
    
    # Create run_id
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{args.tag}_{run_timestamp}"

    # Setup logging first with run_id
    logger = setup_logging(run_id, args.verbose)

    # Build config with run_id
    config = OptimizationConfig(
        run_id=run_id,
        price_data=args.price_data,
        funding_data=args.funding_data,
        tag=args.tag,
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
        wfo_mode=args.wfo,
        exhaustive_mode=args.exhaustive,
        long_only_futures=args.long_only_futures,
        futures_only=args.futures_only,
        window_days=args.window_days,
        step_days=args.step_days,
        selection_strategy=args.selection_strategy,
        ensemble_n=args.ensemble_n,
        stability_lambda=args.stability_lambda,
        mr_trials=args.mr_trials,
        trend_trials=args.trend_trials,
        meta_trials=args.meta_trials,
        max_workers=args.max_workers,
        verbose=args.verbose,
    )

    # Signal handler for Ctrl+C
    def signal_handler(sig, frame):
        logger.error("Interrupt received! Cleaning up...")
        sys.exit(1)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Setup
    os.makedirs("results", exist_ok=True)
    os.makedirs("configs", exist_ok=True)

    console = Console() if RICH_AVAILABLE else None
    
    start_time = time.time()
    
    # Print header
    if RICH_AVAILABLE:
        start_info = get_start_info(config)
        console.print(Panel.fit(
            f"[bold blue]Optimization Workflow: {config.tag}[/bold blue]\n\n" + start_info,
            title="🚀 Starting Optimization",
            border_style="blue"
        ))
    else:
        logger.info(f"Starting optimization for {config.tag}")
        logger.info(f"Mode: {'WFO' if config.wfo_mode else 'Static'}{' + Exhaustive' if config.exhaustive_mode else ''}")
    
    try:
        # Run Steps
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn, TimeElapsedColumn, MofNCompleteColumn
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold green]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            # Scale to 100 for better fractional granularity visibility
            overall_task = progress.add_task("Overall Optimization", total=100)
            
            mr_csv = step1_mr_optimization(config, logger, progress, overall_task)
            step2_mr_selection(config, mr_csv, logger, progress, overall_task)
            mark_stage_done(progress, overall_task, 2)
            
            trend_csv = step3_trend_optimization(config, logger, progress, overall_task)
            mark_stage_done(progress, overall_task, 3)
            
            step4_trend_selection(config, trend_csv, logger, progress, overall_task)
            mark_stage_done(progress, overall_task, 4)
            
            meta_csv = step5_meta_optimization(config, logger, progress, overall_task)
            mark_stage_done(progress, overall_task, 5)
            
            final_config = step6_assemble_config(config, meta_csv, logger, progress, overall_task)
            # Finish the bar.
            mark_stage_done(progress, overall_task, 6)
            step7_wfo_audit(config, logger, progress, overall_task)
        
        # Final summary
        elapsed = time.time() - start_time
        elapsed_mins = int(elapsed // 60)
        elapsed_secs = int(elapsed % 60)
        
        if RICH_AVAILABLE:
            table = Table(title="Optimization Complete")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")
            table.add_row("Tag", config.tag)
            table.add_row("Total Time", f"{elapsed_mins}m {elapsed_secs}s")
            table.add_row("Final Config", final_config)
            if config.wfo_mode:
                table.add_row("MR WFO Analysis", f"results/{config.run_id}_analysis_mr.json")
                table.add_row("Trend WFO Analysis", f"results/{config.run_id}_analysis_trend.json")
            console.print(table)
        else:
            logger.info("=" * 60)
            logger.info("OPTIMIZATION COMPLETE")
            logger.info("=" * 60)
            logger.info(f"  Total Time: {elapsed_mins}m {elapsed_secs}s")
            logger.info(f"  Final Config: {final_config}")
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise


if __name__ == "__main__":
    main()
