"""
Progress Tracking Module

Provides unified progress tracking for multi-process optimization.
Uses multiprocessing.Manager().Queue() for cross-process communication.

Author: AI Audit System
Date: 2024-12-30
"""

from dataclasses import dataclass
from typing import Any, Optional
from multiprocessing import Queue
import time


@dataclass
class ProgressEvent:
    """A progress event that can be sent through the queue."""
    event_type: str  # 'started', 'window', 'combo_done', 'error'
    combo_name: str
    data: Any = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class ProgressTracker:
    """
    Unified progress tracking with Queue for multiprocessing.
    
    Usage in worker process:
        tracker = ProgressTracker(queue, total_windows)
        tracker.report_started("sma-static-long", combo_idx=0)
        for window in windows:
            # ... do work ...
            tracker.report_window("sma-static-long", window_idx)
        tracker.report_combo_done("sma-static-long", elapsed=123.4, success=True)
    
    Usage in main process (reader thread):
        while True:
            event = queue.get(timeout=0.5)
            if event.event_type == 'window':
                update_progress_bar()
    """
    
    def __init__(
        self,
        queue: Optional[Queue] = None,
        total_windows: int = 0,
        total_combos: int = 0,
        use_stdout: bool = False,
    ):
        """
        Initialize progress tracker.
        
        Args:
            queue: Manager().Queue() for sending events (None = no tracking)
            total_windows: Expected total windows across all combos
            total_combos: Expected total number of combos
            use_stdout: If True, print JSON signals to stdout for parent process to catch.
        """
        self.queue = queue
        self.total_windows = total_windows
        self.total_combos = total_combos
        self.use_stdout = use_stdout
        self.trial_count = 0
    
    def _send(self, event_type: str, combo_name: str, data: Any = None):
        """Send an event to the queue or stdout."""
        ts = time.time()
        
        # 1. Send to queue (for intra-process communication)
        if self.queue is not None:
            try:
                self.queue.put((event_type, combo_name, data, ts))
            except Exception:
                pass 
                
        # 2. Print to stdout (for subprocess -> parent communication)
        if self.use_stdout:
            import json
            signal_msg = {
                "signal": "OPTIMIZER_PROGRESS",
                "type": event_type,
                "combo": combo_name,
                "data": data,
                "ts": ts
            }
            print(f"\n{json.dumps(signal_msg)}", flush=True)
    
    def report_started(self, combo_name: str, combo_idx: int):
        """Report that a combo has started processing."""
        self._send('started', combo_name, combo_idx)
    
    def report_window(self, combo_name: str, window_idx: int):
        """Report completion of a single WFO window."""
        self._send('window', combo_name, window_idx)
    
    def report_total_windows(self, combo_name: str, total_windows: int):
        """Report the actual total windows for a combo."""
        self._send('total_windows', combo_name, total_windows)
    
    def report_combo_done(
        self,
        combo_name: str,
        elapsed: float,
        success: bool,
        window_count: int = 0
    ):
        """Report that a combo has finished."""
        self._send('combo_done', combo_name, {
            'elapsed': elapsed,
            'success': success,
            'window_count': window_count,
        })
    
    def report_trial(self, combo_name: str, trial_idx: int, value: float):
        """Report completion of a single Optuna trial."""
        self.trial_count += 1
        self._send('trial', combo_name, {
            'trial_idx': trial_idx,
            'trial_count': self.trial_count,
            'value': value
        })
    
    def report_error(self, combo_name: str, error: str):
        """Report an error during processing."""
        self._send('error', combo_name, error)


def create_progress_queue():
    """
    Factory function to create a Manager().Queue() for progress tracking.
    
    Returns:
        Tuple of (Queue, Manager) - keep manager alive!
    """
    from multiprocessing import Manager
    manager = Manager()
    queue = manager.Queue()
    return queue, manager
