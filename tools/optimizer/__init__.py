"""
Optimizer Framework Package

A standardized framework for strategy optimization with:
- Unified base class for all strategy optimizers
- Consistent progress tracking
- Shared scoring functions
- WFO and static mode support
"""

from .base import BaseOptimizer
from .progress import ProgressTracker
from .scoring import compute_robust_score

__all__ = ["BaseOptimizer", "ProgressTracker", "compute_robust_score"]
