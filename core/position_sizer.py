"""
Dynamic Position Sizing Module

Provides adaptive position sizing based on market volatility.
Supports multiple approaches:
- Static: Fixed step allocation
- Volatility: Target volatility approach
- Kelly: Kelly Criterion with optional dynamic stats
"""
from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple
from collections import deque
import logging
import numpy as np

log = logging.getLogger(__name__)


class RollingTradeStats:
    """
    Tracks rolling trade statistics for dynamic Kelly Criterion.
    
    Maintains a sliding window of trade P&L to calculate:
    - Win rate (percentage of profitable trades)
    - Average win size
    - Average loss size
    
    These are used to dynamically adjust Kelly position sizing.
    """
    
    def __init__(self, lookback: int = 100, min_trades: int = 20):
        """
        Initialize rolling stats tracker.
        
        Args:
            lookback: Maximum number of trades to keep in history
            min_trades: Minimum trades required before stats are considered valid
        """
        self.lookback = lookback
        self.min_trades = min_trades
        self.trades: deque = deque(maxlen=lookback)
    
    def add_trade(self, pnl: float) -> None:
        """Record a completed trade's P&L."""
        self.trades.append(pnl)
    
    def get_stats(self) -> Optional[Tuple[float, float, float]]:
        """
        Calculate rolling statistics from trade history.
        
        Returns:
            Tuple of (win_rate, avg_win, avg_loss) or None if insufficient data.
            - win_rate: Fraction of trades with positive P&L (0.0 to 1.0)
            - avg_win: Average P&L of winning trades (positive)
            - avg_loss: Average P&L of losing trades (positive, absolute value)
        """
        if len(self.trades) < self.min_trades:
            return None
        
        trades = np.array(self.trades)
        wins = trades[trades > 0]
        losses = trades[trades < 0]
        
        # Calculate win rate
        win_rate = len(wins) / len(trades) if len(trades) > 0 else 0.5
        
        # Calculate average win (default to small value if no wins)
        avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.01
        
        # Calculate average loss (absolute value, default to small value if no losses)
        avg_loss = float(np.abs(np.mean(losses))) if len(losses) > 0 else 0.01
        
        return (win_rate, avg_win, avg_loss)
    
    def has_sufficient_data(self) -> bool:
        """Check if enough trades exist for valid statistics."""
        return len(self.trades) >= self.min_trades
    
    def __len__(self) -> int:
        return len(self.trades)


@dataclass
class PositionSizerConfig:
    """Configuration for position sizing strategy."""
    
    mode: Literal["static", "volatility", "kelly"] = "static"
    """Position sizing mode: static (default), volatility targeting, or Kelly Criterion"""
    
    base_step: float = 0.5
    """Base step allocation (0.0 to 1.0). Acts as default for static mode."""
    
    target_vol: float = 0.5
    """Target annualized volatility for volatility targeting mode."""
    
    min_step: float = 0.1
    """Minimum position step (floor)."""
    
    max_step: float = 1.0
    """Maximum position step (ceiling)."""
    
    # Kelly Criterion parameters - Static defaults (used when dynamic not available)
    kelly_win_rate: float = 0.55
    """Historical win rate for Kelly Criterion (0.0 to 1.0)."""
    
    kelly_avg_win: float = 0.02
    """Average win size for Kelly Criterion."""
    
    kelly_avg_loss: float = 0.01
    """Average loss size for Kelly Criterion."""
    
    # Dynamic Kelly parameters
    kelly_lookback: int = 100
    """Number of recent trades to consider for dynamic Kelly stats."""
    
    kelly_min_trades: int = 20
    """Minimum trades required before using dynamic Kelly (fallback to static)."""
    
    kelly_fraction: float = 0.5
    """Fraction of full Kelly to use (0.5 = Half-Kelly, safer)."""
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if not 0.0 <= self.base_step <= 1.0:
            raise ValueError(f"base_step must be in [0, 1], got {self.base_step}")
        if not 0.0 <= self.min_step <= 1.0:
            raise ValueError(f"min_step must be in [0, 1], got {self.min_step}")
        if not 0.0 <= self.max_step <= 1.0:
            raise ValueError(f"max_step must be in [0, 1], got {self.max_step}")
        if self.min_step > self.max_step:
            raise ValueError(f"min_step ({self.min_step}) > max_step ({self.max_step})")
        if self.target_vol <= 0:
            raise ValueError(f"target_vol must be positive, got {self.target_vol}")
        if not 0.0 <= self.kelly_win_rate <= 1.0:
            raise ValueError(f"kelly_win_rate must be in [0, 1], got {self.kelly_win_rate}")


class PositionSizer:
    """
    Dynamic position sizing with volatility targeting and Kelly Criterion.
    
    Supports three modes:
    - static: Fixed step allocation
    - volatility: Inverse volatility scaling
    - kelly: Kelly Criterion with optional dynamic stats from trade history
    
    Examples:
        >>> config = PositionSizerConfig(mode="volatility", base_step=0.5, target_vol=0.5)
        >>> sizer = PositionSizer(config)
        >>> sizer.calculate_step(realized_vol=0.25)  # Low volatility
        1.0  # Increased to max_step
        >>> sizer.calculate_step(realized_vol=1.0)   # High volatility
        0.25  # Decreased
        
        # Dynamic Kelly example
        >>> config = PositionSizerConfig(mode="kelly")
        >>> sizer = PositionSizer(config)
        >>> sizer.record_trade(0.05)  # Record winning trade
        >>> sizer.record_trade(-0.02)  # Record losing trade
        >>> # After enough trades, Kelly params will be dynamically calculated
    """
    
    def __init__(self, config: PositionSizerConfig, trade_stats: Optional[RollingTradeStats] = None):
        """
        Initialize position sizer.
        
        Args:
            config: Position sizer configuration
            trade_stats: Optional external RollingTradeStats for shared state across sessions.
                         If None, creates internal stats tracker (resets on restart).
        """
        self.config = config
        
        # Create or use provided trade stats tracker for dynamic Kelly
        if trade_stats is not None:
            self.trade_stats = trade_stats
        else:
            self.trade_stats = RollingTradeStats(
                lookback=config.kelly_lookback,
                min_trades=config.kelly_min_trades
            )
        
        log.debug(
            "[PositionSizer] Initialized with mode=%s, base_step=%.2f, target_vol=%.2f",
            config.mode, config.base_step, config.target_vol
        )
    
    def record_trade(self, pnl: float) -> None:
        """
        Record a completed trade's P&L for dynamic Kelly calculation.
        
        Args:
            pnl: Trade profit/loss as a fraction (e.g., 0.02 for 2% gain)
        """
        self.trade_stats.add_trade(pnl)
        log.debug(
            "[PositionSizer] Recorded trade pnl=%.4f, total_trades=%d",
            pnl, len(self.trade_stats)
        )
    
    def calculate_step(self, realized_vol: float) -> float:
        """
        Calculate position size step based on configuration and market volatility.
        
        Args:
            realized_vol: Current realized volatility (annualized)
        
        Returns:
            Position step size in [min_step, max_step]
        
        Examples:
            >>> config = PositionSizerConfig(mode="static", base_step=0.5)
            >>> sizer = PositionSizer(config)
            >>> sizer.calculate_step(0.5)
            0.5
            
            >>> config = PositionSizerConfig(
            ...     mode="volatility", 
            ...     base_step=0.5, 
            ...     target_vol=0.5,
            ...     min_step=0.1,
            ...     max_step=1.0
            ... )
            >>> sizer = PositionSizer(config)
            >>> sizer.calculate_step(0.5)  # Normal volatility
            0.5
            >>> sizer.calculate_step(1.0)  # High volatility
            0.25
        """
        if self.config.mode == "static":
            return self.config.base_step
        
        elif self.config.mode == "volatility":
            return self._volatility_targeting(realized_vol)
        
        elif self.config.mode == "kelly":
            return self._kelly_criterion(realized_vol)
        
        else:
            log.warning(
                "[PositionSizer] Unknown mode '%s', falling back to static",
                self.config.mode
            )
            return self.config.base_step
    
    def _volatility_targeting(self, realized_vol: float) -> float:
        """
        Volatility targeting approach.
        
        Formula:
            step = base_step * (target_vol / realized_vol)
            step_clamped = clamp(step, min_step, max_step)
        
        Args:
            realized_vol: Current realized volatility (annualized)
        
        Returns:
            Position step size in [min_step, max_step]
        """
        # Handle edge cases
        if realized_vol <= 0 or not realized_vol == realized_vol:  # NaN check
            log.warning(
                "[PositionSizer] Invalid realized_vol=%.4f, using base_step",
                realized_vol
            )
            return self.config.base_step
        
        # Calculate scaled step
        step_raw = self.config.base_step * (self.config.target_vol / realized_vol)
        
        # Clamp to bounds
        step_clamped = max(self.config.min_step, min(self.config.max_step, step_raw))
        
        log.debug(
            "[PositionSizer] Volatility targeting: rv=%.4f, target=%.4f, "
            "raw_step=%.4f, clamped=%.4f",
            realized_vol, self.config.target_vol, step_raw, step_clamped
        )
        
        return step_clamped
    
    def _kelly_criterion(self, realized_vol: float) -> float:
        """
        Kelly Criterion approach with dynamic statistics.
        
        Uses rolling trade history to calculate win_rate and average P&L when
        sufficient data is available. Falls back to static config parameters
        when trade history is insufficient.
        
        Formula:
            kelly_fraction = (win_rate * avg_win - (1-win_rate) * avg_loss) / avg_win
            kelly_adjusted = kelly_fraction * kelly_fraction_config  # e.g., 0.5 for Half-Kelly
            step = kelly_adjusted * volatility_adjustment
        
        Args:
            realized_vol: Current realized volatility (annualized)
        
        Returns:
            Position step size in [min_step, max_step]
        """
        # Try to get dynamic stats from trade history
        dynamic_stats = self.trade_stats.get_stats()
        
        if dynamic_stats is not None:
            w, avg_w, avg_l = dynamic_stats
            log.debug(
                "[PositionSizer] Using DYNAMIC Kelly: win_rate=%.2f, avg_win=%.4f, avg_loss=%.4f (n=%d trades)",
                w, avg_w, avg_l, len(self.trade_stats)
            )
        else:
            # Fallback to static config
            w = self.config.kelly_win_rate
            avg_w = self.config.kelly_avg_win
            avg_l = self.config.kelly_avg_loss
            log.debug(
                "[PositionSizer] Using STATIC Kelly: win_rate=%.2f, avg_win=%.4f, avg_loss=%.4f (need %d more trades for dynamic)",
                w, avg_w, avg_l, self.config.kelly_min_trades - len(self.trade_stats)
            )
        
        # Kelly formula: (w*W - (1-w)*L) / W
        if avg_w <= 0:
            log.warning("[PositionSizer] Invalid avg_win=%.4f, using base_step", avg_w)
            return self.config.base_step
        
        kelly_full = (w * avg_w - (1 - w) * avg_l) / avg_w
        
        # Apply fractional Kelly (configurable, default 0.5 for Half-Kelly)
        kelly_adjusted = kelly_full * self.config.kelly_fraction
        
        # Sanity check: Kelly can go negative if expected value is negative
        if kelly_adjusted < 0:
            log.warning(
                "[PositionSizer] Negative Kelly=%.4f (w=%.2f, avg_w=%.4f, avg_l=%.4f). Using min_step.",
                kelly_adjusted, w, avg_w, avg_l
            )
            return self.config.min_step
        
        # Adjust for volatility (scale down in high vol environments)
        vol_adj = self._volatility_targeting(realized_vol)
        
        # Combine Kelly with volatility adjustment
        step = kelly_adjusted * vol_adj
        
        # Clamp to bounds
        step_clamped = max(self.config.min_step, min(self.config.max_step, step))
        
        log.debug(
            "[PositionSizer] Kelly: full=%.4f, adj=%.4f, vol_scale=%.4f, final_step=%.4f",
            kelly_full, kelly_adjusted, vol_adj, step_clamped
        )
        
        return step_clamped
