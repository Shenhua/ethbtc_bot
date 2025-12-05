"""
Dynamic Position Sizing Module

Provides adaptive position sizing based on market volatility.
Supports multiple approaches:
- Static: Fixed step allocation
- Volatility: Target volatility approach
- Kelly: Kelly Criterion (future)
"""
from dataclasses import dataclass
from typing import Literal
import logging

log = logging.getLogger(__name__)


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
    
    # Kelly Criterion parameters (for future use)
    kelly_win_rate: float = 0.55
    """Historical win rate for Kelly Criterion (0.0 to 1.0)."""
    
    kelly_avg_win: float = 0.02
    """Average win size for Kelly Criterion."""
    
    kelly_avg_loss: float = 0.01
    """Average loss size for Kelly Criterion."""
    
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
    Dynamic position sizing with volatility targeting.
    
    Examples:
        >>> config = PositionSizerConfig(mode="volatility", base_step=0.5, target_vol=0.5)
        >>> sizer = PositionSizer(config)
        >>> sizer.calculate_step(realized_vol=0.25)  # Low volatility
        1.0  # Increased to max_step
        >>> sizer.calculate_step(realized_vol=1.0)   # High volatility
        0.25  # Decreased
    """
    
    def __init__(self, config: PositionSizerConfig):
        """
        Initialize position sizer.
        
        Args:
            config: Position sizer configuration
        """
        self.config = config
        log.debug(
            "[PositionSizer] Initialized with mode=%s, base_step=%.2f, target_vol=%.2f",
            config.mode, config.base_step, config.target_vol
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
        Kelly Criterion approach (to be implemented).
        
        Formula:
            kelly_fraction = (win_rate * avg_win - (1-win_rate) * avg_loss) / avg_win
            kelly_half = kelly_fraction * 0.5  # Half-Kelly for safety
            step = kelly_half * volatility_adjustment
        
        Args:
            realized_vol: Current realized volatility (annualized)
        
        Returns:
            Position step size in [min_step, max_step]
        
        Note:
            Currently uses half-Kelly with volatility adjustment for safety.
        """
        # Calculate Kelly fraction
        w = self.config.kelly_win_rate
        avg_w = self.config.kelly_avg_win
        avg_l = self.config.kelly_avg_loss
        
        # Kelly formula: (w*W - (1-w)*L) / W
        if avg_w <= 0:
            log.warning("[PositionSizer] Invalid avg_win, using base_step")
            return self.config.base_step
        
        kelly_full = (w * avg_w - (1 - w) * avg_l) / avg_w
        kelly_half = kelly_full * 0.5  # Half-Kelly for safety
        
        # Adjust for volatility
        vol_adj = self._volatility_targeting(realized_vol)
        
        # Combine Kelly with volatility adjustment
        step = kelly_half * vol_adj
        
        # Clamp to bounds
        step_clamped = max(self.config.min_step, min(self.config.max_step, step))
        
        log.debug(
            "[PositionSizer] Kelly: kelly_full=%.4f, kelly_half=%.4f, "
            "vol_adj=%.4f, step=%.4f",
            kelly_full, kelly_half, vol_adj, step_clamped
        )
        
        return step_clamped
