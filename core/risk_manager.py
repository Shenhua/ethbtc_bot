"""
Risk Manager Module

Extracted from live_executor.py to improve modularity and testability.
This module handles all risk state management:
- High Water Mark (HWM) tracking
- Max Drawdown detection
- Daily loss limit tracking
- Phoenix Protocol state

IMPORTANT: This preserves ALL existing behavior from live_executor.py.
No features have been removed or modified - only extracted.
"""
from __future__ import annotations
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
import pandas as pd

log = logging.getLogger("risk_manager")


@dataclass
class RiskConfig:
    """Risk configuration from AppConfig.risk"""
    risk_mode: str = "fixed_basis"
    max_dd_frac: float = 0.0
    max_dd_btc: float = 0.0
    max_daily_loss_frac: float = 0.0
    max_daily_loss_btc: float = 0.0
    drawdown_reset_days: float = 0.0
    drawdown_reset_score: float = 25.0
    
    @classmethod
    def from_config(cls, cfg_risk: Any) -> "RiskConfig":
        """Create RiskConfig from AppConfig.risk object."""
        return cls(
            risk_mode=getattr(cfg_risk, "risk_mode", "fixed_basis"),
            max_dd_frac=float(getattr(cfg_risk, "max_dd_frac", 0.0) or 0.0),
            max_dd_btc=float(getattr(cfg_risk, "max_dd_btc", 0.0) or 0.0),
            max_daily_loss_frac=float(getattr(cfg_risk, "max_daily_loss_frac", 0.0) or 0.0),
            max_daily_loss_btc=float(getattr(cfg_risk, "max_daily_loss_btc", 0.0) or 0.0),
            drawdown_reset_days=float(getattr(cfg_risk, "drawdown_reset_days", 0.0) or 0.0),
            drawdown_reset_score=float(getattr(cfg_risk, "drawdown_reset_score", 25.0) or 25.0),
        )


class RiskManager:
    """
    Manages risk state for live trading.
    
    Tracks:
    - High Water Mark (HWM) for max drawdown calculation
    - Daily loss limits with automatic reset at UTC midnight
    - Max drawdown detection with Phoenix Protocol integration
    
    Example:
        config = RiskConfig.from_config(cfg.risk)
        risk_mgr = RiskManager(config)
        
        # On each bar:
        risk_mgr.update(state, wealth, bar_dt)
        
        if risk_mgr.is_halted(state):
            # Skip trading
    """
    
    def __init__(self, config: RiskConfig):
        """Initialize with risk configuration."""
        self.config = config
    
    def ensure_state(self, state: Dict[str, Any], wealth: float, ts: pd.Timestamp) -> None:
        """
        Ensure all risk state keys exist with proper defaults.
        
        Call this once at startup and after state loading.
        Preserves existing behavior from _ensure_risk_state().
        """
        if "risk_equity_high" not in state:
            state["risk_equity_high"] = wealth
        if "risk_current_date" not in state:
            state["risk_current_date"] = ts.normalize().date().isoformat()
        if "risk_daily_start_wealth" not in state:
            state["risk_daily_start_wealth"] = wealth
        if "risk_daily_limit_hit" not in state:
            state["risk_daily_limit_hit"] = False
        if "risk_maxdd_hit" not in state:
            state["risk_maxdd_hit"] = False
    
    def update(self, state: Dict[str, Any], wealth: float, ts: pd.Timestamp) -> None:
        """
        Update risk state (HWM, daily loss, max DD).
        
        Detects crashes but does NOT handle resets (Phoenix Protocol).
        Phoenix resets are handled in the main execution loop.
        
        Preserves ALL existing behavior from _update_risk_state().
        """
        cfg = self.config
        risk_mode = cfg.risk_mode
        max_dd_frac = cfg.max_dd_frac
        max_daily_loss_frac = cfg.max_daily_loss_frac
        
        # 1. Load Current State
        equity_high = float(state.get("risk_equity_high", wealth))
        
        current_date_str = state.get("risk_current_date")
        if current_date_str:
            try:
                current_date = pd.to_datetime(current_date_str).date()
            except Exception:
                current_date = ts.normalize().date()
        else:
            current_date = ts.normalize().date()
        
        daily_start = float(state.get("risk_daily_start_wealth", wealth))
        daily_limit_hit = bool(state.get("risk_daily_limit_hit", False))
        maxdd_hit = bool(state.get("risk_maxdd_hit", False))
        
        # 2. Update High Water Mark (Only if we aren't already crashed)
        # If we are in MaxDD state, we do NOT update HWM (we are in the penalty box)
        if not maxdd_hit:
            if wealth > equity_high:
                equity_high = wealth
        
        dd_now = equity_high - wealth
        
        # 3. Check for Max Drawdown Violation
        if not maxdd_hit:
            if risk_mode == "dynamic":
                if max_dd_frac > 0.0:
                    threshold_dd = equity_high * max_dd_frac
                else:
                    threshold_dd = cfg.max_dd_btc
            else:
                threshold_dd = cfg.max_dd_btc
            
            if threshold_dd > 0.0 and dd_now >= threshold_dd:
                maxdd_hit = True
                # Record Time of Death for Phoenix Protocol
                state["risk_maxdd_hit_ts"] = ts.isoformat()
        
        # 4. Check for Daily Loss Violation
        cur_date = ts.normalize().date()
        if cur_date != current_date:
            # New Day: Reset Daily Counters
            current_date = cur_date
            daily_start = wealth
            daily_limit_hit = False
        
        daily_pnl = wealth - daily_start
        
        if risk_mode == "dynamic":
            if max_daily_loss_frac > 0.0:
                threshold_loss = daily_start * max_daily_loss_frac
            else:
                threshold_loss = cfg.max_daily_loss_btc
        else:
            threshold_loss = cfg.max_daily_loss_btc
        
        if threshold_loss > 0.0 and daily_pnl <= -threshold_loss:
            daily_limit_hit = True
        
        # 5. Save State
        state["risk_equity_high"] = equity_high
        state["risk_current_date"] = current_date.isoformat()
        state["risk_daily_start_wealth"] = daily_start
        state["risk_daily_limit_hit"] = daily_limit_hit
        state["risk_maxdd_hit"] = maxdd_hit
    
    def is_halted(self, state: Dict[str, Any]) -> bool:
        """Check if trading is halted due to risk limits."""
        return bool(state.get("risk_maxdd_hit", False)) or bool(state.get("risk_daily_limit_hit", False))
    
    def is_maxdd_hit(self, state: Dict[str, Any]) -> bool:
        """Check if max drawdown has been hit."""
        return bool(state.get("risk_maxdd_hit", False))
    
    def is_daily_limit_hit(self, state: Dict[str, Any]) -> bool:
        """Check if daily loss limit has been hit."""
        return bool(state.get("risk_daily_limit_hit", False))
    
    def get_drawdown_pct(self, state: Dict[str, Any], wealth: float) -> float:
        """Get current drawdown as a percentage."""
        equity_high = float(state.get("risk_equity_high", wealth))
        if equity_high <= 0:
            return 0.0
        return (equity_high - wealth) / equity_high
    
    def reset_phoenix(self, state: Dict[str, Any], wealth: float) -> None:
        """
        Reset risk state after Phoenix Protocol activation.
        
        Called when Phoenix conditions are met:
        - Time cooldown passed
        - Regime score above threshold
        """
        state["risk_maxdd_hit"] = False
        state["risk_maxdd_hit_ts"] = None
        state["risk_equity_high"] = wealth  # Reset HWM to current wealth
        state["alert_sent_maxdd"] = False
        log.info("Phoenix reset complete. HWM reset to %.4f", wealth)
    
    def can_phoenix_reset(
        self, 
        state: Dict[str, Any], 
        bar_dt: pd.Timestamp,
        current_regime_score: float
    ) -> bool:
        """
        Check if Phoenix Protocol reset conditions are met.
        
        Conditions:
        1. MaxDD is currently hit
        2. Enough time has passed (drawdown_reset_days)
        3. Regime score is favorable (>= drawdown_reset_score)
        """
        if not state.get("risk_maxdd_hit", False):
            return False
        
        reset_days = self.config.drawdown_reset_days
        reset_score = self.config.drawdown_reset_score
        
        if reset_days <= 0:
            return False
        
        hit_ts_str = state.get("risk_maxdd_hit_ts")
        if not hit_ts_str:
            return False
        
        try:
            crash_time = pd.to_datetime(hit_ts_str)
            if crash_time.tzinfo is None and bar_dt.tzinfo is not None:
                crash_time = crash_time.replace(tzinfo=bar_dt.tzinfo)
            time_passed = bar_dt - crash_time
            
            time_ok = time_passed.total_seconds() >= (reset_days * 86400)
            score_ok = current_regime_score >= reset_score
            
            return time_ok and score_ok
        except Exception as e:
            log.warning("Phoenix check failed: %s", e)
            return False
