from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime

class RiskState(BaseModel):
    """Sub-model for risk-related state."""
    equity_high: float = 0.0
    current_date: Optional[str] = None
    daily_start_wealth: float = 0.0
    daily_limit_hit: bool = False
    maxdd_hit: bool = False
    maxdd_hit_ts: Optional[str] = None

class BotState(BaseModel):
    """Root model for application state."""
    session_start_wealth: float = Field(0.0, alias="session_start_W")
    last_known_position: float = 0.0
    last_position_fetch_ts: Optional[str] = None
    last_balance_log_ts: int = 0
    last_bar_close: Optional[str] = None
    last_target_w: Optional[float] = None
    last_regime: Optional[str] = None
    last_regime_score: float = 0.0
    alert_sent_maxdd: bool = False
    loop_started_at: Optional[str] = None
    
    # Generic storage for arbitrary state (backward compatibility)
    extra: Dict[str, Any] = Field(default_factory=dict)
    
    # Nested Risk State
    risk: RiskState = Field(default_factory=RiskState)

    class Config:
        populate_by_name = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BotState:
        """
        Custom factory to handle legacy flat state mapping.
        """
        # Extract risk keys if they exist in flat structure
        risk_data = {}
        for key in ["equity_high", "current_date", "daily_start_wealth", "daily_limit_hit", "maxdd_hit", "maxdd_hit_ts"]:
            flat_key = f"risk_{key}"
            if flat_key in data:
                risk_data[key] = data[flat_key]
        
        # Build core data
        core_data = {k: v for k, v in data.items() if not k.startswith("risk_")}
        if risk_data:
            core_data["risk"] = risk_data
            
        return cls(**core_data)

    def to_flat_dict(self) -> Dict[str, Any]:
        """
        Flatten state for legacy compatibility with JSON storage.
        """
        data = self.model_dump(by_alias=True, exclude={"risk", "extra"})
        
        # Add risk keys
        risk_data = self.risk.model_dump()
        for k, v in risk_data.items():
            data[f"risk_{k}"] = v
            
        # Add extra keys
        data.update(self.extra)
        
        return data
