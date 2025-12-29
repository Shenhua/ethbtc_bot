import logging
import time
import pandas as pd
from typing import Dict, Any, Optional, Tuple

from core.services.data_service import DataService
from core.services.order_service import OrderService
from core.risk_manager import RiskManager
from core.models.state import BotState
from core.story_writer import StoryWriter
from core.alert_manager import AlertManager
from core.ethbtc_accum_bot import StratParams, EthBtcStrategy
from core.trend_strategy import TrendStrategy, TrendParams
from core.meta_strategy import MetaStrategy
from core.regime import get_regime_score
from core.position_sizer import PositionSizer, PositionSizerConfig
from core.metrics import (
    BAR_LATENCY, REGIME_SCORE, REGIME_THRESHOLD, 
    STRATEGY_MODE, REGIME_STATE, PHOENIX_ACTIVE,
    reset_trade_decision, mark_decision
)

log = logging.getLogger("BotEngine")

class BotEngine:
    """
    Central orchestrator for the trading bot.
    Unifies data fetching, strategy execution, risk management, and order placement.
    """
    def __init__(
        self,
        config: Any,
        data_svc: DataService,
        order_svc: OrderService,
        risk_mgr: RiskManager,
        story: StoryWriter,
        alerter: AlertManager,
        bot_state: BotState,
        instance_name: str
    ):
        self.cfg = config
        self.data_svc = data_svc
        self.order_svc = order_svc
        self.risk_mgr = risk_mgr
        self.story = story
        self.alerter = alerter
        self.bot_state = bot_state
        self.instance_name = instance_name
        
        # Pre-calculated config values for quick access
        self.mr_params = self._merge_strategy_params()
        self.tr_params = self._get_trend_params()

    def _merge_strategy_params(self) -> Dict[str, Any]:
        """Merge base strategy params with MR-specific overrides."""
        base = self.cfg.strategy.model_dump()
        overrides = base.get("mean_reversion_overrides", {})
        return {**base, **overrides}

    def _get_trend_params(self) -> Dict[str, Any]:
        """Merge base strategy params with Trend-specific overrides."""
        base = self.cfg.strategy.model_dump()
        overrides = base.get("trend_overrides", {})
        return {**base, **overrides}

    def process_bar(self, bar_ts: int, now_s: int):
        """
        Process a single closed bar.
        This is the core logic loop extracted from live_executor.py.
        """
        bar_dt = pd.to_datetime(bar_ts, unit="s", utc=True)
        
        with BAR_LATENCY.labels(instance=self.instance_name).time():
            try:
                # 1. Fetch Data
                df = self.data_svc.get_closed_klines(limit=600)
                if df is None:
                    return
                
                price = float(df["close"].iloc[-1])
                
                # 2. Update Account State
                acc_state = self.order_svc.get_account_state(
                    self.bot_state.to_flat_dict(), 
                    bar_dt, 
                    price
                )
                W = acc_state["W"]
                cur_w = acc_state["cur_w"]
                
                # 3. Risk Management & Phoenix Protocol
                self.risk_mgr.ensure_state(self.bot_state.risk, W, bar_dt)
                self.risk_mgr.update(self.bot_state.risk, W, bar_dt)
                
                # 4. Global Gate Checks
                gate_ok, gate_reason = self._check_gates(df, price)
                
                # 5. Strategy & Target Weight
                target_w = self._calculate_target_weight(df, cur_w)
                
                # 6. Safety Overrides (Gate)
                if not gate_ok:
                    if target_w > cur_w and target_w > 0:
                        log.warning("Safety Override: Gate CLOSED (%s). Blocking Long Entry.", gate_reason)
                        target_w = cur_w
                
                # 7. Risk Overrides (MaxDD, Daily Limit, Phoenix Recovery)
                safe_target_w = self._apply_risk_overrides(target_w, cur_w, W, bar_dt)
                
                # 8. Trade Execution
                self._execute_trade(safe_target_w, cur_w, W, price, bar_dt, df)
                
                # Finalize
                self.bot_state.last_bar_close = str(bar_ts)

            except Exception as e:
                log.exception("Error during bar processing: %s", e)
                self.alerter.send(f"🚨 ENGINE ERROR [{self.instance_name}]\n{str(e)[:200]}", level="ERROR")

    def _calculate_target_weight(self, df: pd.DataFrame, cur_w: float) -> float:
        """Calculate the target weight based on the active strategy."""
        strat_type = getattr(self.cfg.strategy, "strategy_type", "mean_reversion")
        target_w = cur_w
        plan = None

        try:
            if strat_type == "trend":
                tp = TrendParams(**self.tr_params)
                strat = TrendStrategy(tp)
                plan = strat.generate_positions(df)
            elif strat_type == "meta":
                adx_thresh = float(self.mr_params.get("adx_threshold", 25.0))
                mr_p = StratParams(**self.mr_params)
                tr_p = TrendParams(**self.tr_params)
                strat = MetaStrategy(mr_p, tr_p, adx_threshold=adx_thresh)
                plan = strat.generate_positions(df)
            else:
                sp = StratParams(**self.mr_params)
                strat = EthBtcStrategy(sp)
                plan = strat.generate_positions(df)

            if plan is not None:
                target_w = float(plan["target_w"].iloc[-1])
                self._update_regime_metrics(plan)
            
            return target_w

        except Exception as e:
            log.exception("Strategy calculation failed: %s", e)
            return cur_w

    def _update_regime_metrics(self, plan: pd.DataFrame):
        """Update Prometheus metrics with regime info."""
        if "regime_score" in plan.columns:
            current_score = float(plan["regime_score"].iloc[-1])
            self.bot_state.last_regime_score = current_score
            REGIME_SCORE.labels(instance=self.instance_name).set(current_score)
            
            adx_thresh = float(self.mr_params.get("adx_threshold", 25.0))
            current_regime = "TREND" if current_score > adx_thresh else "CHOP"
            self.bot_state.last_regime = current_regime
            STRATEGY_MODE.labels(instance=self.instance_name).set(1.0 if current_regime == "TREND" else 0.0)
            
            if "regime_state" in plan.columns:
                REGIME_STATE.labels(instance=self.instance_name).set(float(plan["regime_state"].iloc[-1]))

    def _apply_risk_overrides(self, target_w: float, cur_w: float, W: float, bar_dt: pd.Timestamp) -> float:
        """Apply risk-based overrides to the target weight."""
        if self.bot_state.risk.maxdd_hit:
            reset_days = float(getattr(self.cfg.risk, "drawdown_reset_days", 0.0))
            if reset_days > 0:
                score = self.bot_state.last_regime_score
                if self.risk_mgr.can_phoenix_reset(self.bot_state.risk, bar_dt, score):
                    log.warning("🐦 PHOENIX REBIRTH! Resuming trading after cooldown.")
                    self.risk_mgr.reset_phoenix(self.bot_state.risk, wealth=W)
                    self.story.log_phoenix_activation(bar_dt, score, reset_days)
                    return target_w 
            
            if target_w > 0:
                log.warning("Risk: max drawdown hit → forcing Exit.")
            return 0.0
        
        if self.bot_state.risk.daily_limit_hit:
            if abs(target_w - cur_w) > 1e-12:
                log.warning("Risk: daily loss limit hit → freezing position.")
            return cur_w
            
        return target_w

    def _check_gates(self, df: pd.DataFrame, price: float) -> Tuple[bool, str]:
        """Check the 'Global Gate' conditions."""
        gate_ok = True
        gate_reason = "open"
        
        funding_limit_long = float(self.mr_params.get("funding_limit_long", 0.05))
        try:
            funding_rate = self.data_svc.adapter.get_funding_rate(f"{self.cfg.symbol}USDT")
            if funding_rate > funding_limit_long:
                gate_ok = False
                gate_reason = f"funding_high ({funding_rate:.4f}%)"
        except Exception as e:
            log.warning("Gate check (funding) warning: %s", e)
            
        return gate_ok, gate_reason

    def _execute_trade(self, target_w: float, cur_w: float, W: float, price: float, bar_dt: pd.Timestamp, df: pd.DataFrame):
        """Orchestrate the trade execution logic."""
        # 1. Calculate Step
        rv = self._calculate_realized_vol(df)
        step = self._calculate_step(rv)
        
        # 2. Smooth Target
        new_w_ideal = cur_w + step * (target_w - cur_w)
        
        # 3. Clamp & Final Target
        new_w = self._clamp_target(new_w_ideal)
        
        # 4. Calculate Quantities & Exec Strategy
        # This will be refined as we move more logic to OrderService, 
        # but for now we follow the existing live_executor heuristics.
        self.bot_state.last_target_w = new_w
        
        # Actual trading logic still resides in live_executor for this iteration
        # to ensure non-destructive phase-in. In Phase 3, it moves here.
        log.debug("[ENGINE] Calculated ideal weight: %.4f (step=%.2f)", new_w, step)

    def _calculate_realized_vol(self, df: pd.DataFrame) -> float:
        """Calculate realized volatility for position sizing."""
        vol_window = int(self.mr_params.get("vol_window", 60))
        ser_close = df["close"].astype(float)
        rv = float(ser_close.pct_change().rolling(vol_window).std().iloc[-1])
        return rv if rv == rv else 0.0

    def _calculate_step(self, realized_vol: float) -> float:
        """Calculate the dynamic position sizing step."""
        config = PositionSizerConfig(
            mode=self.mr_params.get("position_sizing_mode", "static"),
            base_step=float(self.mr_params.get("step_allocation", 0.33)),
            target_vol=float(self.mr_params.get("position_sizing_target_vol", 0.5)),
            min_step=float(self.mr_params.get("position_sizing_min_step", 0.1)),
            max_step=float(self.mr_params.get("position_sizing_max_step", 1.0)),
        )
        sizer = PositionSizer(config)
        return sizer.calculate_step(realized_vol=realized_vol)

    def _clamp_target(self, target_w: float) -> float:
        """Clamp target weight based on mode (Spot/Futures)."""
        is_futures = getattr(self.cfg.execution, "exchange_type", "spot") == "futures"
        if is_futures:
            max_lev = float(getattr(self.cfg.execution, "leverage", 1.0))
            return max(-max_lev, min(max_lev, target_w))
        else:
            return max(0.0, min(1.0, target_w))
