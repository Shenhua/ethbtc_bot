"""
core/story_writer.py - Real-time Trading Narrative Logger

Tracks and logs key trading events to create a human-readable story file.
"""
import os
from datetime import datetime
from typing import Optional


class StoryWriter:
    """Writes real-time trading events to a story log file."""
    
    def __init__(self, filepath: str, symbol: str = "ASSET", alerter=None):
        """
        Initialize the story writer.
        
        Args:
            filepath: Path to the story log file
            symbol: Trading symbol for display
            alerter: Optional AlertManager instance for Discord/Telegram integration
        """
        self.filepath = filepath
        self.symbol = symbol
        self.alerter = alerter
        
        # State tracking
        self.last_regime: Optional[str] = None
        self.last_target_w: float = 0.0
        self.last_wealth: float = 0.0
        self.peak_wealth: float = 0.0
        self.is_halted: bool = False
        
        # Period tracking for summaries
        self.last_daily_report_date: Optional[datetime] = None
        self.last_weekly_report_date: Optional[datetime] = None
        self.last_monthly_report_date: Optional[datetime] = None
        self.last_annual_report_date: Optional[datetime] = None
        
        # Period stats
        self.period_start_wealth = {
            'daily': 0.0,
            'weekly': 0.0,
            'monthly': 0.0,
            'annual': 0.0
        }
        self.period_trade_count = {
            'daily': 0,
            'weekly': 0,
            'monthly': 0,
            'annual': 0
        }
        self.period_regime_counts = {
            'weekly_trend': 0,
            'weekly_mr': 0,
            'monthly_trend': 0,
            'monthly_mr': 0
        }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        
    def _write_line(self, timestamp: datetime, icon: str, event: str, details: str = ""):
        """Write a formatted line to the story file."""
        ts_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        line = f"{ts_str} | {icon} {event:<40} | {details}\n"
        
        try:
            with open(self.filepath, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception as e:
            # Silent fail - don't crash the bot if logging fails
            print(f"[StoryWriter] Warning: Failed to write to {self.filepath}: {e}")
    
    def log_startup(self, timestamp: datetime, initial_wealth: float, mode: str, quote_asset: str):
        """Log bot startup."""
        self.peak_wealth = initial_wealth
        self.last_wealth = initial_wealth
        
        self._write_line(timestamp, "🚀", "BOT STARTED", 
                         f"Mode: {mode.upper()} | Wealth: {initial_wealth:.6f} {quote_asset}")
        self._write_line(timestamp, "=" * 80, "", "")
    
    def check_ath(self, timestamp: datetime, current_wealth: float, quote_asset: str) -> bool:
        """
        Check for All-Time High and log if detected.
        
        Returns:
            True if new ATH was detected
        """
        # BUG FIX: If peak_wealth is 0 (initial state), just set it and return
        if self.peak_wealth <= 1e-9:
            self.peak_wealth = current_wealth
            self.last_wealth = current_wealth
            return False

        if current_wealth > self.peak_wealth:
            # Only log if it's a significant increase (>2%)
            if current_wealth > self.last_wealth * 1.02:
                gain_pct = ((current_wealth / self.peak_wealth) - 1.0) * 100
                self._write_line(timestamp, "🚀", "NEW ALL-TIME HIGH", 
                               f"{current_wealth:.6f} {quote_asset} (+{gain_pct:.2f}%)")
                
                # Send to Discord if enabled
                if self.alerter:
                    self.alerter.send(
                        f"🚀 NEW ATH [{self.symbol}]\n"
                        f"Wealth: {current_wealth:.6f} {quote_asset} (+{gain_pct:.2f}%)",
                        level="INFO"
                    )
            self.peak_wealth = current_wealth
            self.last_wealth = current_wealth
            return True
        
        self.last_wealth = current_wealth
        return False
    
    def check_regime_switch(self, timestamp: datetime, score: float, 
                            threshold: float, strategy_type: str = "meta"):
        """Check for regime switch and log if detected."""
        if strategy_type != "meta":
            return  # Only log for meta strategy
        
        current_regime = "TREND" if score > threshold else "MEAN_REV"
        
        if self.last_regime is not None and current_regime != self.last_regime:
            self._write_line(timestamp, "🔄", f"REGIME SWITCH: {current_regime:<15}", 
                            f"Score: {score:.1f} (Threshold: {threshold:.1f})")
        
        # Track regime for summaries (even if no switch)
        self._track_regime(current_regime)
        
        self.last_regime = current_regime
    
    def log_trade(self, timestamp: datetime, side: str, quantity: float, 
                  price: float, base_asset: str, quote_asset: str):
        """Log a trade execution."""
        icon = "🟢" if side == "BUY" else "🔴"
        notional = abs(quantity * price)
        
        self._write_line(timestamp, icon, f"{side:<40}", 
                         f"{abs(quantity):.6f} {base_asset} @ {price:.8f} ({notional:.6f} {quote_asset})")
        
        # Track trade count for summaries
        self._increment_trade_count()
    
    def log_safety_breaker(self, timestamp: datetime, drawdown_pct: float):
        """Log safety breaker trip."""
        if not self.is_halted:
            self._write_line(timestamp, "🚨", "SAFETY BREAKER TRIPPED", 
                             f"Drawdown: -{drawdown_pct:.1%} | Trading HALTED")
            
            # Send to Discord if enabled
            if self.alerter:
                self.alerter.send(
                    f"🚨 SAFETY BREAKER TRIPPED [{self.symbol}]\n"
                    f"Drawdown: -{drawdown_pct:.1%} | Trading HALTED",
                    level="CRITICAL"
                )
            
            self.is_halted = True
    
    def log_phoenix_activation(self, timestamp: datetime, score: float, 
                               cooldown_days: float):
        """Log Phoenix Protocol activation (reset)."""
        if self.is_halted:
            self._write_line(timestamp, "🔥", "PHOENIX PROTOCOL ACTIVATED", 
                             f"Score: {score:.1f} | Cooldown: {cooldown_days:.1f}d | Resuming Trading")
            
            # Send to Discord if enabled
            if self.alerter:
                self.alerter.send(
                    f"🔥 PHOENIX PROTOCOL ACTIVATED [{self.symbol}]\n"
                    f"Score: {score:.1f} | Resuming Trading",
                    level="INFO"
                )
            
            self.is_halted = False
    
    def log_daily_summary(self, timestamp: datetime, daily_pnl: float, 
                          current_wealth: float, quote_asset: str):
        """Log daily summary (call once per day)."""
        icon = "📈" if daily_pnl >= 0 else "📉"
        sign = "+" if daily_pnl >= 0 else ""
        
        self._write_line(timestamp, icon, "DAILY SUMMARY", 
                         f"PnL: {sign}{daily_pnl:.6f} {quote_asset} | Wealth: {current_wealth:.6f} {quote_asset}")
    
    def check_and_log_daily(self, timestamp: datetime, daily_pnl: float, 
                          current_wealth: float, quote_asset: str):
        """Check if day changed and log summary."""
        if self.last_daily_report_date and self.last_daily_report_date.day == timestamp.day:
            return

        if self.period_start_wealth['daily'] == 0.0:
            self.period_start_wealth['daily'] = current_wealth
            self.last_daily_report_date = timestamp
            return

        self.log_daily_summary(timestamp, daily_pnl, current_wealth, quote_asset)
        self.period_start_wealth['daily'] = current_wealth
        self.last_daily_report_date = timestamp

    def log_custom(self, timestamp: datetime, icon: str, event: str, details: str = ""):
        """Log a custom event."""
        self._write_line(timestamp, icon, event, details)
    
    def _increment_trade_count(self):
        """Increment trade counters for all periods."""
        for period in ['daily', 'weekly', 'monthly', 'annual']:
            self.period_trade_count[period] += 1
    
    def _track_regime(self, regime: str):
        """Track regime for weekly/monthly stats."""
        if regime == "TREND":
            self.period_regime_counts['weekly_trend'] += 1
            self.period_regime_counts['monthly_trend'] += 1
        else:
            self.period_regime_counts['weekly_mr'] += 1
            self.period_regime_counts['monthly_mr'] += 1
    
    def _reset_period_stats(self, period: str, current_wealth: float):
        """Reset stats for a given period."""
        if period in self.period_start_wealth:
            self.period_start_wealth[period] = current_wealth
            self.period_trade_count[period] = 0
        
        if period == 'weekly':
            self.period_regime_counts['weekly_trend'] = 0
            self.period_regime_counts['weekly_mr'] = 0
        elif period == 'monthly':
            self.period_regime_counts['monthly_trend'] = 0
            self.period_regime_counts['monthly_mr'] = 0
    
    def check_and_log_weekly(self, timestamp: datetime, current_wealth: float, quote_asset: str):
        """Check if Sunday (week end) and log weekly summary."""
        # Check if it's Sunday (weekday() returns 6 for Sunday)
        if timestamp.weekday() != 6:
            return
        
        # Check if we already reported this week
        if self.last_weekly_report_date and self.last_weekly_report_date.isocalendar()[1] == timestamp.isocalendar()[1]:
            return
        
        # Initialize if first run
        if self.period_start_wealth['weekly'] == 0.0:
            self._reset_period_stats('weekly', current_wealth)
            self.last_weekly_report_date = timestamp
            return
        
        # Calculate stats
        start_wealth = self.period_start_wealth['weekly']
        pnl = current_wealth - start_wealth
        pnl_pct = (pnl / start_wealth * 100) if start_wealth > 0 else 0.0
        trades = self.period_trade_count['weekly']
        
        trend_count = self.period_regime_counts['weekly_trend']
        mr_count = self.period_regime_counts['weekly_mr']
        total_regime = trend_count + mr_count
        trend_pct = (trend_count / total_regime * 100) if total_regime > 0 else 0
        mr_pct = (mr_count / total_regime * 100) if total_regime > 0 else 0
        
        # Format message
        sign = "+" if pnl >= 0 else ""
        icon = "📊" if pnl >= 0 else "📉"
        
        details = (f"PnL: {sign}{pnl:.6f} {quote_asset} ({sign}{pnl_pct:.2f}%) | "
                   f"Trades: {trades} | Regime: {trend_pct:.0f}% TREND, {mr_pct:.0f}% MR")
        
        self._write_line(timestamp, icon, "WEEKLY SUMMARY", details)
        self._write_line(timestamp, "=" * 80, "", "")
        
        # Send to Discord if enabled
        if self.alerter:
            self.alerter.send(f"📊 Weekly Summary [{self.symbol}]\n{details}", level="INFO")
        
        # Reset for next week
        self._reset_period_stats('weekly', current_wealth)
        self.last_weekly_report_date = timestamp
    
    def check_and_log_monthly(self, timestamp: datetime, current_wealth: float, quote_asset: str):
        """Check if 1st of month and log monthly summary."""
        # Check if it's the 1st of the month
        if timestamp.day != 1:
            return
        
        # Check if we already reported this month
        if self.last_monthly_report_date and self.last_monthly_report_date.month == timestamp.month:
            return
        
        # Initialize if first run
        if self.period_start_wealth['monthly'] == 0.0:
            self._reset_period_stats('monthly', current_wealth)
            self.last_monthly_report_date = timestamp
            return
        
        # Calculate stats
        start_wealth = self.period_start_wealth['monthly']
        pnl = current_wealth - start_wealth
        pnl_pct = (pnl / start_wealth * 100) if start_wealth > 0 else 0.0
        trades = self.period_trade_count['monthly']
        
        trend_count = self.period_regime_counts['monthly_trend']
        mr_count = self.period_regime_counts['monthly_mr']
        total_regime = trend_count + mr_count
        trend_pct = (trend_count / total_regime * 100) if total_regime > 0 else 0
        mr_pct = (mr_count / total_regime * 100) if total_regime > 0 else 0
        
        # Format message
        sign = "+" if pnl >= 0 else ""
        icon = "📊" if pnl >= 0 else "📉"
        month_name = timestamp.strftime("%B %Y")
        
        details = (f"{month_name} | PnL: {sign}{pnl:.6f} {quote_asset} ({sign}{pnl_pct:.2f}%) | "
                   f"Trades: {trades} | Regime: {trend_pct:.0f}% TREND, {mr_pct:.0f}% MR")
        
        self._write_line(timestamp, icon, "MONTHLY SUMMARY", details)
        self._write_line(timestamp, "=" * 80, "", "")
        
        # Send to Discord if enabled  
        if self.alerter:
            self.alerter.send(f"📊 Monthly Summary [{self.symbol}]\n{details}", level="INFO")
        
        # Reset for next month
        self._reset_period_stats('monthly', current_wealth)
        self.last_monthly_report_date = timestamp
    
    def check_and_log_annual(self, timestamp: datetime, current_wealth: float, quote_asset: str):
        """Check if Jan 1st and log annual summary."""
        # Check if it's January 1st
        if timestamp.month != 1 or timestamp.day != 1:
            return
        
        # Check if we already reported this year
        if self.last_annual_report_date and self.last_annual_report_date.year == timestamp.year:
            return
        
        # Initialize if first run
        if self.period_start_wealth['annual'] == 0.0:
            self._reset_period_stats('annual', current_wealth)
            self.last_annual_report_date = timestamp
            return
        
        # Calculate stats
        start_wealth = self.period_start_wealth['annual']
        pnl = current_wealth - start_wealth
        pnl_pct = (pnl / start_wealth * 100) if start_wealth > 0 else 0.0
        trades = self.period_trade_count['annual']
        
        # Format message
        sign = "+" if pnl >= 0 else ""
        icon = "🎉" if pnl >= 0 else "📉"
        year = timestamp.year - 1  # Reporting for the year that just ended
        
        details = (f"{year} | PnL: {sign}{pnl:.6f} {quote_asset} ({sign}{pnl_pct:.2f}%) | "
                   f"Total Trades: {trades}")
        
        self._write_line(timestamp, icon, "ANNUAL SUMMARY", details)
        self._write_line(timestamp, "=" * 80, "", "")
        
        # Send to Discord if enabled
        if self.alerter:
            self.alerter.send(f"🎉 Annual Summary [{self.symbol}] - {year}\n{details}", level="INFO")
        
        # Reset for next year
        self._reset_period_stats('annual', current_wealth)
        self.last_annual_report_date = timestamp