"""
core/backtest_report.py - Enhanced Backtest Reporting

Generates comprehensive backtest reports with:
- HODL comparison and alpha calculation
- Risk-adjusted metrics (Sharpe, Sortino, Calmar, VaR, CVaR)
- Trading analysis (win rate, profit factor, time in market)
- Formatted output (terminal, Markdown)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import os


@dataclass
class BacktestReport:
    """
    Comprehensive backtest report generator.
    
    Usage:
        report = BacktestReport.from_backtest_result(result, price_series)
        report.print_report()  # Terminal output
        report.to_markdown("results/report.md")  # Save to file
    """
    
    # Strategy identification
    strategy_name: str = "Strategy"
    symbol: str = "ASSET"
    base_asset: str = "ETH"
    quote_asset: str = "BTC"
    start_date: str = ""
    end_date: str = ""
    duration_days: float = 0.0
    
    # Basic Performance
    initial_capital: float = 1.0
    final_capital: float = 1.0
    total_return_pct: float = 0.0
    cagr_pct: float = 0.0
    
    # Benchmark Comparison
    hodl_base_return_pct: float = 0.0  # If held 100% base asset (e.g., ETH)
    alpha_pct: float = 0.0  # Strategy return - 0 (vs holding quote)
    
    # Risk Metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_duration_days: float = 0.0
    avg_drawdown_pct: float = 0.0
    volatility_ann_pct: float = 0.0
    var_95_pct: float = 0.0
    cvar_95_pct: float = 0.0
    
    # Trading Analysis
    n_trades: int = 0
    win_rate_pct: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    profit_factor: float = 0.0
    avg_holding_bars: float = 0.0
    time_in_market_pct: float = 0.0
    avg_exposure_pct: float = 0.0
    total_fees: float = 0.0
    fees_as_pct_gross: float = 0.0
    
    # Period Breakdown
    monthly_returns: Dict[str, float] = field(default_factory=dict)
    yearly_returns: Dict[str, float] = field(default_factory=dict)
    best_month_pct: float = 0.0
    worst_month_pct: float = 0.0
    positive_months_pct: float = 0.0
    
    # Regime Analysis (for Meta Strategy)
    mr_regime_pnl: float = 0.0
    trend_regime_pnl: float = 0.0
    mr_regime_pct: float = 0.0
    trend_regime_pct: float = 0.0
    
    @classmethod
    def from_backtest_result(
        cls,
        result: Dict[str, Any],
        price_series: pd.Series,
        strategy_name: str = "Strategy",
        symbol: str = "ASSET",
        base_asset: str = "ETH",
        quote_asset: str = "BTC",
        exposure_series: Optional[pd.Series] = None,
        regime_series: Optional[pd.Series] = None,
    ) -> "BacktestReport":
        """
        Create a BacktestReport from Backtester.simulate() output.
        
        Args:
            result: Dict from Backtester.simulate()
            price_series: Close price series (for HODL comparison)
            strategy_name: Name of strategy for display
            symbol: Trading symbol
            exposure_series: Optional series of position weights
            regime_series: Optional series of regime states (-1=MR, 1=Trend)
        """
        summary = result.get("summary", {})
        portfolio = result.get("portfolio", pd.DataFrame())
        trades_df = result.get("trades", pd.DataFrame())
        diagnostics = result.get("diagnostics")
        
        # Extract wealth series
        if "wealth_btc" in portfolio.columns:
            wealth = portfolio["wealth_btc"]
        else:
            wealth = portfolio.iloc[:, 0] if len(portfolio.columns) > 0 else pd.Series([1.0])
        
        # Dates
        if len(wealth) > 0:
            start_date = str(wealth.index[0].date()) if hasattr(wealth.index[0], 'date') else str(wealth.index[0])
            end_date = str(wealth.index[-1].date()) if hasattr(wealth.index[-1], 'date') else str(wealth.index[-1])
            duration_days = (wealth.index[-1] - wealth.index[0]).days if hasattr(wealth.index[-1], 'days') or hasattr((wealth.index[-1] - wealth.index[0]), 'days') else len(wealth) / 96  # Assume 15m bars
        else:
            start_date, end_date, duration_days = "", "", 0.0
        
        # Try to get duration correctly
        try:
            duration_days = (wealth.index[-1] - wealth.index[0]).total_seconds() / 86400
        except:
            duration_days = len(wealth) / 96  # Fallback: assume 15m bars
        
        # Basic Performance
        initial = summary.get("initial_btc", 1.0)
        final = summary.get("final_btc", wealth.iloc[-1] if len(wealth) > 0 else 1.0)
        total_return = (final / initial) - 1.0 if initial > 0 else 0.0
        
        # CAGR
        years = duration_days / 365.25
        cagr = ((final / initial) ** (1 / years) - 1.0) if years > 0 and initial > 0 else 0.0
        
        # HODL Comparison
        # HODL quote (BTC) = 0% return (you're already in quote)
        # HODL base (ETH) = price change
        if len(price_series) > 1:
            hodl_base_return = (price_series.iloc[-1] / price_series.iloc[0]) - 1.0
        else:
            hodl_base_return = 0.0
        alpha = total_return - hodl_base_return  # Excess return vs Base Asset
        
        # Daily Returns for Risk Metrics
        daily_wealth = wealth.resample('D').last().dropna()
        daily_returns = daily_wealth.pct_change().dropna()
        
        # Sharpe Ratio (assuming 0% risk-free rate for crypto)
        sharpe = cls._calculate_sharpe(daily_returns, risk_free=0.0)
        sortino = cls._calculate_sortino(daily_returns, risk_free=0.0)
        
        # Volatility
        volatility_ann = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 1 else 0.0
        
        # Max Drawdown
        max_dd_pct, max_dd_duration = cls._calculate_max_drawdown(wealth)
        avg_dd_pct = cls._calculate_avg_drawdown(wealth)
        
        # Calmar
        calmar = abs(cagr / max_dd_pct) if max_dd_pct != 0 else 0.0
        
        # VaR and CVaR
        var_95, cvar_95 = cls._calculate_var_cvar(daily_returns)
        
        # Trading Analysis
        n_trades = summary.get("n_trades", len(trades_df))
        total_fees = summary.get("fees_btc", 0.0)
        
        win_rate, avg_win, avg_loss, profit_factor = cls._calculate_trade_stats(trades_df)
        avg_holding = cls._calculate_avg_holding(trades_df)
        
        # Exposure Analysis
        time_in_market, avg_exposure = cls._calculate_exposure_stats(
            exposure_series, diagnostics, wealth
        )
        
        # Fee efficiency
        gross_profit = sum(t.get("pnl", 0) for _, t in trades_df.iterrows() if t.get("pnl", 0) > 0) if "pnl" in trades_df.columns else (final - initial)
        fees_as_pct = (total_fees / gross_profit) * 100 if gross_profit > 0 else 0.0
        
        # Monthly/Yearly Returns
        monthly_rets, yearly_rets = cls._calculate_period_returns(daily_wealth)
        
        best_month = max(monthly_rets.values()) * 100 if monthly_rets else 0.0
        worst_month = min(monthly_rets.values()) * 100 if monthly_rets else 0.0
        positive_months = (sum(1 for v in monthly_rets.values() if v > 0) / len(monthly_rets) * 100) if monthly_rets else 0.0
        
        # Regime Analysis
        mr_pnl, trend_pnl, mr_pct, trend_pct = cls._calculate_regime_pnl(
            wealth, regime_series, diagnostics
        )
        
        return cls(
            strategy_name=strategy_name,
            symbol=symbol,
            base_asset=base_asset,
            quote_asset=quote_asset,
            start_date=start_date,
            end_date=end_date,
            duration_days=duration_days,
            initial_capital=initial,
            final_capital=final,
            total_return_pct=total_return * 100,
            cagr_pct=cagr * 100,
            hodl_base_return_pct=hodl_base_return * 100,
            alpha_pct=alpha * 100,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown_pct=max_dd_pct * 100,
            max_drawdown_duration_days=max_dd_duration,
            avg_drawdown_pct=avg_dd_pct * 100,
            volatility_ann_pct=volatility_ann * 100,
            var_95_pct=var_95 * 100,
            cvar_95_pct=cvar_95 * 100,
            n_trades=n_trades,
            win_rate_pct=win_rate,
            avg_win_pct=avg_win,
            avg_loss_pct=avg_loss,
            profit_factor=profit_factor,
            avg_holding_bars=avg_holding,
            time_in_market_pct=time_in_market,
            avg_exposure_pct=avg_exposure,
            total_fees=total_fees,
            fees_as_pct_gross=fees_as_pct,
            monthly_returns={k: v * 100 for k, v in monthly_rets.items()},
            yearly_returns={k: v * 100 for k, v in yearly_rets.items()},
            best_month_pct=best_month,
            worst_month_pct=worst_month,
            positive_months_pct=positive_months,
            mr_regime_pnl=mr_pnl,
            trend_regime_pnl=trend_pnl,
            mr_regime_pct=mr_pct,
            trend_regime_pct=trend_pct,
        )
    
    # ==================== CALCULATION METHODS ====================
    
    @staticmethod
    def _calculate_sharpe(daily_returns: pd.Series, risk_free: float = 0.0) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(daily_returns) < 2:
            return 0.0
        excess = daily_returns - (risk_free / 252)
        std = excess.std()
        if std == 0 or np.isnan(std):
            return 0.0
        return float((excess.mean() / std) * np.sqrt(252))
    
    @staticmethod
    def _calculate_sortino(daily_returns: pd.Series, risk_free: float = 0.0) -> float:
        """Calculate annualized Sortino ratio (downside deviation only)."""
        if len(daily_returns) < 2:
            return 0.0
        excess = daily_returns - (risk_free / 252)
        downside = excess[excess < 0]
        if len(downside) == 0:
            return float('inf') if excess.mean() > 0 else 0.0
        downside_std = downside.std()
        if downside_std == 0 or np.isnan(downside_std):
            return 0.0
        return float((excess.mean() / downside_std) * np.sqrt(252))
    
    @staticmethod
    def _calculate_max_drawdown(wealth: pd.Series) -> tuple:
        """Calculate max drawdown percentage and duration in days."""
        if len(wealth) < 2:
            return 0.0, 0.0
        
        running_max = wealth.expanding().max()
        drawdown = (wealth - running_max) / running_max
        max_dd = abs(drawdown.min())
        
        # Duration: find longest underwater period
        underwater = drawdown < 0
        if not underwater.any():
            return float(max_dd), 0.0
        
        # Calculate longest streak
        groups = (underwater != underwater.shift()).cumsum()
        underwater_periods = underwater.groupby(groups).sum()
        max_duration_bars = underwater_periods.max()
        
        # Convert bars to days (assume 15m = 96 bars/day)
        try:
            bar_duration = (wealth.index[1] - wealth.index[0]).total_seconds() / 86400
        except:
            bar_duration = 15 / (24 * 60)  # 15min default
        
        return float(max_dd), float(max_duration_bars * bar_duration)
    
    @staticmethod
    def _calculate_avg_drawdown(wealth: pd.Series) -> float:
        """Calculate average drawdown."""
        if len(wealth) < 2:
            return 0.0
        running_max = wealth.expanding().max()
        drawdown = (wealth - running_max) / running_max
        return float(abs(drawdown[drawdown < 0].mean())) if (drawdown < 0).any() else 0.0
    
    @staticmethod
    def _calculate_var_cvar(daily_returns: pd.Series, confidence: float = 0.95) -> tuple:
        """Calculate Value at Risk and Conditional VaR (Expected Shortfall)."""
        if len(daily_returns) < 10:
            return 0.0, 0.0
        
        var = daily_returns.quantile(1 - confidence)
        cvar = daily_returns[daily_returns <= var].mean()
        
        return float(abs(var)), float(abs(cvar)) if not np.isnan(cvar) else float(abs(var))
    
    @staticmethod
    def _calculate_trade_stats(trades_df: pd.DataFrame) -> tuple:
        """Calculate win rate, avg win/loss, profit factor."""
        if trades_df.empty or len(trades_df) == 0:
            return 0.0, 0.0, 0.0, 0.0
        
        # If we have PnL per trade
        if "pnl" in trades_df.columns:
            pnl = trades_df["pnl"]
            wins = pnl[pnl > 0]
            losses = pnl[pnl < 0]
        else:
            # Estimate from qty/price if available
            # For now, return defaults
            return 50.0, 0.0, 0.0, 1.0
        
        win_rate = (len(wins) / len(trades_df)) * 100 if len(trades_df) > 0 else 0.0
        avg_win = float(wins.mean()) * 100 if len(wins) > 0 else 0.0
        avg_loss = float(abs(losses.mean())) * 100 if len(losses) > 0 else 0.0
        
        gross_profit = wins.sum() if len(wins) > 0 else 0.0
        gross_loss = abs(losses.sum()) if len(losses) > 0 else 0.0001
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return win_rate, avg_win, avg_loss, min(profit_factor, 999.0)
    
    @staticmethod
    def _calculate_avg_holding(trades_df: pd.DataFrame) -> float:
        """Calculate average holding period in bars."""
        if trades_df.empty or "time" not in trades_df.columns:
            return 0.0
        
        # Simple approximation: bars between trades
        if len(trades_df) < 2:
            return 0.0
        
        times = pd.to_datetime(trades_df["time"])
        avg_delta = times.diff().mean()
        
        if pd.isna(avg_delta):
            return 0.0
        
        # Convert to bars (assume 15m)
        return float(avg_delta.total_seconds() / (15 * 60))
    
    @staticmethod
    def _calculate_exposure_stats(
        exposure_series: Optional[pd.Series],
        diagnostics: Optional[pd.DataFrame],
        wealth: pd.Series
    ) -> tuple:
        """Calculate time in market and average exposure."""
        exposure = None
        
        if exposure_series is not None:
            exposure = exposure_series
        elif diagnostics is not None and "target_w" in diagnostics.columns:
            exposure = diagnostics["target_w"].abs()
        
        if exposure is None or len(exposure) == 0:
            return 50.0, 50.0  # Default fallback
        
        time_in_market = (exposure.abs() > 0.01).mean() * 100
        avg_exposure = exposure.abs().mean() * 100
        
        return float(time_in_market), float(avg_exposure)
    
    @staticmethod
    def _calculate_period_returns(daily_wealth: pd.Series) -> tuple:
        """Calculate monthly and yearly returns."""
        if len(daily_wealth) < 2:
            return {}, {}
        
        monthly = {}
        yearly = {}
        
        # Monthly
        try:
            monthly_wealth = daily_wealth.resample('ME').last().dropna()
            monthly_rets = monthly_wealth.pct_change().dropna()
            for idx, ret in monthly_rets.items():
                key = idx.strftime('%Y-%m')
                monthly[key] = float(ret)
        except:
            pass
        
        # Yearly
        try:
            yearly_wealth = daily_wealth.resample('YE').last().dropna()
            yearly_rets = yearly_wealth.pct_change().dropna()
            for idx, ret in yearly_rets.items():
                key = str(idx.year)
                yearly[key] = float(ret)
        except:
            pass
        
        return monthly, yearly
    
    @staticmethod
    def _calculate_regime_pnl(
        wealth: pd.Series,
        regime_series: Optional[pd.Series],
        diagnostics: Optional[pd.DataFrame]
    ) -> tuple:
        """Calculate PnL breakdown by regime (MR vs Trend)."""
        if diagnostics is None or "regime_state" not in diagnostics.columns:
            return 0.0, 0.0, 50.0, 50.0
        
        regime = diagnostics["regime_state"]
        
        # Align
        common_idx = wealth.index.intersection(regime.index)
        if len(common_idx) < 2:
            return 0.0, 0.0, 50.0, 50.0
        
        w = wealth.reindex(common_idx)
        r = regime.reindex(common_idx)
        
        # Calculate returns during each regime
        rets = w.pct_change().fillna(0)
        
        mr_mask = r < 0  # -1 = MR
        trend_mask = r > 0  # 1 = Trend
        
        mr_pnl = float(rets[mr_mask].sum()) if mr_mask.any() else 0.0
        trend_pnl = float(rets[trend_mask].sum()) if trend_mask.any() else 0.0
        
        total_bars = len(r)
        mr_pct = (mr_mask.sum() / total_bars) * 100 if total_bars > 0 else 50.0
        trend_pct = (trend_mask.sum() / total_bars) * 100 if total_bars > 0 else 50.0
        
        return mr_pnl * 100, trend_pnl * 100, mr_pct, trend_pct
    
    # ==================== OUTPUT METHODS ====================
    
    def to_dict(self) -> Dict[str, Any]:
        """Return all metrics as a dictionary."""
        return {
            "strategy_name": self.strategy_name,
            "symbol": self.symbol,
            "period": f"{self.start_date} → {self.end_date}",
            "duration_days": round(self.duration_days, 1),
            
            "performance": {
                "initial_capital": self.initial_capital,
                "final_capital": round(self.final_capital, 6),
                "total_return_pct": round(self.total_return_pct, 2),
                "cagr_pct": round(self.cagr_pct, 2),
            },
            
            "benchmark": {
                "hodl_base_return_pct": round(self.hodl_base_return_pct, 2),
                "alpha_pct": round(self.alpha_pct, 2),
            },
            
            "risk": {
                "sharpe_ratio": round(self.sharpe_ratio, 2),
                "sortino_ratio": round(self.sortino_ratio, 2),
                "calmar_ratio": round(self.calmar_ratio, 2),
                "max_drawdown_pct": round(self.max_drawdown_pct, 2),
                "max_dd_duration_days": round(self.max_drawdown_duration_days, 1),
                "volatility_ann_pct": round(self.volatility_ann_pct, 2),
                "var_95_pct": round(self.var_95_pct, 2),
                "cvar_95_pct": round(self.cvar_95_pct, 2),
            },
            
            "trading": {
                "n_trades": self.n_trades,
                "win_rate_pct": round(self.win_rate_pct, 1),
                "profit_factor": round(self.profit_factor, 2),
                "time_in_market_pct": round(self.time_in_market_pct, 1),
                "avg_exposure_pct": round(self.avg_exposure_pct, 1),
                "total_fees": round(self.total_fees, 6),
            },
            
            "period_returns": {
                "monthly": self.monthly_returns,
                "yearly": self.yearly_returns,
                "best_month_pct": round(self.best_month_pct, 2),
                "worst_month_pct": round(self.worst_month_pct, 2),
                "positive_months_pct": round(self.positive_months_pct, 1),
            },
        }
    
    def print_report(self) -> None:
        """Print formatted report to terminal with colors."""
        # ANSI color codes
        GREEN = "\033[92m"
        RED = "\033[91m"
        YELLOW = "\033[93m"
        CYAN = "\033[96m"
        BOLD = "\033[1m"
        RESET = "\033[0m"
        
        def color_value(val, good_threshold=0, bad_threshold=None, invert=False):
            """Color a value green/red based on thresholds."""
            if invert:
                if val < good_threshold:
                    return f"{GREEN}{val:+.2f}{RESET}"
                elif bad_threshold and val > bad_threshold:
                    return f"{RED}{val:+.2f}{RESET}"
            else:
                if val > good_threshold:
                    return f"{GREEN}{val:+.2f}{RESET}"
                elif bad_threshold is not None and val < bad_threshold:
                    return f"{RED}{val:+.2f}{RESET}"
            return f"{val:+.2f}"
        
        def color_ratio(val, good=1.0, great=2.0):
            if val >= great:
                return f"{GREEN}{BOLD}{val:.2f}{RESET}"
            elif val >= good:
                return f"{GREEN}{val:.2f}{RESET}"
            elif val < 0:
                return f"{RED}{val:.2f}{RESET}"
            return f"{YELLOW}{val:.2f}{RESET}"
        
        print()
        print(f"{BOLD}═══════════════════════════════════════════════════════════════════════════{RESET}")
        print(f"{BOLD}                    BACKTEST REPORT: {self.symbol} {self.strategy_name.upper()}{RESET}")
        print(f"{BOLD}                    Period: {self.start_date} → {self.end_date} ({self.duration_days:.0f} days){RESET}")
        print(f"{BOLD}═══════════════════════════════════════════════════════════════════════════{RESET}")
        print()
        
        # Performance Section
        print(f"{CYAN}{BOLD}💰 PERFORMANCE{RESET}")
        print(f"   Initial Capital:        {self.initial_capital:.4f}")
        print(f"   Final Capital:          {self.final_capital:.4f}")
        print(f"   Total Return:           {color_value(self.total_return_pct)}%  (CAGR: {self.cagr_pct:+.1f}%)")
        print()
        
        # Benchmark Section
        print(f"{CYAN}{BOLD}📊 BENCHMARK COMPARISON{RESET}")
        print(f"   HODL Quote ({self.quote_asset}):       {GREEN}+0.00%{RESET}  (Your baseline)")
        print(f"   HODL {self.base_asset}:        {color_value(self.hodl_base_return_pct)}%")
        alpha_color = GREEN if self.alpha_pct > 0 else RED
        print(f"   {BOLD}✨ ALPHA vs HODL:        {alpha_color}{self.alpha_pct:+.2f}%{RESET}  ← {BOLD}YOUR VALUE ADD{RESET}")
        print()
        
        # Risk Section
        print(f"{CYAN}{BOLD}📉 RISK METRICS{RESET}")
        print(f"   Sharpe Ratio:           {color_ratio(self.sharpe_ratio)}")
        print(f"   Sortino Ratio:          {color_ratio(self.sortino_ratio)}")
        print(f"   Calmar Ratio:           {color_ratio(self.calmar_ratio)}")
        dd_color = GREEN if self.max_drawdown_pct < 10 else (YELLOW if self.max_drawdown_pct < 20 else RED)
        print(f"   Max Drawdown:           {dd_color}{self.max_drawdown_pct:.1f}%{RESET} (Duration: {self.max_drawdown_duration_days:.0f} days)")
        print(f"   Volatility (Ann.):      {self.volatility_ann_pct:.1f}%")
        print(f"   VaR (95%):              {self.var_95_pct:.2f}%")
        print(f"   CVaR (95%):             {self.cvar_95_pct:.2f}%")
        print()
        
        # Trading Section
        print(f"{CYAN}{BOLD}📈 TRADING ANALYSIS{RESET}")
        print(f"   Total Trades:           {self.n_trades}")
        wr_color = GREEN if self.win_rate_pct >= 50 else YELLOW
        print(f"   Win Rate:               {wr_color}{self.win_rate_pct:.1f}%{RESET}")
        pf_color = GREEN if self.profit_factor >= 1.5 else (YELLOW if self.profit_factor >= 1.0 else RED)
        print(f"   Profit Factor:          {pf_color}{self.profit_factor:.2f}{RESET}")
        print(f"   Time in Market:         {self.time_in_market_pct:.1f}%")
        print(f"   Avg Exposure:           {self.avg_exposure_pct:.1f}%")
        print(f"   Total Fees Paid:        {self.total_fees:.6f}")
        print()
        
        # Period Returns
        if self.yearly_returns:
            print(f"{CYAN}{BOLD}📅 YEARLY RETURNS{RESET}")
            for year, ret in sorted(self.yearly_returns.items()):
                ret_color = GREEN if ret > 0 else RED
                print(f"   {year}:                   {ret_color}{ret:+.1f}%{RESET}")
            print()
        
        # Monthly Stats
        print(f"{CYAN}{BOLD}📆 MONTHLY STATS{RESET}")
        print(f"   Best Month:             {GREEN}{self.best_month_pct:+.1f}%{RESET}")
        print(f"   Worst Month:            {RED}{self.worst_month_pct:+.1f}%{RESET}")
        print(f"   Positive Months:        {self.positive_months_pct:.0f}%")
        print()
        
        # Regime breakdown (if available)
        if self.mr_regime_pct > 0 or self.trend_regime_pct > 0:
            print(f"{CYAN}{BOLD}🔄 REGIME BREAKDOWN{RESET}")
            print(f"   Mean Reversion:         {self.mr_regime_pnl:+.2f}% PnL ({self.mr_regime_pct:.0f}% of time)")
            print(f"   Trend Following:        {self.trend_regime_pnl:+.2f}% PnL ({self.trend_regime_pct:.0f}% of time)")
            print()
        
        print(f"{BOLD}═══════════════════════════════════════════════════════════════════════════{RESET}")
        print()
    
    def to_markdown(self, filepath: str) -> str:
        """Save report as Markdown file and return content."""
        
        # Monthly returns table
        monthly_table = ""
        if self.monthly_returns:
            # Group by year
            years = sorted(set(k.split('-')[0] for k in self.monthly_returns.keys()))
            monthly_table = "\n### Monthly Returns (%)\n\n"
            monthly_table += "| Year |"
            for m in range(1, 13):
                monthly_table += f" {['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][m-1]} |"
            monthly_table += " Total |\n"
            monthly_table += "|------|" + "---:|" * 13 + "\n"
            
            for year in years:
                monthly_table += f"| {year} |"
                year_total = 0.0
                for m in range(1, 13):
                    key = f"{year}-{m:02d}"
                    val = self.monthly_returns.get(key, None)
                    if val is not None:
                        monthly_table += f" {val:+.1f} |"
                        year_total += val
                    else:
                        monthly_table += " - |"
                monthly_table += f" **{year_total:+.1f}** |\n"
        
        content = f"""# Backtest Report: {self.symbol} {self.strategy_name}

**Period:** {self.start_date} → {self.end_date} ({self.duration_days:.0f} days)

---

## 💰 Performance Summary

| Metric | Value |
|--------|------:|
| Initial Capital | {self.initial_capital:.4f} |
| Final Capital | {self.final_capital:.4f} |
| **Total Return** | **{self.total_return_pct:+.2f}%** |
| CAGR | {self.cagr_pct:+.2f}% |

---

## 📊 Benchmark Comparison

| Benchmark | Return |
|-----------|-------:|
| HODL Quote ({self.quote_asset}) | 0.00% |
| HODL {self.base_asset} | {self.hodl_base_return_pct:+.2f}% |
| **✨ Alpha (Value Added)** | **{self.alpha_pct:+.2f}%** |

---

## 📉 Risk Metrics

| Metric | Value | Assessment |
|--------|------:|------------|
| Sharpe Ratio | {self.sharpe_ratio:.2f} | {"🟢 Good" if self.sharpe_ratio >= 1 else "🟡 Moderate" if self.sharpe_ratio >= 0.5 else "🔴 Poor"} |
| Sortino Ratio | {self.sortino_ratio:.2f} | {"🟢 Good" if self.sortino_ratio >= 1 else "🟡 Moderate"} |
| Calmar Ratio | {self.calmar_ratio:.2f} | {"🟢 Good" if self.calmar_ratio >= 1 else "🟡 Moderate"} |
| Max Drawdown | {self.max_drawdown_pct:.1f}% | {"🟢 Low" if self.max_drawdown_pct < 10 else "🟡 Moderate" if self.max_drawdown_pct < 20 else "🔴 High"} |
| Max DD Duration | {self.max_drawdown_duration_days:.0f} days | |
| Volatility (Ann.) | {self.volatility_ann_pct:.1f}% | |
| VaR (95%) | {self.var_95_pct:.2f}% | |
| CVaR (95%) | {self.cvar_95_pct:.2f}% | |

---

## 📈 Trading Analysis

| Metric | Value |
|--------|------:|
| Total Trades | {self.n_trades} |
| Win Rate | {self.win_rate_pct:.1f}% |
| Profit Factor | {self.profit_factor:.2f} |
| Time in Market | {self.time_in_market_pct:.1f}% |
| Avg Exposure | {self.avg_exposure_pct:.1f}% |
| Total Fees | {self.total_fees:.6f} |

---

## 📆 Period Performance

| Metric | Value |
|--------|------:|
| Best Month | {self.best_month_pct:+.1f}% |
| Worst Month | {self.worst_month_pct:+.1f}% |
| Positive Months | {self.positive_months_pct:.0f}% |

{monthly_table}

---

*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        
        with open(filepath, 'w') as f:
            f.write(content)
        
        return content
