#!/usr/bin/env python3
"""
Optuna-based ML Regime Optimizer.

Optimizes ML regime detection hyperparameters for TRADING PERFORMANCE (Sharpe ratio),
not just classification accuracy. This is critical because high accuracy does not
guarantee profitable trading.

Key optimization targets:
1. Maximize Sharpe ratio
2. Minimize regime switching (penalty for excessive trades)
3. Ensure robustness via walk-forward validation

Usage:
    python tools/optimize_ml_regime.py \
        --data data/raw/ETHBTC_15m_2021-2025_vision.csv \
        --funding data/raw/ETHUSDT_funding_2021-2025.csv \
        --fear-greed data/raw/fear_greed_index_2021-2025.csv \
        --config configs/prod_meta_live.json \
        --n-trials 50 \
        --output models/regime_classifier_optimized.pkl
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import joblib
import logging
import json
import warnings
import os
import sys

# --- MAGIC PATH FIX ---
# Calculate the root directory (one level up from 'tools')
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
# ----------------------

warnings.filterwarnings("ignore")

try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError:
    print("ERROR: optuna not installed. Run: pip install optuna")
    exit(1)

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

from core.regime_features import build_regime_features, get_feature_names
from core.regime import get_regime_score
from core.ethbtc_accum_bot import load_vision_csv, EthBtcStrategy, StratParams, Backtester, FeeParams
from core.trend_strategy import TrendStrategy, TrendParams
from core.meta_strategy import MetaStrategy
from core.config_schema import load_config

logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("ml_optimizer")


class MLRegimeOptimizer:
    """Optuna-based optimizer for ML regime detection."""
    
    def __init__(
        self,
        ohlcv: pd.DataFrame,
        funding: pd.Series,
        fear_greed: pd.Series,
        config_path: str,
        train_start: str = None,
        train_end: str = None,
        test_start: str = None,
        test_end: str = None,
        validation_ratio: float = 0.2,
    ):
        self.ohlcv = ohlcv
        self.funding = funding
        self.fear_greed = fear_greed
        self.config = load_config(config_path)
        self.validation_ratio = validation_ratio
        
        # Pre-compute features once (they don't change with hyperparams)
        log.info("Pre-computing features...")
        self.features = build_regime_features(ohlcv, funding, fear_greed)
        log.info(f"Features shape: {self.features.shape}")
        
        # Split data based on date ranges OR validation_ratio
        if train_end and test_start:
            # Use explicit date ranges for proper out-of-sample testing
            train_mask = (self.ohlcv.index >= train_start) & (self.ohlcv.index <= train_end) if train_start else (self.ohlcv.index <= train_end)
            test_mask = (self.ohlcv.index >= test_start) & (self.ohlcv.index <= test_end) if test_end else (self.ohlcv.index >= test_start)
            
            self.train_ohlcv = self.ohlcv[train_mask]
            self.val_ohlcv = self.ohlcv[test_mask]
            self.train_features = self.features[train_mask]
            self.val_features = self.features[test_mask]
            
            log.info(f"Train period: {train_start or 'start'} to {train_end}")
            log.info(f"Test period: {test_start} to {test_end or 'end'}")
        else:
            # Fall back to simple ratio split
            split_idx = int(len(self.ohlcv) * (1 - validation_ratio))
            self.train_ohlcv = self.ohlcv.iloc[:split_idx]
            self.val_ohlcv = self.ohlcv.iloc[split_idx:]
            self.train_features = self.features.iloc[:split_idx]
            self.val_features = self.features.iloc[split_idx:]
        
        log.info(f"Train: {len(self.train_ohlcv)} bars, Test/Val: {len(self.val_ohlcv)} bars")
        
        # Cache for ADX baseline
        self._baseline_sharpe = None
    
    def get_baseline_sharpe(self) -> float:
        """Calculate Sharpe ratio for ADX baseline (no ML)."""
        if self._baseline_sharpe is not None:
            return self._baseline_sharpe
            
        log.info("Computing ADX baseline Sharpe...")
        sharpe = self._run_backtest(use_ml=False, ml_threshold=25.0)
        self._baseline_sharpe = sharpe
        log.info(f"ADX Baseline Sharpe: {sharpe:.4f}")
        return sharpe
    
    def create_labels(
        self,
        ohlcv: pd.DataFrame,
        threshold: float,
        lookahead: int,
    ) -> pd.Series:
        """Create binary labels from future ADX regime score."""
        regime_score = get_regime_score(ohlcv)
        future_score = regime_score.shift(-lookahead)
        labels = (future_score > threshold).astype(int)
        return labels
    
    def train_model(
        self,
        lookahead: int,
        adx_threshold: float,
        n_estimators: int,
        max_depth: int,
        min_samples_split: int,
    ) -> tuple:
        """Train a model with given hyperparameters."""
        # Create labels for training data
        labels = self.create_labels(self.train_ohlcv, adx_threshold, lookahead)
        
        # Align features and labels
        common_idx = self.train_features.index.intersection(labels.dropna().index)
        X = self.train_features.loc[common_idx]
        y = labels.loc[common_idx]
        
        if len(X) < 1000:
            raise ValueError(f"Too few training samples: {len(X)}")
        
        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=50,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_scaled, y)
        
        return model, scaler
    
    def _create_ml_regime_scorer(
        self,
        model,
        scaler,
        ml_threshold: float,
    ):
        """Create a function that returns ML-based regime scores."""
        def ml_regime_score(df: pd.DataFrame) -> pd.Series:
            # Build features
            features = build_regime_features(df, self.funding, self.fear_greed)
            X_scaled = scaler.transform(features)
            
            # Predict probabilities
            probs = model.predict_proba(X_scaled)[:, 1] * 100  # Scale to 0-100
            
            return pd.Series(probs, index=df.index)
        
        return ml_regime_score
    
    def _run_backtest(
        self,
        use_ml: bool = False,
        ml_threshold: float = 25.0,
        model=None,
        scaler=None,
    ) -> float:
        """Run backtest on validation data and return Sharpe ratio."""
        try:
            # Build strategy
            cfg = self.config
            
            # Mean Reversion params
            mr_base = cfg.strategy.model_dump()
            mr_overrides = cfg.strategy.mean_reversion_overrides or {}
            mr_merged = {**mr_base, **mr_overrides}
            
            mr_params = StratParams(
                trend_kind=mr_merged.get("trend_kind", "ema_cross"),
                trend_lookback=mr_merged.get("trend_lookback", 100),
                flip_band_entry=mr_merged.get("flip_band_entry", 0.05),
                flip_band_exit=mr_merged.get("flip_band_exit", 0.02),
                vol_window=mr_merged.get("vol_window", 45),
                vol_adapt_k=mr_merged.get("vol_adapt_k", 0.0075),
                target_vol=mr_merged.get("target_vol", 0.5),
                min_mult=mr_merged.get("min_mult", 0.7),
                max_mult=mr_merged.get("max_mult", 1.5),
                fast_period=mr_merged.get("fast_period", 50),
                slow_period=mr_merged.get("slow_period", 200),
                ma_type=mr_merged.get("ma_type", "ema"),
                gate_window_days=mr_merged.get("gate_window_days", 30),
                gate_roc_threshold=mr_merged.get("gate_roc_threshold", 0.0),
                step_allocation=mr_merged.get("step_allocation", 0.33),
                max_position=mr_merged.get("max_position", 1.0),
            )
            
            # Trend params
            tr_overrides = cfg.strategy.trend_overrides or {}
            tr_merged = {**mr_base, **tr_overrides}
            
            tr_params = TrendParams(
                fast_period=tr_merged.get("fast_period", 50),
                slow_period=tr_merged.get("slow_period", 200),
                ma_type=tr_merged.get("ma_type", "ema"),
                step_allocation=tr_merged.get("step_allocation", 1.0),
                max_position=tr_merged.get("max_position", 1.0),
            )
            
            # Create strategy - use REAL ML path for parity with backtest
            # We need to save the model to a temp file so get_regime_score can load it
            if use_ml and model is not None:
                import tempfile
                import os
                
                # Save model and scaler to temp files
                temp_dir = tempfile.mkdtemp()
                temp_model_path = os.path.join(temp_dir, "temp_classifier.pkl")
                temp_scaler_path = os.path.join(temp_dir, "temp_scaler.pkl")
                
                import joblib
                joblib.dump(model, temp_model_path)
                joblib.dump(scaler, temp_scaler_path)
                
                strategy = MetaStrategy(
                    mr_params, tr_params,
                    adx_threshold=mr_merged.get("adx_threshold", 25.0),  # Fallback threshold
                    use_ml_regime=True,  # Use REAL ML path
                    ml_model_path=temp_model_path,
                    ml_threshold=ml_threshold,
                )
            else:
                temp_dir = None
                strategy = MetaStrategy(
                    mr_params, tr_params,
                    adx_threshold=mr_merged.get("adx_threshold", 25.0),
                    use_ml_regime=False,
                )
            
            # Run backtest on validation data
            fee_params = FeeParams(
                maker_fee=cfg.fees.maker_fee,
                taker_fee=cfg.fees.taker_fee,
                slippage_bps=cfg.fees.slippage_bps,
                bnb_discount=cfg.fees.bnb_discount,
                pay_fees_in_bnb=cfg.fees.pay_fees_in_bnb,
            )
            
            bt = Backtester(fee_params)
            
            # Align fear_greed for validation period
            val_fear_greed = None
            if self.fear_greed is not None:
                val_fear_greed = self.fear_greed
            
            try:
                result = bt.simulate(
                    self.val_ohlcv["close"],
                    strategy,
                    funding_series=self.funding,
                    fear_greed_series=val_fear_greed,  # Pass fear_greed for ML!
                    full_df=self.val_ohlcv,
                    initial_btc=1.0,
                )
            finally:
                # Clean up temp files
                if temp_dir:
                    import shutil
                    shutil.rmtree(temp_dir, ignore_errors=True)
            
            # Calculate Sharpe from backtest results
            # Result structure: {'summary': {...}, 'portfolio': df, 'balances': df, ...}
            if "balances" in result:
                balances = result["balances"]
                if "btc" in balances.columns:
                    # Calculate total wealth (btc + eth * price)
                    btc = balances["btc"].values
                    eth = balances["eth"].values
                    prices = self.val_ohlcv["close"].values[:len(btc)]
                    wealth = btc + eth * prices
                    
                    if len(wealth) > 1:
                        returns = pd.Series(wealth).pct_change().dropna()
                        if len(returns) > 10 and returns.std() > 0:
                            sharpe = (returns.mean() / returns.std()) * np.sqrt(252 * 96)  # 15m bars
                            return sharpe
            
            return -1.0  # Failed backtest
            
        except Exception as e:
            log.warning(f"Backtest failed: {e}")
            return -1.0
    
    def objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function."""
        # Sample hyperparameters - WIDER RANGES for more exploration
        lookahead = trial.suggest_int("lookahead_bars", 4, 64)  # More options
        adx_threshold = trial.suggest_float("adx_threshold", 10.0, 40.0)  # Wider, no step
        n_estimators = trial.suggest_int("n_estimators", 30, 300)  # Wider
        max_depth = trial.suggest_int("max_depth", 3, 25)  # Wider
        min_samples_split = trial.suggest_int("min_samples_split", 2, 50)
        ml_threshold = trial.suggest_float("ml_threshold", 20.0, 80.0)  # Much wider
        
        try:
            # Train model
            model, scaler = self.train_model(
                lookahead=lookahead,
                adx_threshold=adx_threshold,
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
            )
            
            # Run backtest with ML
            sharpe = self._run_backtest(
                use_ml=True,
                ml_threshold=ml_threshold,
                model=model,
                scaler=scaler,
            )
            
            # Store model in trial for later retrieval
            trial.set_user_attr("sharpe", sharpe)
            
            return sharpe
            
        except Exception as e:
            log.warning(f"Trial failed: {e}")
            return -10.0  # Penalize failed trials
    
    def optimize(
        self,
        n_trials: int = 50,
        timeout: int = None,
        n_jobs: int = 1,
        storage: str = None,
        study_name: str = "ml_regime_optimization",
    ) -> optuna.Study:
        """Run Optuna optimization."""
        log.info(f"Starting optimization with {n_trials} trials...")
        
        sampler = TPESampler(n_startup_trials=10)  # Random exploration first, no fixed seed
        
        # Use persistent storage if provided, otherwise in-memory
        if storage:
            study = optuna.create_study(
                direction="maximize",
                sampler=sampler,
                study_name=study_name,
                storage=storage,
                load_if_exists=True,
            )
            print(f"   📦 Using persistent storage: {storage}")
            print(f"   📊 Study '{study_name}' has {len(study.trials)} existing trials")
        else:
            study = optuna.create_study(
                direction="maximize",
                sampler=sampler,
                study_name=study_name,
            )
        
        # Add baseline as first trial
        baseline_sharpe = self.get_baseline_sharpe()
        
        study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            show_progress_bar=True,
        )
        
        return study
    
    def train_best_model(self, best_params: dict) -> tuple:
        """Train final model with best parameters on all data."""
        log.info("Training final model with best parameters...")
        
        # Create labels for ALL data
        labels = self.create_labels(
            self.ohlcv,
            threshold=best_params["adx_threshold"],
            lookahead=best_params["lookahead_bars"],
        )
        
        # Align
        common_idx = self.features.index.intersection(labels.dropna().index)
        X = self.features.loc[common_idx]
        y = labels.loc[common_idx]
        
        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train
        model = RandomForestClassifier(
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"],
            min_samples_split=best_params["min_samples_split"],
            min_samples_leaf=50,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_scaled, y)
        
        return model, scaler, best_params


def load_data(data_path: str, funding_path: str = None, fear_greed_path: str = None):
    """Load OHLCV, funding, and fear/greed data."""
    df = load_vision_csv(data_path)
    
    funding = None
    if funding_path:
        funding_df = pd.read_csv(funding_path, index_col=0, parse_dates=True)
        if "fundingRate" in funding_df.columns:
            funding = funding_df["fundingRate"]
        elif "rate" in funding_df.columns:
            funding = funding_df["rate"]
        else:
            funding = funding_df.iloc[:, 0]
        if not isinstance(funding.index, pd.DatetimeIndex):
            funding.index = pd.to_datetime(funding.index, format="ISO8601", utc=True)
        elif funding.index.tz is None:
            funding.index = funding.index.tz_localize("UTC")
    
    fear_greed = None
    if fear_greed_path:
        fg_df = pd.read_csv(fear_greed_path, index_col=0, parse_dates=True)
        fear_greed = fg_df["value"] if "value" in fg_df.columns else fg_df.iloc[:, 0]
    
    return df, funding, fear_greed


def main():
    parser = argparse.ArgumentParser(description="Optimize ML Regime Classifier with Optuna")
    
    parser.add_argument("--data", required=True, help="Path to OHLCV CSV file")
    parser.add_argument("--funding", default=None, help="Path to funding rate CSV")
    parser.add_argument("--fear-greed", default=None, help="Path to Fear & Greed CSV")
    parser.add_argument("--config", required=True, help="Path to strategy config JSON")
    parser.add_argument("--output", default="models/regime_classifier_optimized.pkl",
                        help="Output path for optimized model")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout in seconds")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs (1 = sequential)")
    parser.add_argument("--storage", default="sqlite:///data/db/optuna.db",
                        help="Optuna storage URL (default: sqlite:///data/db/optuna.db)")
    parser.add_argument("--study-name", default="ml_regime_optimization",
                        help="Optuna study name (for resuming)")
    
    # Date range arguments
    parser.add_argument("--train-start", default=None, help="Training period start date (YYYY-MM-DD)")
    parser.add_argument("--train-end", default=None, help="Training period end date (YYYY-MM-DD)")
    parser.add_argument("--test-start", default=None, help="Test period start date (YYYY-MM-DD)")
    parser.add_argument("--test-end", default=None, help="Test period end date (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("ML Regime Optimizer (Optuna)")
    print("=" * 60)
    
    # Load data
    print("\n📊 Loading data...")
    df, funding, fear_greed = load_data(args.data, args.funding, args.fear_greed)
    print(f"   OHLCV: {len(df)} bars")
    print(f"   Funding: {'✅' if funding is not None else '❌'}")
    print(f"   Fear/Greed: {'✅' if fear_greed is not None else '❌'}")
    
    # Create optimizer
    print("\n🔧 Initializing optimizer...")
    
    # Show date ranges if specified
    if args.train_end and args.test_start:
        print(f"   Train period: {args.train_start or 'start'} → {args.train_end}")
        print(f"   Test period: {args.test_start} → {args.test_end or 'end'}")
    else:
        print("   Using 80/20 train/validation split")
    
    optimizer = MLRegimeOptimizer(
        ohlcv=df,
        funding=funding,
        fear_greed=fear_greed,
        config_path=args.config,
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
        validation_ratio=0.2,
    )
    
    # Get baseline
    print("\n📈 Computing ADX baseline...")
    baseline = optimizer.get_baseline_sharpe()
    print(f"   ADX Baseline Sharpe: {baseline:.4f}")
    
    # Run optimization
    print(f"\n🎯 Running {args.n_trials} optimization trials...")
    study = optimizer.optimize(
        n_trials=args.n_trials,
        timeout=args.timeout,
        n_jobs=args.n_jobs,
        storage=args.storage,
        study_name=args.study_name,
    )
    
    # Results
    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)
    
    best = study.best_trial
    print(f"\n✨ Best Trial: #{best.number}")
    print(f"   Sharpe Ratio: {best.value:.4f}")
    print(f"   vs ADX Baseline: {best.value - baseline:+.4f}")
    print(f"\n   Parameters:")
    for k, v in best.params.items():
        print(f"      {k}: {v}")
    
    # Train final model
    if best.value > baseline:
        print("\n🏆 ML outperformed ADX! Training final model...")
        model, scaler, params = optimizer.train_best_model(best.params)
        
        # Save
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(model, output_path)
        print(f"   Model saved: {output_path}")
        
        scaler_path = output_path.parent / output_path.name.replace("classifier", "scaler")
        joblib.dump(scaler, scaler_path)
        print(f"   Scaler saved: {scaler_path}")
        
        # Save params
        params_path = output_path.parent / "best_ml_params.json"
        with open(params_path, "w") as f:
            json.dump(params, f, indent=2)
        print(f"   Params saved: {params_path}")
        
        print("\n✅ Optimization complete! Use these settings in your config:")
        print(f'   "use_ml_regime": true,')
        print(f'   "ml_model_path": "{output_path}",')
        print(f'   "adx_threshold": {params["ml_threshold"]}')
    else:
        print("\n⚠️ ML did not outperform ADX baseline.")
        print("   Recommendation: Keep using ADX-based regime detection.")
        print(f"\n   Best ML Sharpe: {best.value:.4f}")
        print(f"   ADX Baseline:   {baseline:.4f}")
    
    # Save study results
    results_path = Path(args.output).parent / "optimization_results.json"
    results = {
        "baseline_sharpe": baseline,
        "best_sharpe": best.value,
        "best_params": best.params,
        "n_trials": len(study.trials),
        "timestamp": datetime.now().isoformat(),
    }
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n📋 Results saved: {results_path}")


if __name__ == "__main__":
    main()
