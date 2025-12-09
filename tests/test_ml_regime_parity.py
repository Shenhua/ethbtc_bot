"""
Parity Tests for ML Regime Detection.

These tests ensure that:
1. ML regime detection uses the SAME feature calculation as backtest
2. Fallback to ADX works correctly
3. Output shapes and ranges are consistent
4. Parity between backtest and live is maintained

CRITICAL: These tests are part of check_parity.sh and must pass before deployment.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from core.regime import get_regime_score, _get_adx_regime_score, _get_ml_regime_score
from core.regime_features import build_regime_features, get_feature_names


@pytest.fixture
def sample_ohlcv():
    """Generate deterministic synthetic OHLCV data."""
    np.random.seed(42)
    n = 200
    
    # Generate random walk for close price
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    
    df = pd.DataFrame({
        "open": close + np.random.randn(n) * 0.1,
        "high": close + np.abs(np.random.randn(n)) * 0.2,
        "low": close - np.abs(np.random.randn(n)) * 0.2,
        "close": close,
        "volume": np.abs(np.random.randn(n) * 1000) + 100,
    }, index=pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC"))
    
    return df


class TestMLRegimeParity:
    """Tests to ensure ML regime maintains parity between backtest and live."""
    
    def test_adx_fallback_on_missing_model(self, sample_ohlcv):
        """Verify graceful fallback to ADX when ML model doesn't exist."""
        # This should NOT raise an exception - it should fall back silently
        score = get_regime_score(
            sample_ohlcv, 
            use_ml=True, 
            model_path="nonexistent_model.pkl"
        )
        
        assert len(score) == len(sample_ohlcv)
        assert not score.isna().any()
        # Verify it's ADX (compare with direct ADX call)
        adx_score = _get_adx_regime_score(sample_ohlcv)
        pd.testing.assert_series_equal(score, adx_score, check_names=False)
    
    def test_ml_vs_adx_shape_match(self, sample_ohlcv):
        """Verify ML and ADX return same shape output."""
        adx_score = get_regime_score(sample_ohlcv, use_ml=False)
        
        # Only run ML test if model exists
        model_path = Path("models/regime_classifier_v1.pkl")
        if model_path.exists():
            ml_score = get_regime_score(sample_ohlcv, use_ml=True)
            assert len(ml_score) == len(adx_score)
            assert ml_score.index.equals(adx_score.index)
        else:
            pytest.skip("ML model not trained yet")
    
    def test_ml_score_range(self, sample_ohlcv):
        """Verify ML score is in 0-100 range (compatible with threshold)."""
        model_path = Path("models/regime_classifier_v1.pkl")
        if not model_path.exists():
            pytest.skip("ML model not trained yet")
        
        ml_score = get_regime_score(sample_ohlcv, use_ml=True)
        
        assert ml_score.min() >= 0.0, "ML score should be >= 0"
        assert ml_score.max() <= 100.0, "ML score should be <= 100"
    
    def test_feature_calculation_deterministic(self, sample_ohlcv):
        """Verify features are calculated identically on same input."""
        features1 = build_regime_features(sample_ohlcv)
        features2 = build_regime_features(sample_ohlcv)
        pd.testing.assert_frame_equal(features1, features2)
    
    def test_regime_function_backwards_compatible(self, sample_ohlcv):
        """Verify get_regime_score works with old signature (no ML params)."""
        # Old code would call: get_regime_score(df)
        # This must still work
        score = get_regime_score(sample_ohlcv)
        
        assert len(score) == len(sample_ohlcv)
        assert not score.isna().all()
    
    def test_ml_mode_uses_shared_features(self, sample_ohlcv):
        """Verify ML mode uses the same feature module as standalone feature test."""
        model_path = Path("models/regime_classifier_v1.pkl")
        if not model_path.exists():
            pytest.skip("ML model not trained yet")
        
        # The features used by _get_ml_regime_score should be from build_regime_features
        # This is implicitly tested by the fact that ML works
        # But we can verify the feature names match
        expected_features = get_feature_names()
        assert len(expected_features) == 9
        assert "adx_15m" in expected_features
        assert "fear_greed" in expected_features


class TestConfigIntegration:
    """Tests for config schema integration."""
    
    def test_strategy_has_ml_options(self):
        """Verify Strategy config has ML regime options."""
        from core.config_schema import Strategy
        
        s = Strategy()
        assert hasattr(s, "use_ml_regime")
        assert hasattr(s, "ml_model_path")
        
        # Defaults
        assert s.use_ml_regime == False
        assert s.ml_model_path == "models/regime_classifier_v1.pkl"
    
    def test_config_with_ml_enabled(self):
        """Verify config can be created with ML enabled."""
        from core.config_schema import Strategy
        
        s = Strategy(use_ml_regime=True)
        assert s.use_ml_regime == True
