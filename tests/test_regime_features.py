"""
Unit tests for regime_features.py.

These tests verify:
1. Feature calculation is deterministic (same input = same output)
2. Output shape matches input
3. No NaN values in output
4. Feature values are in expected ranges
"""

import pytest
import pandas as pd
import numpy as np
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


@pytest.fixture
def sample_funding():
    """Generate synthetic funding rate data."""
    np.random.seed(42)
    n = 200
    
    # Funding rate typically between -0.1% and 0.1%
    funding = pd.Series(
        np.random.randn(n) * 0.01,
        index=pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC"),
        name="funding_rate"
    )
    return funding


@pytest.fixture
def sample_fear_greed():
    """Generate synthetic Fear & Greed data (daily)."""
    np.random.seed(42)
    n = 14  # 2 weeks of daily data
    
    fg = pd.Series(
        np.random.randint(20, 80, n),
        index=pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC"),
        name="fear_greed"
    )
    return fg


class TestBuildRegimeFeatures:
    """Tests for build_regime_features function."""
    
    def test_output_shape_matches_input(self, sample_ohlcv):
        """Verify output DataFrame has same number of rows as input."""
        features = build_regime_features(sample_ohlcv)
        assert len(features) == len(sample_ohlcv)
    
    def test_all_feature_columns_present(self, sample_ohlcv):
        """Verify all expected feature columns are present."""
        features = build_regime_features(sample_ohlcv)
        expected = get_feature_names()
        assert list(features.columns) == expected
    
    def test_no_nan_values(self, sample_ohlcv):
        """Verify no NaN values in output (all should be filled)."""
        features = build_regime_features(sample_ohlcv)
        assert features.isna().sum().sum() == 0
    
    def test_deterministic_calculation(self, sample_ohlcv):
        """Verify same input produces identical output."""
        features1 = build_regime_features(sample_ohlcv)
        features2 = build_regime_features(sample_ohlcv)
        pd.testing.assert_frame_equal(features1, features2)
    
    def test_with_funding_data(self, sample_ohlcv, sample_funding):
        """Verify features work with funding data."""
        features = build_regime_features(sample_ohlcv, funding=sample_funding)
        
        # Funding columns should have non-zero values
        assert (features["funding_rate"] != 0).any()
    
    def test_with_fear_greed_data(self, sample_ohlcv, sample_fear_greed):
        """Verify features work with Fear & Greed data."""
        features = build_regime_features(sample_ohlcv, fear_greed=sample_fear_greed)
        
        # Should have forward-filled values (not all 50)
        assert (features["fear_greed"] != 50.0).any()
    
    def test_adx_range(self, sample_ohlcv):
        """Verify ADX is in expected range (0-100)."""
        features = build_regime_features(sample_ohlcv)
        assert features["adx_15m"].min() >= 0
        assert features["adx_15m"].max() <= 100
    
    def test_rsi_range(self, sample_ohlcv):
        """Verify RSI is in expected range (0-100)."""
        features = build_regime_features(sample_ohlcv)
        assert features["rsi_14"].min() >= 0
        assert features["rsi_14"].max() <= 100
    
    def test_fear_greed_range(self, sample_ohlcv, sample_fear_greed):
        """Verify Fear & Greed is in expected range (0-100)."""
        features = build_regime_features(sample_ohlcv, fear_greed=sample_fear_greed)
        assert features["fear_greed"].min() >= 0
        assert features["fear_greed"].max() <= 100


class TestGetFeatureNames:
    """Tests for get_feature_names function."""
    
    def test_returns_list(self):
        """Verify returns a list."""
        names = get_feature_names()
        assert isinstance(names, list)
    
    def test_correct_count(self):
        """Verify correct number of features."""
        names = get_feature_names()
        assert len(names) == 9
    
    def test_no_duplicates(self):
        """Verify no duplicate feature names."""
        names = get_feature_names()
        assert len(names) == len(set(names))
