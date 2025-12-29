"""
Tests for core/services/data_service.py

Tests cover:
- Successful kline fetching and cleaning
- Empty data handling
- Duplicate removal
- Incomplete candle dropping
- Circuit breaker activation
- Exception handling
"""

import pytest
import pandas as pd
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timezone

from core.services.data_service import DataService
from core.resilience import CircuitBreaker, CircuitBreakerOpen


class TestDataServiceInit:
    """Test DataService initialization."""
    
    def test_init_stores_all_dependencies(self):
        """Verify all dependencies are properly stored."""
        adapter = Mock()
        alerter = Mock()
        cb = CircuitBreaker(max_failures=5, reset_timeout=60.0)
        
        svc = DataService(
            adapter=adapter,
            symbol="ETHBTC",
            interval="15m",
            circuit_breaker=cb,
            alerter=alerter
        )
        
        assert svc.adapter is adapter
        assert svc.symbol == "ETHBTC"
        assert svc.interval == "15m"
        assert svc.circuit_breaker is cb
        assert svc.alerter is alerter


class TestDataServiceGetClosedKlines:
    """Test the get_closed_klines method."""
    
    @pytest.fixture
    def mock_adapter(self):
        """Create a mock adapter with kline data."""
        adapter = Mock()
        return adapter
    
    @pytest.fixture
    def data_service(self, mock_adapter):
        """Create a DataService with mocks."""
        alerter = Mock()
        cb = CircuitBreaker(max_failures=5, reset_timeout=60.0)
        return DataService(
            adapter=mock_adapter,
            symbol="ETHBTC",
            interval="15m",
            circuit_breaker=cb,
            alerter=alerter
        )
    
    def test_successful_fetch_returns_dataframe(self, data_service, mock_adapter):
        """Test that successful fetch returns a cleaned DataFrame."""
        # Mock closed klines (close_time in the past)
        now = pd.Timestamp.now(tz="UTC")
        past = now - pd.Timedelta(hours=1)
        
        mock_adapter.get_klines.return_value = [
            {"open": "0.034", "high": "0.035", "low": "0.033", "close": "0.0345", 
             "volume": "100", "close_time": int(past.timestamp() * 1000)},
            {"open": "0.0345", "high": "0.036", "low": "0.034", "close": "0.0355",
             "volume": "150", "close_time": int((past + pd.Timedelta(minutes=15)).timestamp() * 1000)},
        ]
        
        with patch("core.services.data_service.retry_api_call", side_effect=lambda fn, *args, **kwargs: fn(*args)):
            df = data_service.get_closed_klines(limit=100)
        
        assert df is not None
        assert len(df) == 2
        assert "close" in df.columns
        assert df["close"].dtype == float
    
    def test_empty_klines_returns_none(self, data_service, mock_adapter):
        """Test that empty klines returns None."""
        mock_adapter.get_klines.return_value = []
        
        with patch("core.services.data_service.retry_api_call", side_effect=lambda fn, *args, **kwargs: fn(*args)):
            df = data_service.get_closed_klines()
        
        assert df is None
    
    def test_drops_incomplete_candle(self, data_service, mock_adapter):
        """Test that the currently open candle is dropped."""
        now = pd.Timestamp.now(tz="UTC")
        past = now - pd.Timedelta(hours=1)
        future = now + pd.Timedelta(minutes=10)  # Still open
        
        mock_adapter.get_klines.return_value = [
            {"open": "0.034", "high": "0.035", "low": "0.033", "close": "0.0345",
             "volume": "100", "close_time": int(past.timestamp() * 1000)},
            {"open": "0.0345", "high": "0.036", "low": "0.034", "close": "0.0355",
             "volume": "150", "close_time": int(future.timestamp() * 1000)},  # Incomplete
        ]
        
        with patch("core.services.data_service.retry_api_call", side_effect=lambda fn, *args, **kwargs: fn(*args)):
            df = data_service.get_closed_klines(limit=100)
        
        # Only the closed candle should remain
        assert df is not None
        assert len(df) == 1
    
    def test_removes_duplicates(self, data_service, mock_adapter):
        """Test that duplicate candles are removed."""
        now = pd.Timestamp.now(tz="UTC")
        past = now - pd.Timedelta(hours=1)
        ts = int(past.timestamp() * 1000)
        
        mock_adapter.get_klines.return_value = [
            {"open": "0.034", "high": "0.035", "low": "0.033", "close": "0.0345",
             "volume": "100", "close_time": ts},
            {"open": "0.034", "high": "0.035", "low": "0.033", "close": "0.0346",
             "volume": "101", "close_time": ts},  # Duplicate timestamp
        ]
        
        with patch("core.services.data_service.retry_api_call", side_effect=lambda fn, *args, **kwargs: fn(*args)):
            df = data_service.get_closed_klines(limit=100)
        
        # Duplicates should be removed (keep last)
        assert df is not None
        assert len(df) == 1
        assert df["close"].iloc[0] == 0.0346  # Last value kept
    
    def test_circuit_breaker_open_returns_none_and_alerts(self, data_service):
        """Test that CircuitBreakerOpen triggers alert and returns None."""
        with patch("core.services.data_service.retry_api_call", 
                   side_effect=CircuitBreakerOpen("Test CB open")):
            df = data_service.get_closed_klines()
        
        assert df is None
        data_service.alerter.send.assert_called_once()
        assert "CIRCUIT BREAKER" in data_service.alerter.send.call_args[0][0]
    
    def test_exception_returns_none(self, data_service):
        """Test that exceptions return None without crashing."""
        with patch("core.services.data_service.retry_api_call",
                   side_effect=Exception("Network error")):
            df = data_service.get_closed_klines()
        
        assert df is None
