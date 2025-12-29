"""
Tests for core/services/order_service.py

Tests cover:
- Account state fetching for Spot and Futures
- Weight calculations
- Position staleness detection
- Circuit breaker handling
- Metrics updates
"""

import pytest
import pandas as pd
from unittest.mock import Mock, MagicMock, patch, PropertyMock

from core.services.order_service import OrderService
from core.resilience import CircuitBreaker, CircuitBreakerOpen


class TestOrderServiceInit:
    """Test OrderService initialization."""
    
    def test_init_stores_all_dependencies(self):
        """Verify all dependencies are properly stored."""
        adapter = Mock()
        alerter = Mock()
        cb = CircuitBreaker(max_failures=5, reset_timeout=60.0)
        
        svc = OrderService(
            adapter=adapter,
            symbol="ETHBTC",
            quote_asset="BTC",
            is_futures=False,
            leverage=1,
            circuit_breaker=cb,
            alerter=alerter,
            instance_name="test_ethbtc"
        )
        
        assert svc.adapter is adapter
        assert svc.symbol == "ETHBTC"
        assert svc.quote_asset == "BTC"
        assert svc.is_futures is False
        assert svc.leverage == 1
        assert svc.instance_name == "test_ethbtc"


class TestOrderServiceSpotMode:
    """Test OrderService in Spot mode."""
    
    @pytest.fixture
    def spot_service(self):
        """Create a Spot mode OrderService."""
        adapter = Mock()
        adapter.get_account_balance.side_effect = lambda asset: {
            "BTC": 1.0,
            "ETH": 10.0
        }.get(asset, 0.0)
        
        alerter = Mock()
        cb = CircuitBreaker(max_failures=5, reset_timeout=60.0)
        
        return OrderService(
            adapter=adapter,
            symbol="ETHBTC",
            quote_asset="BTC",
            is_futures=False,
            leverage=1,
            circuit_breaker=cb,
            alerter=alerter,
            instance_name="test_ethbtc"
        )
    
    def test_get_account_state_returns_correct_values(self, spot_service):
        """Test that spot account state is calculated correctly."""
        bar_dt = pd.Timestamp.now(tz="UTC")
        price = 0.034  # 1 ETH = 0.034 BTC
        state = {}
        
        result = spot_service.get_account_state(state, bar_dt, price)
        
        # quote_bal = 1.0 BTC
        # current_position = 10.0 ETH
        # W = 1.0 + (10.0 * 0.034) = 1.34 BTC
        # cur_w = (10.0 * 0.034) / 1.34 ≈ 0.254
        
        assert result["quote_bal"] == 1.0
        assert result["current_position"] == 10.0
        assert abs(result["W"] - 1.34) < 0.01
        assert abs(result["cur_w"] - 0.254) < 0.01
    
    def test_zero_balance_returns_zero_weight(self):
        """Test that zero balance returns zero weight without division error."""
        adapter = Mock()
        adapter.get_account_balance.return_value = 0.0
        
        svc = OrderService(
            adapter=adapter,
            symbol="ETHBTC",
            quote_asset="BTC",
            is_futures=False,
            leverage=1,
            circuit_breaker=CircuitBreaker(max_failures=5, reset_timeout=60.0),
            alerter=Mock(),
            instance_name="test"
        )
        
        bar_dt = pd.Timestamp.now(tz="UTC")
        state = {}
        
        result = svc.get_account_state(state, bar_dt, price=0.034)
        
        assert result["W"] == 0.0
        assert result["cur_w"] == 0.0


class TestOrderServiceFuturesMode:
    """Test OrderService in Futures mode."""
    
    @pytest.fixture
    def futures_service(self):
        """Create a Futures mode OrderService."""
        adapter = Mock()
        adapter.get_account_balance.return_value = 1000.0  # USDT
        adapter.get_position.return_value = 0.5  # 0.5 BTC position
        adapter.client = Mock()
        adapter.client.account.return_value = {
            "totalMaintMargin": "100",
            "totalMarginBalance": "1000"
        }
        adapter.client.get_position_risk.return_value = [
            {"symbol": "BTCUSDT", "liquidationPrice": "50000", "markPrice": "88000"}
        ]
        
        alerter = Mock()
        cb = CircuitBreaker(max_failures=5, reset_timeout=60.0)
        
        return OrderService(
            adapter=adapter,
            symbol="BTCUSDT",
            quote_asset="USDT",
            is_futures=True,
            leverage=10,
            circuit_breaker=cb,
            alerter=alerter,
            instance_name="test_btcusdt"
        )
    
    @patch("core.services.order_service.retry_api_call")
    @patch("core.services.order_service.mark_futures_risk")
    def test_futures_account_state(self, mock_risk, mock_retry, futures_service):
        """Test that futures account state includes risk metrics."""
        mock_retry.side_effect = lambda fn, *args, **kwargs: fn(*args)
        
        bar_dt = pd.Timestamp.now(tz="UTC")
        price = 88000.0
        state = {}
        
        result = futures_service.get_account_state(state, bar_dt, price)
        
        assert result["quote_bal"] == 1000.0
        assert result["W"] == 1000.0  # In futures, W = margin balance
        assert result["margin_util_pct"] == 10.0  # 100/1000 * 100
        
        # Verify risk metrics were set
        mock_risk.assert_called_once()
    
    @patch("core.services.order_service.retry_api_call")
    def test_position_staleness_on_circuit_breaker(self, mock_retry, futures_service):
        """Test that position staleness is flagged on circuit breaker."""
        mock_retry.side_effect = CircuitBreakerOpen("CB Open")
        
        bar_dt = pd.Timestamp.now(tz="UTC")
        state = {"last_known_position": 0.3}
        
        result = futures_service.get_account_state(state, bar_dt, price=88000.0)
        
        assert result["position_stale"] is True
        assert result["current_position"] == 0.3  # fallback to last known


class TestOrderServiceExceptionHandling:
    """Test OrderService exception handling."""
    
    def test_critical_error_returns_stale_state(self):
        """Test that critical errors return stale position state."""
        adapter = Mock()
        adapter.get_account_balance.side_effect = Exception("API Down")
        
        svc = OrderService(
            adapter=adapter,
            symbol="ETHBTC",
            quote_asset="BTC",
            is_futures=False,
            leverage=1,
            circuit_breaker=CircuitBreaker(max_failures=5, reset_timeout=60.0),
            alerter=Mock(),
            instance_name="test"
        )
        
        result = svc.get_account_state({}, pd.Timestamp.now(tz="UTC"), 0.034)
        
        assert result["position_stale"] is True
        assert result["W"] == 0.0
