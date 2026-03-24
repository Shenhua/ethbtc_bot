"""
Tests for Phase 7: Adapter failure hardening.

Covers:
- BinanceSpotAdapter.cancel(): suppresses "Unknown order" / -2011, raises on real errors
- BinanceFuturesAdapter.get_position(): raises ValueError on malformed positionAmt
"""
import pytest
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# BinanceSpotAdapter.cancel() tests
# ---------------------------------------------------------------------------

class TestSpotAdapterCancel:
    def _make_adapter(self, cancel_side_effect=None):
        """Build a BinanceSpotAdapter with a mocked Spot client."""
        from core.binance_adapter import BinanceSpotAdapter

        mock_client = MagicMock()
        if cancel_side_effect is not None:
            mock_client.cancel_order.side_effect = cancel_side_effect

        adapter = BinanceSpotAdapter.__new__(BinanceSpotAdapter)
        adapter.client = mock_client
        return adapter

    def test_cancel_success(self):
        """Successful cancel does not raise."""
        adapter = self._make_adapter()
        adapter.cancel("ETHBTC", "ORD-1")  # No exception
        adapter.client.cancel_order.assert_called_once_with(symbol="ETHBTC", orderId="ORD-1")

    def test_cancel_unknown_order_suppressed(self):
        """'Unknown order' error (order already gone) is silently suppressed."""
        adapter = self._make_adapter(
            cancel_side_effect=Exception("APIError: Unknown order")
        )
        # Should not raise
        adapter.cancel("ETHBTC", "ORD-2")

    def test_cancel_error_code_2011_suppressed(self):
        """Error code -2011 (unknown order) is silently suppressed."""
        adapter = self._make_adapter(
            cancel_side_effect=Exception("code=-2011, msg=Unknown order")
        )
        adapter.cancel("ETHBTC", "ORD-3")  # Should not raise

    def test_cancel_real_network_error_raises(self):
        """Network-level errors (not order-gone) propagate so caller can handle."""
        adapter = self._make_adapter(
            cancel_side_effect=ConnectionError("connection refused")
        )
        with pytest.raises(ConnectionError):
            adapter.cancel("ETHBTC", "ORD-4")

    def test_cancel_unexpected_api_error_raises(self):
        """Unexpected API errors (not order-gone) propagate."""
        adapter = self._make_adapter(
            cancel_side_effect=Exception("APIError: insufficient balance")
        )
        with pytest.raises(Exception, match="insufficient balance"):
            adapter.cancel("ETHBTC", "ORD-5")


# ---------------------------------------------------------------------------
# BinanceFuturesAdapter.get_position() tests
# ---------------------------------------------------------------------------

class TestFuturesAdapterGetPosition:
    def _make_adapter(self, position_risk_return=None, position_risk_side_effect=None):
        """Build a BinanceFuturesAdapter with a mocked UMFutures client."""
        from core.futures_adapter import BinanceFuturesAdapter

        mock_client = MagicMock()
        if position_risk_side_effect is not None:
            mock_client.get_position_risk.side_effect = position_risk_side_effect
        elif position_risk_return is not None:
            mock_client.get_position_risk.return_value = position_risk_return

        adapter = BinanceFuturesAdapter.__new__(BinanceFuturesAdapter)
        adapter.client = mock_client
        return adapter

    def test_get_position_normal(self):
        """Returns the correct position amount for a found symbol."""
        adapter = self._make_adapter(position_risk_return=[
            {"symbol": "ETHUSDT", "positionAmt": "0.5", "entryPrice": "2000"}
        ])
        result = adapter.get_position("ETHUSDT")
        assert result == pytest.approx(0.5)

    def test_get_position_not_found_returns_zero(self):
        """Returns 0.0 when symbol not in response list."""
        adapter = self._make_adapter(position_risk_return=[
            {"symbol": "BTCUSDT", "positionAmt": "1.0", "entryPrice": "50000"}
        ])
        result = adapter.get_position("ETHUSDT")
        assert result == 0.0

    def test_get_position_missing_position_amt_raises(self):
        """Missing positionAmt key raises ValueError with context."""
        adapter = self._make_adapter(position_risk_return=[
            {"symbol": "ETHUSDT", "entryPrice": "2000"}  # no positionAmt
        ])
        with pytest.raises(ValueError, match="Malformed position response"):
            adapter.get_position("ETHUSDT")

    def test_get_position_non_numeric_position_amt_raises(self):
        """Non-numeric positionAmt raises ValueError."""
        adapter = self._make_adapter(position_risk_return=[
            {"symbol": "ETHUSDT", "positionAmt": "not_a_number", "entryPrice": "2000"}
        ])
        with pytest.raises(ValueError, match="Malformed position response"):
            adapter.get_position("ETHUSDT")

    def test_get_position_api_error_propagates(self):
        """API-level errors (network, auth) still propagate unchanged."""
        adapter = self._make_adapter(
            position_risk_side_effect=ConnectionError("exchange unreachable")
        )
        with pytest.raises(ConnectionError):
            adapter.get_position("ETHUSDT")
