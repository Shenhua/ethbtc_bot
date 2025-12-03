import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# --- MOCK MISSING DEPENDENCIES ---
# We mock these BEFORE importing core modules so the test runs 
# even if binance-connector is not installed locally.
sys.modules["binance.um_futures"] = MagicMock()
sys.modules["binance.spot"] = MagicMock()
sys.modules["binance.error"] = MagicMock()
# ---------------------------------

# Adjust path to import core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.futures_adapter import BinanceFuturesAdapter
import core.metrics as metrics

class TestFixes(unittest.TestCase):
    def test_hedge_mode_logic(self):
        """
        Verify that get_position correctly sums Long and Short positions
        to support Hedge Mode.
        """
        # Mock the client
        mock_client = MagicMock()
        adapter = BinanceFuturesAdapter(mock_client)
        
        # Scenario 1: Hedge Mode (Long 1.5, Short -0.5) -> Net 1.0
        mock_client.account.return_value = {
            "positions": [
                {"symbol": "ETHUSDT", "positionAmt": "1.5"},   # Long Side
                {"symbol": "ETHUSDT", "positionAmt": "-0.5"},  # Short Side
                {"symbol": "BTCUSDT", "positionAmt": "10.0"}   # Irrelevant symbol
            ]
        }
        
        net_pos = adapter.get_position("ETHUSDT")
        print(f"\n[Test] Hedge Mode: Long(1.5) + Short(-0.5) = {net_pos}")
        self.assertAlmostEqual(net_pos, 1.0)
        
        # Scenario 2: One-Way Mode (Single entry)
        mock_client.account.return_value = {
            "positions": [
                {"symbol": "ETHUSDT", "positionAmt": "2.5"}
            ]
        }
        net_pos = adapter.get_position("ETHUSDT")
        print(f"[Test] One-Way Mode: Pos(2.5) = {net_pos}")
        self.assertAlmostEqual(net_pos, 2.5)
        
        # Scenario 3: No Position
        mock_client.account.return_value = {
            "positions": []
        }
        net_pos = adapter.get_position("ETHUSDT")
        print(f"[Test] No Position: {net_pos}")
        self.assertAlmostEqual(net_pos, 0.0)

    def test_metrics_exist(self):
        """
        Verify that the new metrics are defined in the metrics module.
        """
        print("\n[Test] Checking Metrics...")
        self.assertTrue(hasattr(metrics, "REGIME_SCORE"), "REGIME_SCORE metric missing")
        self.assertTrue(hasattr(metrics, "STRATEGY_MODE"), "STRATEGY_MODE metric missing")
        print("✅ New metrics found.")

if __name__ == "__main__":
    unittest.main()
