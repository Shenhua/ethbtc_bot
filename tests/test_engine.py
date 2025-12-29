import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from core.engine import BotEngine
from core.models.state import BotState, RiskState

class MockConfig:
    def __init__(self):
        self.symbol = "ETHBTC"
        self.strategy = MagicMock()
        self.strategy.strategy_type = "mean_reversion"
        self.strategy.model_dump.return_value = {
            "step_allocation": 0.33,
            "adx_threshold": 25.0
        }
        self.execution = MagicMock()
        self.execution.interval = 3600
        self.execution.poll_sec = 10
        self.execution.exchange_type = "spot"
        self.risk = MagicMock()
        self.risk.drawdown_reset_days = 0

@pytest.fixture
def mock_deps():
    return {
        "config": MockConfig(),
        "data_svc": MagicMock(),
        "order_svc": MagicMock(),
        "risk_mgr": MagicMock(),
        "story": MagicMock(),
        "alerter": MagicMock(),
        "bot_state": BotState(risk=RiskState()),
        "instance_name": "test_bot"
    }

def test_engine_initialization(mock_deps):
    engine = BotEngine(**mock_deps)
    assert engine.instance_name == "test_bot"
    assert engine.mr_params["step_allocation"] == 0.33

def test_process_bar_skips_if_no_data(mock_deps):
    mock_deps["data_svc"].get_closed_klines.return_value = None
    engine = BotEngine(**mock_deps)
    
    engine.process_bar(1700000000, 1700000005)
    
    mock_deps["order_svc"].get_account_state.assert_not_called()

def test_process_bar_full_flow(mock_deps):
    # Setup mocks
    df = pd.DataFrame({"close": [0.05, 0.06]})
    mock_deps["data_svc"].get_closed_klines.return_value = df
    mock_deps["order_svc"].get_account_state.return_value = {
        "W": 1000.0, "cur_w": 0.5, "quote_bal": 500.0, "current_position": 8.33
    }
    
    engine = BotEngine(**mock_deps)
    
    # Mock strategy generation
    with patch("core.engine.EthBtcStrategy") as mock_strat_cls:
        mock_strat = MagicMock()
        mock_strat.generate_positions.return_value = pd.DataFrame({
            "target_w": [0.6],
            "regime_score": [20.0]
        })
        mock_strat_cls.return_value = mock_strat
        
        engine.process_bar(1700000000, 1700000005)
        
    # Verify calls
    mock_deps["risk_mgr"].update.assert_called()
    assert engine.bot_state.last_bar_close == "1700000000"
    assert engine.bot_state.last_regime == "CHOP"

def test_apply_risk_overrides_maxdd(mock_deps):
    mock_deps["bot_state"].risk.maxdd_hit = True
    engine = BotEngine(**mock_deps)
    
    # Even if strategy wants 0.5, risk forces 0.0
    safe_w = engine._apply_risk_overrides(0.5, 0.5, 1000.0, pd.Timestamp.now())
    assert safe_w == 0.0
