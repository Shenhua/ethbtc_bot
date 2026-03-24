"""
Tests for Phase 1: State Persistence hardening.

Covers:
- load_state() behaviour for missing, corrupted, and backed-up state files
- save_state() atomic write and .bak rotation (no silent fallback directory)
- RiskState field validators for NaN, Inf, negative equity_high, malformed timestamps
"""
import json
import pytest
from pathlib import Path
from unittest.mock import patch

from core.models.state import RiskState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


# ---------------------------------------------------------------------------
# load_state tests
# ---------------------------------------------------------------------------

class TestLoadState:
    def test_load_state_missing_file(self, tmp_path):
        """Returns {} and logs INFO when file does not exist."""
        from live_executor import load_state
        result = load_state(str(tmp_path / "nonexistent.json"))
        assert result == {}

    def test_load_state_valid_json(self, tmp_path):
        """Returns parsed dict for a valid state file."""
        from live_executor import load_state
        p = tmp_path / "state.json"
        _write_json(p, {"session_start_W": 1.5, "risk_equity_high": 1.5})
        result = load_state(str(p))
        assert result["session_start_W"] == 1.5

    def test_load_state_corrupted_json_no_backup_raises(self, tmp_path):
        """Raises JSONDecodeError when file is corrupt and no backup exists."""
        from live_executor import load_state
        p = tmp_path / "state.json"
        p.write_text("{ this is not valid json !!!", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            load_state(str(p))

    def test_load_state_corrupted_json_with_backup(self, tmp_path):
        """Falls back to .bak when main file is corrupt."""
        from live_executor import load_state
        p = tmp_path / "state.json"
        bak = tmp_path / "state.json.bak"
        p.write_text("{ bad json", encoding="utf-8")
        _write_json(bak, {"session_start_W": 2.0})
        result = load_state(str(p))
        assert result["session_start_W"] == 2.0

    def test_load_state_other_error_raises(self, tmp_path):
        """Raises non-JSON exceptions (e.g. permission errors) instead of swallowing."""
        from live_executor import load_state
        p = tmp_path / "state.json"
        _write_json(p, {"k": "v"})
        # Simulate a read error by patching open
        import builtins
        original_open = builtins.open
        def broken_open(path, *args, **kwargs):
            if str(p) in str(path):
                raise PermissionError("access denied")
            return original_open(path, *args, **kwargs)
        with patch("builtins.open", broken_open):
            with pytest.raises(PermissionError):
                load_state(str(p))


# ---------------------------------------------------------------------------
# save_state tests
# ---------------------------------------------------------------------------

class TestSaveState:
    def test_save_state_creates_file(self, tmp_path):
        """Creates state file with correct content."""
        from live_executor import save_state
        p = tmp_path / "state.json"
        save_state(str(p), {"session_start_W": 1.0})
        assert p.exists()
        with open(p) as f:
            data = json.load(f)
        assert data["session_start_W"] == 1.0

    def test_save_state_creates_backup(self, tmp_path):
        """Creates .bak of existing state before overwriting."""
        from live_executor import save_state
        p = tmp_path / "state.json"
        _write_json(p, {"session_start_W": 1.0})
        save_state(str(p), {"session_start_W": 2.0})
        bak = tmp_path / "state.json.bak"
        assert bak.exists()
        with open(bak) as f:
            bak_data = json.load(f)
        assert bak_data["session_start_W"] == 1.0  # old value preserved in .bak

    def test_save_state_no_fallback_directory(self, tmp_path):
        """Raises OSError on write failure — no silent fallback to another directory."""
        from live_executor import save_state
        p = tmp_path / "state.json"
        import builtins
        original_open = builtins.open
        def broken_open(path, mode="r", *args, **kwargs):
            if str(p) in str(path) and "w" in str(mode):
                raise OSError(30, "Read-only file system")  # errno 30 = EROFS
            return original_open(path, mode, *args, **kwargs)
        with patch("builtins.open", broken_open):
            with pytest.raises(OSError):
                save_state(str(p), {"k": "v"})
        # Ensure no fallback directory was created
        fallback = Path.cwd() / "run_state"
        assert not (fallback / "state.json").exists()

    def test_save_state_atomic_tmp_cleanup(self, tmp_path):
        """After successful save, .tmp file is removed."""
        from live_executor import save_state
        p = tmp_path / "state.json"
        save_state(str(p), {"k": "v"})
        tmp = tmp_path / "state.json.tmp"
        assert not tmp.exists()


# ---------------------------------------------------------------------------
# RiskState validator tests
# ---------------------------------------------------------------------------

class TestRiskStateValidators:
    def test_nan_equity_high_coerced_to_zero(self):
        state = RiskState(equity_high=float("nan"))
        assert state.equity_high == 0.0

    def test_inf_equity_high_coerced_to_zero(self):
        state = RiskState(equity_high=float("inf"))
        assert state.equity_high == 0.0

    def test_negative_inf_equity_high_coerced_to_zero(self):
        state = RiskState(equity_high=float("-inf"))
        assert state.equity_high == 0.0

    def test_negative_equity_high_coerced_to_zero(self):
        state = RiskState(equity_high=-5.0)
        assert state.equity_high == 0.0

    def test_valid_equity_high_preserved(self):
        state = RiskState(equity_high=1.5)
        assert state.equity_high == 1.5

    def test_zero_equity_high_preserved(self):
        """Zero is valid (means uninitialised) — should stay 0.0."""
        state = RiskState(equity_high=0.0)
        assert state.equity_high == 0.0

    def test_malformed_maxdd_hit_ts_coerced_to_none(self):
        state = RiskState(maxdd_hit_ts="not-a-timestamp-99:99:99")
        assert state.maxdd_hit_ts is None

    def test_valid_maxdd_hit_ts_preserved(self):
        ts = "2024-01-15T10:30:00+00:00"
        state = RiskState(maxdd_hit_ts=ts)
        assert state.maxdd_hit_ts == ts

    def test_none_maxdd_hit_ts_preserved(self):
        state = RiskState(maxdd_hit_ts=None)
        assert state.maxdd_hit_ts is None
