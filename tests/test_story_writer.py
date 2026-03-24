"""
Tests for Phase 6: Story Writer hardening.

Covers:
- _write_line() uses structured logger on failure (not print)
- Division guards in period summaries (zero start_wealth)
- _format_benchmark() handles zero/tiny prices safely
- check_and_log_daily() initialises on first call, logs summary on day change
"""
from datetime import datetime
from unittest.mock import MagicMock, patch
from pathlib import Path

from core.story_writer import StoryWriter


def _make_writer(tmp_path: Path) -> StoryWriter:
    filepath = str(tmp_path / "story.log")
    return StoryWriter(filepath=filepath, symbol="ETHBTC", base_asset="ETH")


# ---------------------------------------------------------------------------
# _write_line tests
# ---------------------------------------------------------------------------

class TestWriteLine:
    def test_write_line_success(self, tmp_path):
        """Writes the formatted line to the file."""
        writer = _make_writer(tmp_path)
        ts = datetime(2024, 1, 15, 10, 30, 0)
        writer._write_line(ts, "📈", "TEST EVENT", "detail text")

        content = Path(writer.filepath).read_text(encoding="utf-8")
        assert "TEST EVENT" in content
        assert "detail text" in content

    def test_write_line_readonly_logs_error(self, tmp_path):
        """When the file cannot be written, log.error is called (not print)."""
        writer = _make_writer(tmp_path)
        ts = datetime(2024, 1, 15, 10, 30, 0)

        mock_log = MagicMock()
        writer._log = mock_log

        import builtins
        original_open = builtins.open
        def broken_open(path, mode="r", *args, **kwargs):
            if str(writer.filepath) in str(path) and "a" in str(mode):
                raise PermissionError("read-only filesystem")
            return original_open(path, mode, *args, **kwargs)

        with patch("builtins.open", broken_open):
            writer._write_line(ts, "📈", "TEST EVENT", "detail")

        mock_log.error.assert_called_once()
        call_kwargs = mock_log.error.call_args
        assert "Failed to write story line" in call_kwargs[0][0]

    def test_write_line_does_not_crash_on_failure(self, tmp_path):
        """A write failure must not propagate an exception (bot must keep running)."""
        writer = _make_writer(tmp_path)
        ts = datetime(2024, 1, 15, 10, 30, 0)

        import builtins
        original_open = builtins.open
        def broken_open(path, mode="r", *args, **kwargs):
            if str(writer.filepath) in str(path) and "a" in str(mode):
                raise OSError("disk full")
            return original_open(path, mode, *args, **kwargs)

        with patch("builtins.open", broken_open):
            # Should not raise
            writer._write_line(ts, "📈", "EVENT", "details")


# ---------------------------------------------------------------------------
# _format_benchmark guard tests
# ---------------------------------------------------------------------------

class TestFormatBenchmark:
    def test_zero_start_price_returns_none(self, tmp_path):
        """Returns None when period start price is zero (no HODL data yet)."""
        writer = _make_writer(tmp_path)
        result = writer._format_benchmark('daily', bot_pct=1.0, current_price=0.05)
        assert result is None

    def test_zero_current_price_returns_none(self, tmp_path):
        """Returns None when current price is zero (exchange data issue)."""
        writer = _make_writer(tmp_path)
        writer.period_start_price['daily'] = 0.05
        result = writer._format_benchmark('daily', bot_pct=1.0, current_price=0.0)
        assert result is None

    def test_valid_prices_returns_formatted_string(self, tmp_path):
        """Returns a formatted string when both prices are valid."""
        writer = _make_writer(tmp_path)
        writer.period_start_price['daily'] = 0.05
        result = writer._format_benchmark('daily', bot_pct=2.0, current_price=0.06)
        assert result is not None
        assert "Alpha:" in result
        assert "Bot:" in result


# ---------------------------------------------------------------------------
# check_and_log_daily tests
# ---------------------------------------------------------------------------

class TestCheckAndLogDaily:
    def test_first_call_initialises_no_summary(self, tmp_path):
        """On first call, sets start values but writes no summary."""
        writer = _make_writer(tmp_path)
        ts = datetime(2024, 1, 15, 10, 0, 0)

        writer.check_and_log_daily(ts, daily_pnl=0.0, current_wealth=1.0, quote_asset="BTC")

        # No summary written yet
        content = Path(writer.filepath).read_text(encoding="utf-8") if Path(writer.filepath).exists() else ""
        assert "DAILY SUMMARY" not in content
        assert writer.period_start_wealth['daily'] == 1.0

    def test_same_day_no_duplicate_summary(self, tmp_path):
        """Calling twice on the same day does not write two summaries."""
        writer = _make_writer(tmp_path)
        day1 = datetime(2024, 1, 15, 10, 0, 0)
        day1b = datetime(2024, 1, 15, 18, 0, 0)

        writer.check_and_log_daily(day1, 0.0, 1.0, "BTC")
        writer.check_and_log_daily(day1b, 0.01, 1.01, "BTC")

        content = Path(writer.filepath).read_text(encoding="utf-8") if Path(writer.filepath).exists() else ""
        assert content.count("DAILY SUMMARY") == 0  # Still no summary on same day

    def test_day_change_triggers_summary(self, tmp_path):
        """Crossing midnight triggers a DAILY SUMMARY entry."""
        writer = _make_writer(tmp_path)
        day1 = datetime(2024, 1, 15, 10, 0, 0)
        day2 = datetime(2024, 1, 16, 0, 5, 0)

        writer.check_and_log_daily(day1, 0.0, 1.0, "BTC")
        writer.check_and_log_daily(day2, 0.01, 1.01, "BTC")

        content = Path(writer.filepath).read_text(encoding="utf-8")
        assert "DAILY SUMMARY" in content

    def test_daily_summary_zero_start_wealth_no_crash(self, tmp_path):
        """log_daily_summary with zero period_start_wealth does not divide by zero."""
        writer = _make_writer(tmp_path)
        ts = datetime(2024, 1, 15, 10, 0, 0)

        writer.period_start_wealth['daily'] = 0.0
        # Should not raise
        writer.log_daily_summary(ts, daily_pnl=0.0, current_wealth=1.0, quote_asset="BTC")
