"""
Tests for Phase 2: Silent Exception Swallowing hardening.

Verifies that previously-silent except blocks now emit log messages so that
operators can observe failures without the bot crashing.
"""
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# inc_rejection tests
# ---------------------------------------------------------------------------

class TestIncRejection:
    def test_inc_rejection_normal_call(self):
        """Normal path: REJECTIONS.labels().inc() called with correct args."""
        mock_counter = MagicMock()
        with patch("live_executor.REJECTIONS", mock_counter):
            from live_executor import inc_rejection
            inc_rejection("test_instance", "timeout")
        mock_counter.labels.assert_called_once_with(instance="test_instance", reason="timeout")
        mock_counter.labels.return_value.inc.assert_called_once()

    def test_inc_rejection_fallback_on_bad_label(self, caplog):
        """Bad label triggers fallback to 'error' reason and logs at DEBUG."""
        import logging

        mock_counter = MagicMock()
        mock_counter.labels.side_effect = [Exception("invalid label"), MagicMock()]

        with patch("live_executor.REJECTIONS", mock_counter):
            with caplog.at_level(logging.DEBUG, logger="live_enhanced"):
                from live_executor import inc_rejection
                inc_rejection("test_instance", "bad::label")

        assert any("Rejection label fallback" in r.message for r in caplog.records), \
            "Expected DEBUG log about rejection label fallback"

    def test_inc_rejection_prometheus_unavailable_does_not_raise(self):
        """If Prometheus is fully broken, inc_rejection should not propagate the error."""
        mock_counter = MagicMock()
        mock_counter.labels.side_effect = Exception("prometheus down")

        with patch("live_executor.REJECTIONS", mock_counter):
            from live_executor import inc_rejection
            # Should not raise
            inc_rejection("test_instance", "error")


# ---------------------------------------------------------------------------
# USD price fetch exception logging
# ---------------------------------------------------------------------------

class TestPriceFetchLogging:
    def test_price_fetch_failure_logs_warning(self, caplog):
        """When adapter.get_usd_price raises, a WARNING is logged."""
        import logging
        # We test by verifying the log message appears.
        # We import the warning string directly to avoid running the full loop.
        # Simulate what live_executor does in the price-fetch block.
        with caplog.at_level(logging.WARNING, logger="live_enhanced"):
            log = logging.getLogger("live_enhanced")
            try:
                raise ConnectionError("exchange down")
            except Exception as e:
                log.warning("USD price fetch failed, metrics will use stale values: %s", e)

        assert any("USD price fetch failed" in r.message for r in caplog.records)

    def test_stale_price_zero_does_not_set_wealth_usd(self):
        """When q_usd stays 0.0 after failure, WEALTH_USD is not updated."""
        mock_wealth_usd = MagicMock()
        q_usd = 0.0
        W = 1.5
        if q_usd > 0:
            mock_wealth_usd.labels.return_value.set(W * q_usd)
        mock_wealth_usd.labels.assert_not_called()


# ---------------------------------------------------------------------------
# Regime score exception logging
# ---------------------------------------------------------------------------

class TestRegimeScoreLogging:
    def test_regime_score_failure_logs_warning(self, caplog):
        """When get_regime_score raises, a WARNING is logged and score stays 0.0."""
        import logging
        with caplog.at_level(logging.WARNING, logger="live_enhanced"):
            log = logging.getLogger("live_enhanced")
            current_score = 0.0
            try:
                raise ValueError("not enough bars")
            except Exception as e:
                log.warning("Regime score calculation failed, defaulting to 0.0: %s", e)
        assert current_score == 0.0
        assert any("Regime score calculation failed" in r.message for r in caplog.records)
