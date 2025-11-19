import importlib

from core import metrics


def _reset_gauges():
    """
    Helper: reload module to reset Gauge state between tests.

    Prometheus Gauges keep their value globally; reloading the module is a
    simple way to ensure a clean state across tests.
    """
    importlib.reload(metrics)


def test_mark_risk_mode_sets_dynamic_and_fixed_basis_correctly():
    _reset_gauges()

    # First: dynamic
    metrics.mark_risk_mode("dynamic")
    dynamic_val = metrics.RISK_MODE.labels(mode="dynamic")._value.get()
    fixed_val = metrics.RISK_MODE.labels(mode="fixed_basis")._value.get()
    assert dynamic_val == 1.0
    assert fixed_val == 0.0

    # Then: fixed_basis
    metrics.mark_risk_mode("fixed_basis")
    dynamic_val = metrics.RISK_MODE.labels(mode="dynamic")._value.get()
    fixed_val = metrics.RISK_MODE.labels(mode="fixed_basis")._value.get()
    assert dynamic_val == 0.0
    assert fixed_val == 1.0


def test_mark_risk_flags_sets_flags_as_expected():
    _reset_gauges()

    # Initially: no flags
    metrics.mark_risk_flags(daily_limit_hit=False, maxdd_hit=False)
    daily = metrics.RISK_FLAGS.labels(kind="daily_limit_hit")._value.get()
    maxdd = metrics.RISK_FLAGS.labels(kind="maxdd_hit")._value.get()
    assert daily == 0.0
    assert maxdd == 0.0

    # Daily limit hit only
    metrics.mark_risk_flags(daily_limit_hit=True, maxdd_hit=False)
    daily = metrics.RISK_FLAGS.labels(kind="daily_limit_hit")._value.get()
    maxdd = metrics.RISK_FLAGS.labels(kind="maxdd_hit")._value.get()
    assert daily == 1.0
    assert maxdd == 0.0

    # Max DD hit only
    metrics.mark_risk_flags(daily_limit_hit=False, maxdd_hit=True)
    daily = metrics.RISK_FLAGS.labels(kind="daily_limit_hit")._value.get()
    maxdd = metrics.RISK_FLAGS.labels(kind="maxdd_hit")._value.get()
    assert daily == 0.0
    assert maxdd == 1.0

    # Both hit (e.g., during a very bad move)
    metrics.mark_risk_flags(daily_limit_hit=True, maxdd_hit=True)
    daily = metrics.RISK_FLAGS.labels(kind="daily_limit_hit")._value.get()
    maxdd = metrics.RISK_FLAGS.labels(kind="maxdd_hit")._value.get()
    assert daily == 1.0
    assert maxdd == 1.0

def test_live_executor_imports_without_side_effect_errors():
    """
    Quick smoke test: import live_executor and ensure module import succeeds.

    This catches issues like mis-typed metric names, wrong imports, etc.
    """
    import importlib

    import live_executor  # noqa: F401

    # Force reload to mimic fresh process startup
    importlib.reload(live_executor)