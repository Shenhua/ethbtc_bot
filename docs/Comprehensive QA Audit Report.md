1. Logic vs. Goal Discrepancies
	•	Goal: Backtest/live config parity (“single source of truth” for strategy params; parity checks exist).  ￼
Reality: live_executor.py imports merge_strategy_params from core.strategy_factory then redefines a local function with the same name, shadowing the imported one. That makes parity fragile (and likely false) because live execution can silently diverge from factory/backtest behavior.  ￼  ￼
Severity: Critical
Fix Required: 🔴 OBJECTIVE FLAW — Eliminate the duplicate merge_strategy_params definition in live_executor.py; enforce one canonical path (factory) and fail fast if parity drift is detected at runtime (not just via scripts).
	•	Goal: Robust alerting to Discord/Telegram (rate-limit handling, retries).
Reality: core/alert_manager.py calls time.sleep(...) in both Discord and Telegram retry/error paths but never imports time. Any path that hits rate limiting (429) or exception handling will throw NameError and kill alerting exactly when you need it.  ￼
Severity: Critical
Fix Required: 🔴 OBJECTIVE FLAW — Add the missing import and add tests to prove alerting doesn’t crash on 429/network exceptions.
	•	Goal: Maker execution that is “safe” (post-only), with controlled repricing and reliable cancellation.
Reality: In core/twap_maker.py, if check_order() / cancel() fails, the code logs and breaks out of the loop. That can leave a live order orphaned on the exchange (especially if failure happens after placement but before cancel confirmation).  ￼
Severity: Critical
Fix Required: 🔴 OBJECTIVE FLAW — Ensure best-effort cleanup: on any exception after an order ID exists, attempt cancel with bounded retries and confirm final order state before returning.
	•	Goal: Funding-rate monitoring is part of safety (docs explicitly list it).  ￼
Reality: Funding safety is soft-fail open: funding fetch is wrapped in a broad try/except and defaults to allow = True if anything fails. If the adapter doesn’t support funding (or Binance errors), the bot will proceed as if conditions are safe.  ￼  ￼
Severity: Major (can become Critical depending on risk appetite)
Fix Required: 🔴 OBJECTIVE FLAW — Make funding safety “fail closed” (configurable): if funding safety is enabled and funding can’t be fetched/parsed, block entries (or all trading) and alert.
	•	Goal: Risk controls: daily loss limit + max drawdown + “Phoenix” recovery.  ￼
Reality: Phoenix recovery is tightly coupled to plan contents (score signals), and the code only triggers Phoenix reset if maxdd_hit is set and conditions are met using current_score derived from the current plan. If strategy calculation fails or the plan lacks the expected score field, Phoenix recovery can stall indefinitely (locked-out state).  ￼  ￼
Severity: Major
Fix Required: 🔴 OBJECTIVE FLAW — Decouple Phoenix gating from fragile plan fields; use a stable, validated metric source (or explicit “phoenix_score_provider”) and handle “no-plan” ticks deterministically.
	•	Goal: Clean execution adapter contract (unified interface).
Reality: ExchangeAdapter defines get_funding_rate() as required.  ￼ But the system also supports Spot vs Futures adapters, and funding is inherently Futures-specific; this mismatch drives the “fail-open” behavior above.
Severity: Minor → Major (depends how often spot mode is used with funding checks enabled)
Fix Required: 🟡 SUBJECTIVE IMPROVEMENT — Split interfaces (SpotAdapter, FuturesAdapter) or make funding optional with explicit capability flags so the caller can decide to fail closed/open safely.

⸻

2. “Mental Sandbox” Findings

Workflow 1 — Live trading loop (plan → risk gate → execute → persist)
	•	Scenario: Happy path
Current Behavior: Loads config, merges params, builds strategy, computes plan, checks risk gates (spread/funding), executes maker/taker orders, updates state and metrics.  ￼  ￼
Expected Behavior: Same, but must be deterministic and parity-consistent with backtest.
	•	Scenario: “Backtest/live parity drift”
Current Behavior: live_executor can run with a locally-defined merge_strategy_params that differs from core.strategy_factory.merge_strategy_params, making the live plan potentially inconsistent with what backtests/optimizers validate.  ￼  ￼
Expected Behavior: Exactly one canonical merge path; if parity scripts exist, runtime should also detect/abort on drift.
	•	Scenario: Funding endpoint fails / adapter lacks funding support
Current Behavior: Funding fetch exception → logs warning → allow = True → trades proceed.  ￼
Expected Behavior: If funding safety is enabled, block entries and alert (or block all trading if configured).
	•	Scenario: State persistence fails (disk full / permissions)
Current Behavior: Not proven safe here: if save throws, depending on exception handling, you risk “trading with stale state” next loop (double-orders, wrong position).
Expected Behavior: Fail closed: stop trading, alert, and require operator intervention if state can’t be committed.

⸻

Workflow 2 — Maker chase execution (place → wait → check → cancel/reprice)
	•	Scenario: Happy path
Current Behavior: Place post-only limit, wait, check fills, cancel & reprice until filled or retries exhausted.  ￼
Expected Behavior: Same, plus strict guarantees: you never return while an order is unknowingly still open.
	•	Scenario: Network drop during check_order() or cancel()
Current Behavior: Logs “Order check/cancel failed”, then breaks. If the order is still live, the function exits without confirming it’s closed.  ￼
Expected Behavior: Bounded retries + reconcile final status; if cannot confirm closure, quarantine trading and alert.
	•	Scenario: Partial fills + cancel failure
Current Behavior: High risk of inconsistent filled quantity accounting vs real exchange state (local thinks “done”; exchange still has remainder working).  ￼
Expected Behavior: Robust reconciliation loop: query final status until terminal or timeout triggers fail-safe shutdown.

⸻

Workflow 3 — Alerting (Discord/Telegram)
	•	Scenario: Discord 429 rate limit
Current Behavior: Code intends to sleep/retry, but throws NameError due to missing time import → alerting path crashes.  ￼
Expected Behavior: Retry with backoff, never crash caller; return a structured failure result and increment metrics.
	•	Scenario: Telegram network exception
Current Behavior: Same: exception path calls time.sleep without import → crash.  ￼
Expected Behavior: Non-fatal, logged, with bounded retry.

⸻

3. The “Matrix of Pain” (Test Plan)

Component	Scenario	Input Data	Expected Outcome	Type (Unit/E2E)
Config/Parity	Shadowed merge fn causes drift	live_executor.py contains local merge_strategy_params	CI fails + runtime self-check fails; bot refuses to start	Unit/CI
Config/Parity	Factory merge vs live merge mismatch	Same config fed through both merge paths	Identical merged config or hard failure	Unit
AlertManager	Discord 429	Mock response status_code=429	No crash; retries; returns failure after N attempts	Unit
AlertManager	Telegram exception	requests.post raises Timeout	No crash; retries; emits warning	Unit
AlertManager	Malformed webhook URL	DISCORD_WEBHOOK_URL="not a url"	No crash; fails gracefully; increments error metric	Unit
Funding Gate	Funding fetch fails	adapter raises NotImplementedError	Fail closed (no entries) + alert	Integration
Funding Gate	Funding extreme	funding = +10.0 vs limits	Entries blocked, state unchanged	Unit
Risk Engine	Daily loss trigger boundary	equity drop exactly equals threshold	Deterministic behavior at boundary (define inclusive/exclusive)	Unit
Phoenix	No plan / missing score	plan=None or no regime_score	Phoenix logic deterministic; no infinite lockout	Unit
Phoenix	Reset after wait + score	now - maxdd_hit_ts > wait + score >= threshold	Resets lockout exactly once; persists state	Integration
Execution (Maker)	cancel/check throws mid-loop	check_order raises	Bot confirms order terminal OR stops trading + alerts	Integration
Execution (Maker)	partial fill + cancel fails	check_order returns partial, then cancel raises	No orphan order; reconciled final filled qty	Integration
Execution (Maker)	max reprices hit	max_reprices=0	Returns cleanly; no open orders; emits metric	Unit
Execution (Taker)	spread too wide	spread_bps > max_spread_bps_for_taker	Taker blocked; no order placed	Unit
Execution (Taker)	market order rejected	adapter.market_order raises	Retry policy respected; bot doesn’t loop forever	Integration
Adapter Contract	Missing get_funding_rate in spot	spot adapter instance	Capability detection prevents calling unsupported method	Unit
State	Disk write failure	open() raises OSError	Trading halts; alert emitted; process exits non-zero	E2E
State	Corrupted state file on load	invalid JSON	Bot refuses start or resets safely with explicit operator flag	E2E
Metrics/Status	Status server thread safety	rapid updates + concurrent reads	No exceptions; JSON always valid	Unit
Metrics	Metric labels explode	random symbol names / modes	Bounded label cardinality	Unit
Data	Klines empty	adapter returns []	No trades; retries data fetch; alerts after threshold	Integration
Data	NaN in prices	close=nan	Guard clause triggers; no orders sent	Unit
Strategy	Output plan negative qty	target_qty=-1	Reject / normalize; never place negative sizes	Unit
Precision	Rounding step/tick edge	step=0 or tick=0	Doesn’t floor-divide by zero; uses passthrough safely	Unit
Docker/Runtime	Two bots share same state path	same mounted /data/state.json	No state corruption; lock or per-instance separation	E2E
Security	Env secrets leakage	log capture while starting	No keys printed in logs	Security Unit
Regression	Order orphan detection	simulated exchange keeps open order	Bot detects mismatch next loop, cancels, quarantines	E2E


⸻

4. Recommendations for Refactoring
	•	Untestable Code (High Risk / SRP Violations)
	•	🔴 live_executor.py is doing configuration, strategy construction, risk gating, execution, state persistence, metrics, and HTTP status in one place. That’s not “hard to mock”; it’s hard to reason about, and it hides safety invariants (like “never exit with an unknown open order”).
	•	🔴 Maker execution flow spans multiple concerns (timing, order lifecycle, reconciliation). The failure mode in maker_chase proves the lifecycle invariants aren’t centralized.  ￼
	•	Hardening Steps (guards that must be added now)
	•	🔴 Runtime parity assertion: assert that live_executor is using core.strategy_factory.merge_strategy_params (or remove the local function entirely).  ￼
	•	🔴 Alerting must never crash the caller: catch all exceptions inside alert senders and return status objects; add missing import.  ￼
	•	🔴 Execution invariant: after any order placement, you must have a “terminal state confirmation” (FILLED/CANCELED/EXPIRED/REJECTED) before returning from execution functions; current code can break out early.  ￼
	•	🔴 Funding gate should be configurable fail-open/fail-closed; defaulting to allow=True on exception is unsafe for a “safety feature”.  ￼

⸻

5. Calibration & Reality Check (CRITICAL)

🔴 OBJECTIVE FLAWS (Must Fix)
	1.	AlertManager missing time import → runtime crash on retry/error paths.  ￼
	2.	maker_chase breaks on check/cancel exception → potential orphan live orders.  ￼
	3.	live_executor shadowing imported merge_strategy_params → live/backtest parity can be false without visibility.  ￼  ￼
	4.	Funding safety fails open on exception → “safety feature” can silently disable.  ￼

🟡 SUBJECTIVE IMPROVEMENTS (Nice to have, not “broken”)
	1.	Split adapter interfaces or add explicit capability flags so callers don’t rely on exceptions to detect feature support.  ￼
	2.	Decouple Phoenix reset logic from fragile plan fields (still a real reliability risk, but the exact correct business rule isn’t fully specified in-code).  ￼