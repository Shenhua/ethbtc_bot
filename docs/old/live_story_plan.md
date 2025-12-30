# Live Story Generation Plan

## Overview
Create a real-time narrative log that tracks trading events as they happen, similar to `explain_story.py` but for live trading.

## Proposed Implementation

### Option 1: Event-Driven Story Writer (Recommended)
**How it works**: `live_executor.py` writes story events to a file as they occur.

**Advantages**:
- Real-time updates
- No polling overhead
- Accurate timestamps
- Low resource usage

**Implementation**:
1. Create `core/story_writer.py`:
```python
class StoryWriter:
    def __init__(self, filepath):
        self.filepath = filepath
        self.last_regime = None
        self.last_target = 0.0
        
    def log_event(self, timestamp, event_type, details):
        # Append to file with formatted message
        pass
    
    def check_ath(self, wealth, peak):
        # Detect and log ATH
        pass
    
    def check_regime_switch(self, score, threshold):
        # Detect and log regime changes
        pass
```

2. Integrate into `live_executor.py`:
```python
story = StoryWriter("results/live_story.txt")

# In main loop:
story.check_ath(W, equity_high)
story.check_regime_switch(regime_score, adx_threshold)
if order_placed:
    story.log_event(now, "TRADE", f"BUY {qty} @ {price}")
```

**File Output** (`results/live_story.txt`):
```
2024-12-03 15:00:00 | 🚀 Bot Started | Initial Wealth: 1.0234 BTC
2024-12-03 15:15:00 | 🔄 Regime Switch: TREND | Score: 32.5
2024-12-03 15:30:00 | 🟢 BUY | +0.25 ETH @ 0.0329 BTC
2024-12-03 16:00:00 | 🚀 NEW ATH | 1.0567 BTC
2024-12-03 18:00:00 | 🚨 SAFETY BREAKER TRIPPED | DD: -21.2%
2024-12-03 20:00:00 | 🔥 PHOENIX ACTIVATED | Resuming Trading
```

---

### Option 2: Periodic Export from Prometheus
**How it works**: Export metrics to CSV periodically, run `explain_story.py`.

**Advantages**:
- Reuses existing tool
- No code changes to `live_executor.py`

**Disadvantages**:
- Delayed updates (depends on export frequency)
- Requires external scheduler (cron)

**Implementation**:
```bash
# Cron job (every hour)
0 * * * * cd /path/to/bot && python tools/export_prometheus.py && python tools/explain_story.py results/export.csv --out results/live_story.txt
```

---

### Option 3: Hybrid (Best of Both)
**How it works**: 
- `live_executor.py` writes critical events immediately (ATH, Safety Breaker, Phoenix)
- Periodic export for full analysis

**Advantages**:
- Critical events logged instantly
- Full analysis available on demand

---

## Recommendation

**Start with Option 1** (Event-Driven Story Writer) because:
1. Real-time visibility
2. Minimal overhead
3. Easy to implement
4. Can be enhanced later

**Next Steps**:
1. Create `core/story_writer.py`
2. Integrate into `live_executor.py`
3. Add rotation logic (keep last 7 days, archive older)
4. Optional: Add web endpoint to view live story

Would you like me to implement Option 1?
