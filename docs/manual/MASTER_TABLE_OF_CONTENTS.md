# Ultimate Reference Manual — Master Table of Contents

> **Version:** 1.0 (Final)  
> **Last Updated:** 2026-01-05  
> **Target Audience:** New users and maintainers of the ETH/BTC Algorithmic Trading Bot  
> **Scope:** Complete feature, configuration, and workflow documentation

---

## How to Use This Manual

- **New Users:** Start with Chapters 1-3, then Chapter 14 (Workflows)
- **Strategy Tuning:** Focus on Chapters 4-5 and Chapter 9 (Optimization)
- **Risk Setup:** Read Chapter 6 carefully before going live
- **Troubleshooting:** Jump to Chapter 15 when issues arise
- **Quick Reference:** Use Appendix A for all configuration parameters

---

## Part I: Foundations

### [Chapter 1: Introduction & System Overview](./CHAPTER_01_INTRODUCTION.md)
- 1.1 Project Philosophy (Four Pillars)
- 1.2 Target Performance Metrics
- 1.3 Architecture Overview
- 1.4 Non-Goals (Explicitly Out of Scope)

### [Chapter 2: Installation & Deployment](./CHAPTER_02_INSTALLATION.md)
- 2.1 Prerequisites
- 2.2 Local Development Setup
- 2.3 Docker Deployment
- 2.4 Directory Structure Reference
- 2.5 First Run Checklist

### [Chapter 3: Configuration Reference](./CHAPTER_03_CONFIGURATION.md)
- 3.1 Configuration File Structure
- 3.2 Fees Configuration
- 3.3 Strategy Configuration
- 3.4 Execution Configuration
- 3.5 Risk Configuration
- 3.6 Complete Production Config Example

---

## Part II: Trading Logic

### [Chapter 4: Trading Strategies](./CHAPTER_04_STRATEGIES.md)
- 4.1 Mean Reversion Strategy
- 4.2 Trend Following Strategy
- 4.3 Meta Strategy (Regime-Adaptive)
- 4.4 Enhanced Indicators (RSI, Volume, Bollinger, HTF, Funding)

### [Chapter 5: Position Sizing](./CHAPTER_05_POSITION_SIZING.md)
- 5.1 Position Sizer Architecture
- 5.2 Static Mode
- 5.3 Volatility Targeting Mode
- 5.4 Kelly Criterion Mode
- 5.5 Comparing Sizing Modes

### [Chapter 6: Risk Management](./CHAPTER_06_RISK_MANAGEMENT.md)
- 6.1 Risk Manager Architecture
- 6.2 High Water Mark (HWM) Tracking
- 6.3 Maximum Drawdown Protection
- 6.4 Daily Loss Limits
- 6.5 Phoenix Protocol
- 6.6 Risk Modes Comparison

---

## Part III: Execution & Operations

### [Chapter 7: Execution Layer](./CHAPTER_07_EXECUTION.md)
- 7.1 Live Executor
- 7.2 Exchange Adapters (Spot & Futures)
- 7.3 Order Management
- 7.4 Precision Handling
- 7.5 Resilience & Error Handling
- 7.6 Status Server

### [Chapter 8: Backtesting Engine](./CHAPTER_08_BACKTESTING.md)
- 8.1 Backtester Architecture
- 8.2 Running a Backtest (CLI)
- 8.3 Fee Modeling & Slippage
- 8.4 Risk Simulation
- 8.5 Backtest Report & Metrics

### [Chapter 9: Walk-Forward Optimization](./CHAPTER_09_OPTIMIZATION.md)
- 9.1 WFO Concept & Philosophy
- 9.2 Mean Reversion Optimizer
- 9.3 Trend Strategy Optimizer
- 9.4 Meta Threshold Optimization
- 9.5 WFO Selection Strategies
- 9.6 Production Pipeline Cookbook

---

## Part IV: Infrastructure

### [Chapter 10: Monitoring & Observability](./CHAPTER_10_MONITORING.md)
- 10.1 Observability Architecture
- 10.2 Prometheus Metrics (30+ metrics)
- 10.3 Prometheus Configuration
- 10.4 Alert Manager (Discord/Telegram)
- 10.5 Structured Logging
- 10.6 Grafana Dashboards
- 10.7 Complete Monitoring Setup
- 10.8 Key PromQL Queries

### [Chapter 11: Data Pipeline](./CHAPTER_11_DATA_PIPELINE.md)
- 11.1 Data Pipeline Architecture
- 11.2 Downloading OHLCV Data
- 11.3 Downloading Funding Rates
- 11.4 Data Format Specification
- 11.5 Loading Data
- 11.6 Data Quality Checks

### [Chapter 12: Utility Tools](./CHAPTER_12_UTILITY_TOOLS.md)
- 12.1 Dust Sweeper
- 12.2 Config Sanity Checker
- 12.3 PnL Reconciler
- 12.4 WFO Window Selector
- 12.5 Futures Testnet Seeder
- 12.6 Additional Tools Reference

---

## Part V: Quality & Operations

### [Chapter 13: Testing & Quality Assurance](./CHAPTER_13_TESTING.md)
- 13.1 Testing Architecture
- 13.2 Running Tests
- 13.3 Phoenix Protocol Tests
- 13.4 Position Sizer Tests
- 13.5 Regression Testing (Golden Snapshots)
- 13.6 Risk Management Tests
- 13.7 Strategy Filter Tests
- 13.8 Writing New Tests

### [Chapter 14: Workflows & Recipes](./CHAPTER_14_WORKFLOWS.md)
- 14.1 First-Time Setup Workflow
- 14.2 Full Optimization Cycle
- 14.3 Testnet Deployment
- 14.4 Production Deployment
- 14.5 Adding a New Trading Pair
- 14.6 Emergency Recovery
- 14.7 Monthly Maintenance
- 14.8 Quick Reference Commands

### [Chapter 15: Troubleshooting & Edge Cases](./CHAPTER_15_TROUBLESHOOTING.md)
- 15.1 Quick Diagnosis Flow
- 15.2 Bot Startup Issues
- 15.3 Bot Running But Not Trading
- 15.4 Trades Not Filling
- 15.5 Unexpected Losses
- 15.6 Metrics Issues
- 15.7 State File Issues
- 15.8 Backtest Issues
- 15.9 Optimization Issues
- 15.10 Docker Issues
- 15.11 Edge Cases
- 15.12 Getting Help

---

## Appendixes

### [Appendix A: Configuration Reference](./APPENDIX_A_CONFIG_REFERENCE.md)
- A.1 Configuration Structure
- A.2 Fees Section (5 parameters)
- A.3 Strategy Section (40+ parameters)
- A.4 Execution Section (12 parameters)
- A.5 Risk Section (8 parameters)
- A.6 Environment Variables
- A.7 Complete Production Config Example

### [Appendix B: Glossary](./APPENDIX_B_GLOSSARY.md)
- Trading & Strategy Terms (40+ definitions)
- Technical Terms
- Abbreviations
- File Extensions

### [Appendix C: Mathematical Formulas](./APPENDIX_C_FORMULAS.md)
- C.1 Signal Generation
- C.2 Position Sizing
- C.3 Risk Metrics
- C.4 Performance Metrics
- C.5 Fee Calculations
- C.6 Volatility Formulas

---

## Quick Links

| Need to... | Go to... |
|------------|----------|
| Install the bot | [Chapter 2](./CHAPTER_02_INSTALLATION.md) |
| Configure parameters | [Chapter 3](./CHAPTER_03_CONFIGURATION.md) or [Appendix A](./APPENDIX_A_CONFIG_REFERENCE.md) |
| Understand strategies | [Chapter 4](./CHAPTER_04_STRATEGIES.md) |
| Set up risk limits | [Chapter 6](./CHAPTER_06_RISK_MANAGEMENT.md) |
| Run backtests | [Chapter 8](./CHAPTER_08_BACKTESTING.md) |
| Optimize parameters | [Chapter 9](./CHAPTER_09_OPTIMIZATION.md) |
| Deploy to production | [Chapter 14](./CHAPTER_14_WORKFLOWS.md) |
| Fix problems | [Chapter 15](./CHAPTER_15_TROUBLESHOOTING.md) |
| Look up a term | [Appendix B](./APPENDIX_B_GLOSSARY.md) |
| Find a formula | [Appendix C](./APPENDIX_C_FORMULAS.md) |

---

## Document Statistics

| Metric | Value |
|--------|-------|
| **Total Chapters** | 15 |
| **Total Appendixes** | 3 |
| **Total Files** | 18 |
| **Configuration Parameters Documented** | 70+ |
| **Prometheus Metrics Documented** | 30+ |
| **Test Files Covered** | 25 |
| **Utility Tools Documented** | 10+ |
