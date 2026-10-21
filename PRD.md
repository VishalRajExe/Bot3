# PRD — BTC research & paper-trading system

Status: implemented, tested, running in paper mode. Document version 1.0 (2026-10-21).

---

## 1. Problem

Retail crypto tooling makes it easy to produce a backtest that looks profitable and hard to
learn whether a strategy has any edge after costs. The goal of this project is the opposite:
a small, auditable system that can **measure** whether BTC/USDT spot strategies survive fees,
slippage, out-of-sample data and hard risk limits — and that never pretends a simulation is a
guarantee.

## 2. Objectives

1. Obtain clean, verifiable BTC/USDT history and keep obtaining live public quotes.
2. Implement several competing strategy families behind one interface.
3. Backtest with next-bar execution, explicit costs and genuine benchmarks.
4. Enforce risk limits centrally, independently of any strategy or model.
5. Run a live-data paper session whose state survives restarts.
6. Record every trade in a permanent journal and learn from groups of losses.
7. Test whether ML and sentiment add out-of-sample value, and say so when they do not.
8. Provide a dashboard that shows what the bot is doing and why.

Out of scope: real-money execution, leverage, margin, shorting, multi-user hosting,
portfolio optimisation across assets, order-book simulation, HFT.

## 3. Users and use cases

* **Researcher** — runs comparisons and experiments, inspects per-trade CSV and equity curves,
  checks whether a change survived out-of-sample validation.
* **Operator** — starts/pauses/stops the paper session, watches equity, position, risk state and
  data freshness.
* **Reviewer** — reads `RESEARCH.md`, `reports/experiments.md` and the test suite to check that
  claims match artifacts.

## 4. Functional requirements and status

| # | Requirement | Status | Evidence |
|---|---|---|---|
| F1 | Download BTC/USDT 1d and 4h candles from a public source | Done | 3,351 daily + 20,093 4h bars; `btcbot/cli.py fetch` |
| F2 | Validate OHLCV, detect gaps/duplicates/stale bars, record provenance | Done | `tests/test_data.py`; `data_quality_checks` rows |
| F3 | Strategy interface with entry/exit/stop/reason/params/version | Done | `btcbot/strategies/base.py` |
| F4 | At least four strategy families plus the EMA-65 baseline | Done | six strategies registered |
| F5 | Backtest with fees, spread, slippage, next-bar fills | Done | `btcbot/backtest/engine.py`, `execution/paper.py` |
| F6 | Benchmarks: buy & hold, cash, simple strategy | Done | `benchmark_json` in every run |
| F7 | Centralised risk sizing and circuit breakers | Done | `btcbot/risk.py`, `tests/test_risk.py` |
| F8 | Paper execution against live quotes, separate from any live path | Done | `btcbot/paper.py`; no live code exists |
| F9 | Persist orders, fills, trades, equity, risk events | Done | SQLite schema in `btcbot/db.py` |
| F10 | Journal with rationale, regime, R-multiple, MFE/MAE, findings | Done | `btcbot/journal.py`, Journal tab |
| F11 | Controlled experiments with OOS acceptance rules | Done | `btcbot/experiments.py`, `reports/experiments.md` |
| F12 | Walk-forward ML with promotion gate | Done (model rejected) | `btcbot/ml/train.py`, `docs/MODEL_VALIDATION.md` |
| F13 | Sentiment features and validation | Done (no benefit found) | `btcbot/sentiment/*`, `reports/experiments.md` |
| F14 | Dashboard: market, chart, portfolio, research, journal, learning, controls | Done | `web/`, screenshots in `reports/screenshots/` |
| F15 | Automated tests incl. regression tests for found defects | Done | 103 tests |
| F16 | Real-money trading | **Explicitly not implemented** | config validation rejects it |

## 5. Non-functional requirements

| # | Requirement | Target | Result |
|---|---|---|---|
| N1 | Reproducibility | a run stores dataset hash, costs, risk, params, period | `backtest_runs` row + `metrics.json` |
| N2 | No look-ahead | signal at bar *i* uses only bars ≤ *i*; fills at *i+1* open | `tests/test_strategies.py`, `tests/test_backtest.py` |
| N3 | Restart safety | state reload, no duplicate orders | `tests/test_paper.py` |
| N4 | No secrets | no credentials required or stored | config + `.env.example` |
| N5 | Performance | full 9-year 1d backtest in seconds | 1.13 s for 3,285 bars |
| N6 | Failure handling | network errors logged, session keeps state, no blind trading | `poll_once` error paths |
| N7 | Observability | structured log + `/api/health` + `/api/paper` | `logs/`, dashboard badges |

## 6. Acceptance criteria (final evaluation)

| Criterion | Result |
|---|---|
| Existing project inspected before changes | Yes — empty workspace, greenfield |
| Strategy materials incorporated | No originals existed; EMA-65 rules reconstructed and documented as such |
| Architecture and framework decision documented | Yes — `RESEARCH.md`, `Architecture.md` |
| Licences reviewed | Yes — `LICENSE_AUDIT.md`, `THIRD_PARTY_NOTICES.md` |
| Market data obtained and validated | Yes — 3,351 daily bars, 0 gaps |
| At least one strategy emits correctly timed signals | Yes — next-bar-open, tested |
| Multiple strategies compared | Yes — 6 strategies, comparison table |
| Backtests produce inspectable results | Yes — 80 runs, CSV/JSON/HTML per run |
| Costs and realistic execution included | Yes — 10 bps fee + 2 bps spread + 5 bps slippage per side |
| Position sizing and hard risk limits work | Yes — tested, incl. breaker behaviour |
| Paper orders execute and update persistent state | Yes — 1 order / 1 fill in the live session; full lifecycle covered by tests |
| Trades appear on charts with correct timestamps | Yes — paper marker + stop line; backtest markers windowed |
| Trade history survives a restart | Yes — verified across two server restarts |
| Tests cover risk and execution | Yes — 103 tests |
| Failed experiments remain visible | Yes — rejected/inconclusive verdicts kept in `reports/experiments.md` |
| System starts in paper mode via documented commands | Yes — `btcbot serve --paper` / `paper` |
| Real-money execution disabled | Yes — no code path, config rejects enabling |
| Final report separates done from not done | Yes — README §9–10, final report |

## 7. Success metrics that are *not* profitability

* Backtest throughput: 3,285 daily bars in 1.13 s; a poll evaluates one closed bar plus
  warm-up (~0.2 s of local compute, network excluded).
* Data quality: zero missing daily bars; 16 genuine exchange outages on 4h.
* Test suite green and regression tests added for every defect found during development
  (float-floor sizing, risk metadata overwrite, replay FK collision, replay forced exit,
  static-asset caching, hidden tab rendering, grid overflow).
* Every claim in reports traceable to a stored run id.

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Overfitting parameters | fixed parameter grids, OOS split, verdicts recorded before looking at results |
| Look-ahead | causal indicators, shifted channels, next-bar fills, unit tests |
| Cost underestimation | fee + spread + slippage on both sides; buy & hold charged entry costs |
| Silent failure | session marked stopped on error paths, `/api/health`, log file |
| False confidence | every report states simulation-only, no profitability claim |
