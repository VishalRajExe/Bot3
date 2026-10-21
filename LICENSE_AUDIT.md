# Licence audit

Scope: code reviewed for reuse, third-party components actually shipped, and the obligations
that follow. Audit date 2026-10-21. This is an engineering record, not legal advice.

## 1. Frameworks reviewed for reuse

| Project | Licence (as published) | Copied? | Rationale |
|---|---|---|---|
| freqtrade/freqtrade | **GPL-3.0-or-later** | **No code** | Copyleft: embedding the engine would force this repository under GPL-3.0. Studied documented behaviour only (fee/slippage concepts, warm-up requirements). |
| jesse-ai/jesse | **MIT** | **No code** | Licence-compatible, but requires a database server and brings its own dashboard; the project needed a smaller, auditable engine. |
| hummingbot/hummingbot | **Apache-2.0** | **No code** | Permissive, but architected around market making and exchange connectors for many venues; wrong shape for a single-pair daily system and far heavier than needed. |
| mementum/backtrader | GPL-3.0 | No | Same copyleft concern as Freqtrade; unmaintained for the target Python version. |
| ccxt/ccxt | MIT | No | Only one public REST endpoint was needed; a full exchange abstraction would add a dependency without benefit. |

**Result: zero lines of third-party trading code are included.** All engine, risk, strategy,
metric and UI code in this repository was written for this project. Concepts (fee models,
next-bar execution, walk-forward splits, Donchian/Bollinger/EMA definitions) are public domain
techniques and are implemented from their mathematical definitions.

Where a concept was inspired by another project's documentation, it is noted in `RESEARCH.md`
with a link; no source file was ported.

## 2. Vendored frontend library

| Component | Version | Licence | Obligation | Satisfied by |
|---|---|---|---|---|
| TradingView Lightweight Charts™ | 5.0.9 | Apache-2.0 | keep copyright/licence notice; attribution | unmodified `web/vendor/lightweight-charts.standalone.production.js` retains its `@license` header; the on-chart "Charting by TradingView" attribution link is left enabled |

The file is included unmodified. Its full licence text is at
<https://www.apache.org/licenses/LICENSE-2.0>. No NOTICE file is shipped by the upstream
package beyond the header above.

## 3. Runtime dependencies (installed via pip, not vendored)

| Package | Installed | Licence | Notes |
|---|---|---|---|
| numpy | 2.5.4 | BSD-3-Clause | permissive, attribution in package metadata |
| pandas | 3.0.6 | BSD-3-Clause | |
| scipy | 1.18.1 | BSD-3-Clause | listed explicitly; also used by scikit-learn |
| scikit-learn | 1.9.1 | BSD-3-Clause | |
| joblib | 1.6.0 | BSD-3-Clause | model persistence |
| fastapi | 0.155.0 | MIT | |
| starlette | 1.9.0 | BSD-3-Clause | indirect |
| pydantic / pydantic-core | 2.14.1 / 2.50.1 | MIT | |
| uvicorn | 0.54.0 | BSD-3-Clause | |
| plotly | 7.1.0 | MIT | research charts |
| requests | 2.34.2 | Apache-2.0 | Binance REST client |
| httpx | 0.28.1 | BSD-3-Clause | test client transport |
| pytest | 9.1.1 | MIT | tests only |

Permissive licences (MIT/BSD/Apache-2.0) impose no copyleft obligation on this repository. They
do require preserving their notices when redistributing a bundled environment; `pip`-installed
packages already carry their own licence files. No GPL/AGPL package is a dependency of the
runtime.

Python itself is distributed under the PSF Licence.

## 4. Data and content

| Source | Terms | Use here |
|---|---|---|
| Binance public REST (klines, ticker) | Binance API terms of use; public market data, no key, rate-limited | read-only market data for research and paper trading; no order endpoints are called |
| Alternative.me Fear & Greed API | free public JSON endpoint, attribution expected | one request per refresh; value and timestamp stored with source |
| RSS headlines (CoinDesk, Cointelegraph, The Block) | publisher feed terms | headlines + link stored as context only; no article text is reproduced |
| TradingView Lightweight Charts | Apache-2.0 | chart rendering only |

No scraped social data, no private data, no personal data, no exchange credentials.

## 5. Project licence

No licence file has been added to this repository. Until the owner chooses one, it remains
"all rights reserved" by default. Given the dependency set (all permissive) and the absence of
copied GPL code, MIT or Apache-2.0 would both be compatible choices; Apache-2.0 would also
propagate the Lightweight Charts attribution requirement cleanly.

## 6. Residual risks

1. Framework licences were read from the repositories at review time; they can change. Re-verify
   before publishing derived work.
2. Binance terms may restrict commercial redistribution of market data; this project stores data
   locally for research and does not redistribute it.
3. If any GPL component is later added (for example by importing a Freqtrade strategy file),
   the licence of the whole distributed work changes. `scripts/` contains no such code today.
