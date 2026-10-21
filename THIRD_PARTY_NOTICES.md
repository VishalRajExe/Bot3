# Third-party notices

This project contains no third-party trading-engine source code. The components below are
either vendored in the repository or installed as Python packages.

## Vendored

### TradingView Lightweight Charts™ 5.0.9

File: `web/vendor/lightweight-charts.standalone.production.js` (unmodified)

```
TradingView Lightweight Charts™ v5.0.9
Copyright (c) 2025 TradingView, Inc.
Licensed under the Apache License, Version 2.0
https://www.apache.org/licenses/LICENSE-2.0
```

The on-chart attribution link ("Charting by TradingView") is left enabled as required by the
library's licence terms.

## Python dependencies

Installed with `pip install -r requirements.txt`; each package ships its own licence file in
`site-packages`. Summary of licences:

| Package | Licence |
|---|---|
| numpy | BSD-3-Clause |
| pandas | BSD-3-Clause |
| scipy | BSD-3-Clause |
| scikit-learn | BSD-3-Clause |
| joblib | BSD-3-Clause |
| fastapi | MIT |
| starlette | BSD-3-Clause |
| pydantic, pydantic-core | MIT |
| uvicorn | BSD-3-Clause |
| plotly | MIT |
| requests | Apache-2.0 |
| httpx | BSD-3-Clause |
| pytest | MIT |

Python is distributed under the Python Software Foundation Licence.

## Market data and content

* BTC/USDT OHLCV and quotes: Binance public REST API. Data is stored locally for research and
  is not redistributed with this repository.
* Fear & Greed Index: Alternative.me (`https://api.alternative.me/fng/`). Values are stored with
  their source and timestamp; attribution is shown in the dashboard's Sentiment tab.
* News headlines and links: publisher RSS feeds. Only headline text and the link are stored;
  no article content is reproduced.

## Frameworks reviewed but not used

Freqtrade (GPL-3.0-or-later), Jesse (MIT) and Hummingbot (Apache-2.0) were evaluated and are not
included, imported or vendored. See `LICENSE_AUDIT.md`.
