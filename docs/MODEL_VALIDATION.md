# Model validation

Machine learning is an **experiment**, not a dependency. The running paper session uses the
rule-based `ema_std` strategy; no model influences it today. This document records how a model
would have to earn that role, and why the current one did not.

## 1. Task definition

* Target: direction of the forward return over `horizon = 5` daily bars, expressed as
  `forward_return > round-trip cost` (cost = 2 × 10 bps fee + spread + slippage).
* Type: binary classification, `P(cost-clearing up move)`.
* Why this label: a model that predicts price direction but not *net* direction is useless
  after costs. The label includes the cost hurdle, so labels near zero are negative.
* Unit of prediction: one closed daily bar. Predictions are stored with their bar timestamp.

## 2. Features (17, all causal)

```
ret_1, ret_3, ret_7, ret_14, ret_30          trailing returns
vol_10, vol_30                                realised volatility
atr_pct                                       ATR14 / price
rsi_14                                        Wilder RSI
adx_14                                        trend strength
z65                                           (close - EMA65) / σ65
ema_dist_50, ema_dist_200                     distance to EMAs
vol_z                                         volume z-score
range_pos                                     position inside the recent range
regime_trend, regime_high_vol                 causal regime flags
```

Every feature at bar *i* is computed from bars ≤ *i*. There is no centring, no forward fill from
the future, and no target leakage (the label is shifted forward by `horizon` and excluded from
the features).

## 3. Validation protocol

```
|---- train (expanding) ----|-- purge(h) --|-- embargo(h) --|-- test (120 bars) --| ...
```

* Expanding-window walk-forward: the model for block *k* sees only bars before it.
* `horizon` bars are purged from the end of each training window, and an equal embargo separates
  training from test, so no training label overlaps a test bar.
* Minimum 700 training bars; 120-bar test blocks; at most 21 folds on the available history.
* Hyper-parameters are **fixed before evaluation** (`GradientBoostingClassifier`,
  `n_estimators=300`, `learning_rate=0.05`, `max_depth=3`, `subsample=0.9`, `random_state=42`).
  No search on the test data.
* Metrics: out-of-fold ROC AUC, Brier score, fold count, prediction count, and the economics of
  the resulting strategy versus the rule-based baseline on the same window.

## 4. Promotion gate

A model may be considered for the paper strategy only if **all** of the following hold
(`promotion_decision` in `btcbot/ml/train.py`):

1. out-of-fold AUC ≥ **0.55**;
2. at least **400** out-of-fold predictions;
3. the gated strategy beats the EMA-65 baseline on the same window **after costs**;
4. no drawdown regression beyond the configured risk tolerance;
5. the result is reproducible from the stored artifact and dataset hash.

A rejected model never replaces an accepted one. The rule-based strategy remains the default if
a model is missing, fails to load, or emits `NaN` probabilities — `MLGated` returns no signal in
that case, so a broken model cannot create trades.

## 5. Current result (2026-10-20)

| Item | Value |
|---|---|
| Model version | `gbdt_1d_h5_20261020` |
| Out-of-fold AUC | **0.533** (2,447 predictions, 21 folds) |
| Required AUC | 0.55 |
| Brier score | recorded in the model artifact |
| ML-gated strategy, same window | +7.69% net, 102 trades, max DD 1.63% |
| EMA-65 baseline, same window | +21.95% net, 21 trades, max DD 5.14% |
| Verdict | **rejected** — AUC below gate and worse net return than the baseline |

Interpretation: at an AUC of 0.533 the model is barely better than a coin flip, and its extra
trading (102 vs 21 round trips) pays more costs than the weak signal earns. This is a normal
outcome for a 5-bar direction classifier on a single asset and should be reported as such, not
tuned until it passes.

## 6. Model registry and rollback

`models` table: version, path, trained_at, horizon, feature list, metrics, status, notes.
`model_predictions` stores out-of-fold probabilities per bar so a study can be reproduced without
retraining.

Rollback is trivial because the model is not in the live path: set `paper.strategy` back to
`ema_std` (its default) and restart. Even if a future model is promoted, `MLGated` reads a
probability column; removing it produces "no signal", never a malformed order.

## 7. What would change the verdict

* A materially larger sample (more history or multiple assets) with the same protocol.
* A different label (e.g. volatility or drawdown-risk prediction) where the signal is stronger.
* Ensembling with regime information, validated on a genuinely held-out period not used for any
  previous study.
* Any of these must be evaluated with the same gate; none is implemented today.

## 8. Reproduce

```bash
./.venv/Scripts/python.exe -m btcbot.cli ml
```

Writes `reports/models/gbdt_1d_h5_<date>.joblib`, updates the `models` table, prints OOF AUC,
Brier score, fold count and the strategy comparison, and records the verdict. The verdict also
appears in `reports/experiments.md`.
